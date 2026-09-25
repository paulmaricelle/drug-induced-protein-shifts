# src/pairs/motor_pairs.py
"""Méthode 3 (Section 4.2) : paires de comparateurs par prototypes k-means sur
les représentations MOTOR z0 blanchies.

Protocole, Section 4.2 : « The third compares whitened MOTOR representations
between cohorts after dividing each cohort into clusters by k-means, so that a
drug with several indications is matched on the cluster that corresponds to the
shared indication. »

Chaîne de traitement (aucune boucle Python sur les patients) :
  1. `fit_reference_whitener` : blanchiment global ajusté sur un échantillon
     de patients tiré dans les cohortes éligibles (pondération paramétrable).
  2. `fit_cohort_prototypes` : k-means par cohorte dans l'espace blanchi
     (sous-échantillonnage pour l'ajustement, affectation de tous les patients,
     fusion des clusters trop petits). Chaque cluster porte son poids (fraction
     de la cohorte), son effectif et sa variance intra-cluster.
  3. `PrototypeBank` + `cluster_distances` : distances entre tous les
     prototypes de toutes les cohortes, par blocs vectorisés O(C²·k²), puis
     min-linkage sur les clusters éligibles de chaque couple de cohortes.
  4. `select_pairs` : règle de décision (seuil et/ou top-N par molécule).

Les représentations sont fournies par un chargeur injectable
`EmbeddingLoader(drug_id) -> np.ndarray (N, 768)` aligné sur
`stanford_index.parquet`. Ce module ne lit jamais d'identifiant patient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Iterable, Literal
import warnings

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

# Chargeur injectable : drug_id -> matrice (N, D) alignée sur stanford_index,
# ou None si les représentations de la cohorte sont absentes.
EmbeddingLoader = Callable[[int], "np.ndarray | None"]

WhiteningMode = Literal["pca", "zca", "center", "none"]
DistanceMetric = Literal["mahalanobis", "mahalanobis_debiased", "cosine"]
PairRule = Literal["threshold", "topn", "threshold_topn", "mutual_topn"]


# =============================================================================
# 0. Chargement des représentations
# =============================================================================
def npy_embedding_loader(
    cohorts_dir: str | Path,
    filename: str = "stanford_motor_z0.npy",
    mmap: bool = True,
) -> EmbeddingLoader:
    """Chargeur par défaut : `<cohorts_dir>/cohort_<id>/<filename>`.

    Convention de `DrugCohort.save`. En mode mmap, l'échantillonnage de lignes
    ne lit que les pages nécessaires. À remplacer par un autre chargeur si le
    format de sortie de l'extraction MOTOR diffère (parquet, fichier unique…).
    """
    base = Path(cohorts_dir)

    def _load(drug_id: int) -> np.ndarray | None:
        f = base / f"cohort_{int(drug_id)}" / filename
        if not f.exists():
            return None
        return np.load(f, mmap_mode="r" if mmap else None)

    return _load


def store_embedding_loader(
    cohorts_dir: str | Path,
    store_dir: str | Path,
    drug_ids: Iterable[int],
    rep_col: str = "z",
) -> EmbeddingLoader:
    """Chargeur sur le magasin central dédupliqué de
    scripts/extract_motor_representations.py (`store_<ancrage>/chunk_*.parquet`,
    colonnes person_id, t0, z[768]).

    Une seule jointure (person_id, t0) entre les index de toutes les cohortes et
    le magasin, puis indexation numpy par cohorte : pas de relecture du magasin
    par cohorte. Mémoire : ~3 Go par million d'instances (float32, 768).
    Ordre des lignes = stanford_index ; inclusions sans représentation = NaN.
    """
    base, sdir = Path(cohorts_dir), Path(store_dir)
    shards = sorted(sdir.glob("chunk_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"Aucun fragment chunk_*.parquet dans {sdir}")
    store = pl.concat(
        [pl.read_parquet(f, columns=["person_id", "t0", rep_col]) for f in shards]
    ).unique(subset=["person_id", "t0"], keep="first")
    Zs = store[rep_col].to_numpy().astype(np.float32, copy=False)
    keys = store.select(
        pl.col("person_id").cast(pl.Int64), pl.col("t0").cast(pl.Date)
    ).with_row_index("_store_row")
    del store

    frames = []
    for cid in drug_ids:
        f = base / f"cohort_{int(cid)}" / "stanford_index.parquet"
        if f.exists():
            frames.append(
                pl.read_parquet(f, columns=["person_id", "t0"])
                .select(pl.col("person_id").cast(pl.Int64), pl.col("t0").cast(pl.Date))
                .with_row_index("_row")
                .with_columns(pl.lit(int(cid), dtype=pl.Int64).alias("drug_id"))
            )
    idx = pl.concat(frames).join(keys, on=["person_id", "t0"], how="left")
    rows: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for (cid,), g in idx.group_by(["drug_id"]):
        m = g.filter(pl.col("_store_row").is_not_null())
        rows[int(cid)] = (
            g.height,
            m["_row"].to_numpy(),
            m["_store_row"].to_numpy(),
        )
    del idx

    def _load(drug_id: int) -> np.ndarray | None:
        if int(drug_id) not in rows:
            return None
        n, dst, src = rows[int(drug_id)]
        out = np.full((n, Zs.shape[1]), np.nan, dtype=np.float32)
        out[dst] = Zs[src]
        return out

    return _load


def _finite(z: np.ndarray) -> np.ndarray:
    """Écarte les lignes non finies (inclusions sans représentation MOTOR)."""
    z = np.asarray(z)
    ok = np.isfinite(z).all(axis=1)
    return z if ok.all() else z[ok]


# =============================================================================
# 1. Blanchiment de référence
# =============================================================================
class MotorWhitener:
    """Opérateur de blanchiment de l'espace MOTOR (Section 4.2).

    z_tilde = (z - mu) @ W, avec W (D, d) :
      - 'pca'    : W = U_d diag(lambda_d^-1/2), tronqué aux d premières
                   composantes (d < D). Sortie de dimension d.
      - 'zca'    : W = U_d diag(lambda_d^-1/2) U_d^T. Sortie de dimension D.
                   Pour la distance euclidienne et le cosinus, ZCA et PCA de même
                   troncature sont équivalents à une rotation près : seul le
                   coût diffère. 'pca' est donc préférable en pratique.
      - 'center' : W = I (centrage seul, sans blanchiment ; contrôle).
      - 'none'   : W = I, mu = 0 (espace brut ; contrôle).

    Régularisation :
      - `shrinkage` alpha : Sigma <- (1 - alpha) Sigma + alpha (tr(Sigma)/D) I ;
      - `eps_rel` : plancher relatif des valeurs propres (eps_rel * lambda_max),
        et non absolu comme dans la v1 ;
      - `n_components` : int (nombre d'axes), float dans ]0, 1] (fraction de
        variance expliquée) ou None (tous les axes de rang numérique non nul).
    """

    def __init__(
        self,
        mean: np.ndarray,
        whitening_matrix: np.ndarray,
        eigenvalues: np.ndarray | None = None,
        mode: str = "zca",
        meta: dict | None = None,
    ):
        self.mean = np.asarray(mean, dtype=np.float32)  # (D,)
        self.W = np.asarray(whitening_matrix, dtype=np.float32)  # (D, d)
        self.eigenvalues = (
            None if eigenvalues is None else np.asarray(eigenvalues, np.float64)
        )
        self.mode = mode
        self.meta = meta or {}

    @property
    def dim_in(self) -> int:
        return int(self.W.shape[0])

    @property
    def dim_out(self) -> int:
        return int(self.W.shape[1])

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        mode: WhiteningMode = "pca",
        n_components: int | float | None = 128,
        shrinkage: float = 0.0,
        eps_rel: float = 1e-6,
    ) -> MotorWhitener:
        """Calibre le blanchiment sur un échantillon de référence (N, D)."""
        X = np.asarray(embeddings, dtype=np.float64)
        n, D = X.shape
        if mode == "none":
            return cls(np.zeros(D), np.eye(D), mode=mode, meta={"n_fit": n})
        mu = X.mean(axis=0)
        if mode == "center":
            return cls(mu, np.eye(D), mode=mode, meta={"n_fit": n})
        if n < 2:
            raise ValueError("Au moins 2 échantillons requis pour le blanchiment.")

        Xc = X - mu
        cov = (Xc.T @ Xc) / (n - 1)
        if shrinkage > 0:
            cov = (1.0 - shrinkage) * cov + shrinkage * (
                np.trace(cov) / D
            ) * np.eye(D)

        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]

        # Rang numérique : les axes sous le plancher relatif sont du bruit
        # (ou nuls si n <= D) ; les amplifier par lambda^-1/2 serait délétère.
        floor = eps_rel * max(evals[0], 1e-300)
        rank = int(np.sum(evals > floor))
        if n <= D and shrinkage == 0:
            rank = min(rank, n - 1)
            warnings.warn(
                f"Blanchiment : n={n} <= D={D}, covariance de rang <= {n - 1}."
            )

        if n_components is None:
            d = rank
        elif isinstance(n_components, float) and 0 < n_components <= 1:
            cum = np.cumsum(evals[:rank]) / np.sum(evals[:rank])
            d = int(np.searchsorted(cum, n_components) + 1)
        else:
            d = int(n_components)
        d = max(1, min(d, rank))

        U, lam = evecs[:, :d], evals[:d]
        W = U / np.sqrt(lam)[None, :]
        if mode == "zca":
            W = W @ U.T
        elif mode != "pca":
            raise ValueError(f"Mode de blanchiment inconnu : {mode}")

        meta = {
            "n_fit": n,
            "n_components": d,
            "rank": rank,
            "shrinkage": shrinkage,
            "eps_rel": eps_rel,
            "explained_variance": float(np.sum(lam) / np.sum(evals[evals > 0])),
        }
        return cls(mu, W, eigenvalues=evals, mode=mode, meta=meta)

    @classmethod
    def fit_from_embeddings(
        cls, embeddings: np.ndarray, eps: float = 1e-5
    ) -> MotorWhitener:
        """Compatibilité v1 : ZCA complet (plancher désormais relatif)."""
        return cls.fit(embeddings, mode="zca", n_components=None, eps_rel=eps)

    def transform(self, z: np.ndarray, chunk_size: int = 65536) -> np.ndarray:
        """Applique z_tilde = (z - mu) @ W par blocs (float32)."""
        z = np.asarray(z)
        out = np.empty((z.shape[0], self.dim_out), dtype=np.float32)
        for s in range(0, z.shape[0], chunk_size):
            blk = np.asarray(z[s : s + chunk_size], dtype=np.float32)
            out[s : s + chunk_size] = (blk - self.mean) @ self.W
        return out

    def save(self, filepath: str | Path) -> None:
        np.savez_compressed(
            filepath,
            mean=self.mean,
            W=self.W,
            eigenvalues=(
                self.eigenvalues if self.eigenvalues is not None else np.array([])
            ),
            mode=np.array(self.mode),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> MotorWhitener:
        data = np.load(filepath, allow_pickle=False)
        ev = data["eigenvalues"] if "eigenvalues" in data else None
        return cls(
            mean=data["mean"],
            whitening_matrix=data["W"],
            eigenvalues=ev if ev is not None and ev.size else None,
            mode=str(data["mode"]) if "mode" in data else "zca",
            meta=json.loads(str(data["meta"])) if "meta" in data else {},
        )

    def __repr__(self) -> str:
        return (
            f"<MotorWhitener mode={self.mode} {self.dim_in}->{self.dim_out}"
            f" n_fit={self.meta.get('n_fit')}>"
        )


def _cohort_rng(seed: int, drug_id: int) -> np.random.Generator:
    """Générateur déterministe par cohorte, indépendant de l'ordre de parcours."""
    return np.random.default_rng([int(seed), int(drug_id) % (2**63)])


def sample_reference_embeddings(
    drug_ids: Iterable[int],
    loader: EmbeddingLoader,
    sizes: dict[int, int] | None = None,
    n_per_cohort: int = 200,
    weighting: Literal["balanced", "sqrt", "proportional"] = "balanced",
    seed: int = 42,
) -> np.ndarray:
    """Échantillon de référence pour le blanchiment.

    - 'balanced'     : min(N_c, n_per_cohort) patients par cohorte (chaque
                       molécule pèse autant, défaut) ;
    - 'sqrt'         : allocation proportionnelle à sqrt(N_c), même budget
                       total que 'balanced' ;
    - 'proportional' : allocation proportionnelle à N_c (population poolée,
                       dominée par les grandes cohortes), même budget total.
    `sizes` (drug_id -> N) évite de charger les matrices pour calculer
    l'allocation ; sinon la taille est lue sur la matrice (mmap).
    """
    drug_ids = list(drug_ids)
    if sizes is None:
        sizes = {}
        for cid in drug_ids:
            z = loader(cid)
            if z is not None:
                sizes[cid] = int(z.shape[0])
    ids = [c for c in drug_ids if sizes.get(c, 0) > 0]
    n_arr = np.array([sizes[c] for c in ids], dtype=np.float64)
    budget = float(np.minimum(n_arr, n_per_cohort).sum())
    if weighting == "balanced":
        alloc = np.minimum(n_arr, n_per_cohort)
    else:
        w = np.sqrt(n_arr) if weighting == "sqrt" else n_arr
        alloc = np.minimum(n_arr, np.round(budget * w / w.sum()))
    alloc = alloc.astype(int)

    parts = []
    for cid, m in zip(ids, alloc):
        if m <= 0:
            continue
        z = loader(cid)
        if z is None:
            continue
        # Sur-échantillonnage de 25 % pour compenser les lignes NaN éventuelles
        m_draw = min(z.shape[0], int(np.ceil(m * 1.25)))
        rows = np.sort(
            _cohort_rng(seed, cid).choice(z.shape[0], size=m_draw, replace=False)
        )
        parts.append(_finite(np.asarray(z[rows], dtype=np.float32))[:m])
    if not parts:
        raise ValueError("Aucune représentation disponible pour le blanchiment.")
    return np.concatenate(parts, axis=0)


def fit_reference_whitener(
    drug_ids: Iterable[int],
    loader: EmbeddingLoader,
    sizes: dict[int, int] | None = None,
    n_per_cohort: int = 200,
    weighting: Literal["balanced", "sqrt", "proportional"] = "balanced",
    seed: int = 42,
    **whitener_kwargs,
) -> MotorWhitener:
    """Ajuste un `MotorWhitener` global sur un échantillon des cohortes."""
    ref = sample_reference_embeddings(
        drug_ids, loader, sizes, n_per_cohort, weighting, seed
    )
    wh = MotorWhitener.fit(ref, **whitener_kwargs)
    wh.meta.update({"weighting": weighting, "n_per_cohort": n_per_cohort})
    return wh


# =============================================================================
# 2. Prototypes k-means par cohorte
# =============================================================================
@dataclass
class CohortPrototypes:
    """Prototypes d'une cohorte dans l'espace blanchi.

    centroids  : (k, d) moyennes des clusters (non normalisées) ;
    weights    : (k,) fraction de la cohorte dans chaque cluster (somme = 1) ;
    counts     : (k,) effectifs ;
    within_var : (k,) variance intra-cluster moyenne par dimension, tr(Sigma)/d,
                 utilisée pour débiaiser les distances entre petits clusters.
    Les clusters sont ordonnés par poids décroissant (cluster 0 = majoritaire).
    """

    drug_id: int
    centroids: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    within_var: np.ndarray
    n: int
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(len(self.weights))


def choose_k(
    n: int,
    k: int | Literal["auto", "bic"] = "bic",
    k_max: int = 5,
    min_cluster_patients: int = 50,
) -> int:
    """Nombre (maximal) de clusters d'une cohorte de taille n.

    - int    : k fixe, borné par n // min_cluster_patients (au moins 1) ;
    - 'auto' : k = clip(n // (2 * min_cluster_patients), 1, k_max) (règle de
               taille seule : sur-segmente les cohortes homogènes) ;
    - 'bic'  : borne supérieure clip(n // min_cluster_patients, 1, k_max) ; le
               k retenu est ensuite choisi par BIC (`_xmeans_bic`).
    """
    cap = max(1, n // max(1, min_cluster_patients))
    if k == "auto":
        return int(np.clip(n // max(1, 2 * min_cluster_patients), 1, k_max))
    if k == "bic":
        return int(min(cap, k_max))
    return int(max(1, min(int(k), cap)))


def _xmeans_bic(sse: float, counts: np.ndarray, d: int) -> float:
    """BIC d'une partition k-means (Pelleg & Moore 2000, X-means) : mélange
    gaussien sphérique de variance commune sigma^2 = SSE / (d (n - k)).
    Plus grand = meilleur. Sur des données blanchies, une cohorte homogène
    n'est pas scindée (le gain de vraisemblance d'une coupe d'un nuage
    gaussien est inférieur à la pénalité (d + 1)/2 · log n par cluster)."""
    counts = counts[counts > 0].astype(np.float64)
    n, k = counts.sum(), len(counts)
    if n <= k:
        return -np.inf
    s2 = max(sse / (d * (n - k)), 1e-12)
    ll = (
        np.sum(counts * np.log(counts / n))
        - 0.5 * n * d * np.log(2 * np.pi * s2)
        - 0.5 * d * (n - k)
    )
    n_params = (k - 1) + k * d + 1
    return float(ll - 0.5 * n_params * np.log(n))


def _assign(Z: np.ndarray, C: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Affectation au centroïde le plus proche, par blocs."""
    c2 = np.einsum("kd,kd->k", C, C)
    labels = np.empty(Z.shape[0], dtype=np.int64)
    for s in range(0, Z.shape[0], chunk):
        blk = Z[s : s + chunk]
        labels[s : s + chunk] = np.argmin(c2[None, :] - 2.0 * blk @ C.T, axis=1)
    return labels


def _cluster_stats(Z: np.ndarray, labels: np.ndarray, k: int):
    """Effectifs, moyennes et variance intra-cluster par dimension (vectorisé)."""
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    d = Z.shape[1]
    sums = np.zeros((k, d), dtype=np.float64)
    for c in np.flatnonzero(counts):  # boucle sur les k clusters, pas les patients
        sums[c] = Z[labels == c].sum(axis=0, dtype=np.float64)
    sq = np.bincount(
        labels, weights=np.einsum("nd,nd->n", Z, Z).astype(np.float64), minlength=k
    )
    safe = np.maximum(counts, 1)
    means = sums / safe[:, None]
    # E||z - mu||^2 = E||z||^2 - ||mu||^2, ramené par dimension
    wv = (sq / safe - np.einsum("kd,kd->k", means, means)) / d
    wv = np.maximum(wv, 0.0) * safe / np.maximum(safe - 1, 1)
    return counts, means, wv


def fit_cohort_prototypes(
    Z: np.ndarray,
    drug_id: int,
    k: int | Literal["auto", "bic"] = "bic",
    k_max: int = 5,
    min_cluster_patients: int = 50,
    max_fit_samples: int = 20000,
    n_init: int = 4,
    seed: int = 42,
) -> CohortPrototypes:
    """k-means d'une cohorte dans l'espace blanchi (Section 4.2).

    - ajustement sur au plus `max_fit_samples` patients tirés au hasard (graine
      dérivée de (seed, drug_id)) ; affectation ensuite de tous les patients ;
    - les clusters de moins de `min_cluster_patients` patients sont fusionnés
      itérativement dans le plus proche (un cluster de quelques patients
      atypiques ne doit pas pouvoir porter une paire par min-linkage) ;
    - centroïdes, poids et variances recalculés sur la cohorte entière.
    """
    Z = np.asarray(Z, dtype=np.float32)
    n = int(Z.shape[0])
    if n == 0:
        raise ValueError(f"Cohorte {drug_id} vide : aucun prototype calculable.")
    k_eff = choose_k(n, k, k_max, min_cluster_patients)
    bic_scores: dict[int, float] = {}

    if k_eff == 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        rng = _cohort_rng(seed, drug_id)
        fit_rows = (
            np.sort(rng.choice(n, size=max_fit_samples, replace=False))
            if n > max_fit_samples
            else slice(None)
        )
        Zf = Z[fit_rows]
        km_seed = int(rng.integers(2**31 - 1))
        candidates = range(1, k_eff + 1) if k == "bic" else [k_eff]
        best, best_bic = None, -np.inf
        for kk in candidates:
            if kk == 1:
                centers = Zf.mean(axis=0, keepdims=True)
                sse = float(((Zf - centers) ** 2).sum())
                cnt = np.array([Zf.shape[0]])
            else:
                km = KMeans(n_clusters=kk, n_init=n_init, random_state=km_seed)
                km.fit(Zf)
                centers, sse = km.cluster_centers_, float(km.inertia_)
                cnt = np.bincount(km.labels_, minlength=kk)
            bic = _xmeans_bic(sse, cnt, Z.shape[1]) if k == "bic" else 0.0
            bic_scores[kk] = bic
            if best is None or bic > best_bic:
                best, best_bic = centers.astype(np.float32), bic
        k_eff = len(best)
        labels = (
            np.zeros(n, dtype=np.int64) if k_eff == 1 else _assign(Z, best)
        )

    # Fusion des clusters trop petits dans le plus proche
    counts, means, wv = _cluster_stats(Z, labels, k_eff)
    keep = np.flatnonzero(counts > 0)
    while len(keep) > 1 and counts[keep].min() < min_cluster_patients:
        worst = keep[np.argmin(counts[keep])]
        keep = keep[keep != worst]
        labels = keep[_assign(Z, means[keep].astype(np.float32))]
        counts, means, wv = _cluster_stats(Z, labels, k_eff)
        keep = np.flatnonzero(counts > 0)

    order = keep[np.argsort(-counts[keep], kind="stable")]
    return CohortPrototypes(
        drug_id=int(drug_id),
        centroids=means[order].astype(np.float32),
        weights=(counts[order] / n).astype(np.float64),
        counts=counts[order],
        within_var=wv[order],
        n=n,
        meta={"k_requested": k, "k_fit": k_eff, "bic": bic_scores},
    )


# =============================================================================
# 3. Banque de prototypes et distances inter-cohortes
# =============================================================================
@dataclass
class PrototypeBank:
    """Prototypes de toutes les cohortes, complétés à K = max k (padding)."""

    drug_ids: np.ndarray  # (C,)
    centroids: np.ndarray  # (C, K, d)
    weights: np.ndarray  # (C, K), 0 sur le padding
    counts: np.ndarray  # (C, K)
    within_var: np.ndarray  # (C, K)
    n: np.ndarray  # (C,)
    k: np.ndarray  # (C,)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_prototypes(
        cls, protos: list[CohortPrototypes], meta: dict | None = None
    ) -> PrototypeBank:
        C = len(protos)
        K = max(p.k for p in protos)
        d = protos[0].centroids.shape[1]
        cent = np.zeros((C, K, d), dtype=np.float32)
        w = np.zeros((C, K))
        cnt = np.zeros((C, K), dtype=np.int64)
        wv = np.zeros((C, K))
        for i, p in enumerate(protos):
            cent[i, : p.k] = p.centroids
            w[i, : p.k] = p.weights
            cnt[i, : p.k] = p.counts
            wv[i, : p.k] = p.within_var
        return cls(
            drug_ids=np.array([p.drug_id for p in protos], dtype=np.int64),
            centroids=cent,
            weights=w,
            counts=cnt,
            within_var=wv,
            n=np.array([p.n for p in protos], dtype=np.int64),
            k=np.array([p.k for p in protos], dtype=np.int64),
            meta=meta or {},
        )

    def save(self, filepath: str | Path) -> None:
        np.savez_compressed(
            filepath,
            drug_ids=self.drug_ids,
            centroids=self.centroids,
            weights=self.weights,
            counts=self.counts,
            within_var=self.within_var,
            n=self.n,
            k=self.k,
            meta=np.array(json.dumps(self.meta, default=str)),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> PrototypeBank:
        f = np.load(filepath, allow_pickle=False)
        return cls(
            **{
                key: f[key]
                for key in (
                    "drug_ids", "centroids", "weights", "counts",
                    "within_var", "n", "k",
                )
            },
            meta=json.loads(str(f["meta"])),
        )

    def __len__(self) -> int:
        return len(self.drug_ids)


def _block_distances(
    bank: PrototypeBank, rows: slice, metric: DistanceMetric
) -> np.ndarray:
    """Distances (b, K, C, K) entre les prototypes d'un bloc et tous les autres."""
    A = bank.centroids[rows].astype(np.float64)  # (b, K, d)
    B = bank.centroids.astype(np.float64)  # (C, K, d)
    d = A.shape[-1]
    if metric == "cosine":
        An = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
        Bn = B / np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
        return 1.0 - np.einsum("ikd,jld->ikjl", An, Bn)
    a2 = np.einsum("ikd,ikd->ik", A, A)
    b2 = np.einsum("jld,jld->jl", B, B)
    sq = (
        a2[:, :, None, None]
        + b2[None, None, :, :]
        - 2.0 * np.einsum("ikd,jld->ikjl", A, B)
    ) / d
    if metric == "mahalanobis_debiased":
        # E||mu_hat_A - mu_hat_B||^2 = ||mu_A - mu_B||^2 + tr(S_A)/n_A + tr(S_B)/n_B
        # -> estimateur sans biais de la distance entre vraies moyennes : les
        # distances deviennent comparables entre petits et grands clusters.
        va = bank.within_var[rows] / np.maximum(bank.counts[rows], 1)
        vb = bank.within_var / np.maximum(bank.counts, 1)
        sq = sq - va[:, :, None, None] - vb[None, None, :, :]
    elif metric != "mahalanobis":
        raise ValueError(f"Métrique inconnue : {metric}")
    return np.sqrt(np.maximum(sq, 0.0))


def _paired_cluster_distances(
    bank: PrototypeBank, ia: np.ndarray, ib: np.ndarray, metric: DistanceMetric
) -> np.ndarray:
    """Distances (P, K, K) entre clusters pour une liste de couples (ia, ib)."""
    A = bank.centroids[ia].astype(np.float64)  # (P, K, d)
    B = bank.centroids[ib].astype(np.float64)
    d = A.shape[-1]
    if metric == "cosine":
        A = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
        B = B / np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
        return 1.0 - np.einsum("pkd,pld->pkl", A, B)
    sq = (
        np.einsum("pkd,pkd->pk", A, A)[:, :, None]
        + np.einsum("pld,pld->pl", B, B)[:, None, :]
        - 2.0 * np.einsum("pkd,pld->pkl", A, B)
    ) / d
    if metric == "mahalanobis_debiased":
        va = bank.within_var[ia] / np.maximum(bank.counts[ia], 1)
        vb = bank.within_var[ib] / np.maximum(bank.counts[ib], 1)
        sq = sq - va[:, :, None] - vb[:, None, :]
    return np.sqrt(np.maximum(sq, 0.0))


def _eligible(bank: PrototypeBank, min_cluster_weight: float) -> np.ndarray:
    """Clusters éligibles au min-linkage (le majoritaire l'est toujours)."""
    # weights == 0 : clusters de padding (cohortes à k < K), jamais éligibles
    elig = (bank.weights >= min_cluster_weight) & (bank.weights > 0)
    elig[:, 0] |= bank.weights[:, 0] > 0
    return elig


def cluster_distances(
    bank: PrototypeBank,
    metric: DistanceMetric = "mahalanobis_debiased",
    min_cluster_weight: float = 0.05,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance min-linkage entre toutes les cohortes (Section 4.2).

        d(A, B) = min_{i, j éligibles} dist(mu_{A,i}, mu_{B,j})

    Un cluster est éligible si son poids >= `min_cluster_weight` (le cluster
    majoritaire l'est toujours). Retourne (dist (C, C), cl_a (C, C), cl_b (C, C))
    où cl_a[i, j] est l'indice du cluster de la cohorte i apparié à la cohorte j.
    Diagonale = +inf. Mémoire par bloc : block_size · K² · C flottants.
    """
    C, K = bank.weights.shape
    elig = _eligible(bank, min_cluster_weight)
    dist = np.full((C, C), np.inf)
    cl_a = np.full((C, C), -1, dtype=np.int16)
    cl_b = np.full((C, C), -1, dtype=np.int16)
    for s in range(0, C, block_size):
        rows = slice(s, min(s + block_size, C))
        D = _block_distances(bank, rows, metric)  # (b, K, C, K)
        mask = elig[rows][:, :, None, None] & elig[None, None, :, :]
        D = np.where(mask, D, np.inf)
        b = D.shape[0]
        flat = D.transpose(0, 2, 1, 3).reshape(b, C, K * K)
        arg = np.argmin(flat, axis=2)
        dist[rows] = np.take_along_axis(flat, arg[..., None], axis=2)[..., 0]
        cl_a[rows] = arg // K
        cl_b[rows] = arg % K
    np.fill_diagonal(dist, np.inf)
    return dist, cl_a, cl_b


# =============================================================================
# 4. Règle de décision
# =============================================================================
def select_pairs(
    bank: PrototypeBank,
    dist: np.ndarray,
    cl_a: np.ndarray,
    cl_b: np.ndarray,
    rule: PairRule = "threshold_topn",
    max_dist: float | None = None,
    top_n: int = 10,
    metric: DistanceMetric = "mahalanobis_debiased",
    min_cluster_weight: float = 0.05,
) -> pl.DataFrame:
    """Sélectionne les paires (a, b), a < b, selon la règle choisie.

    - 'threshold'      : d(A, B) <= max_dist ;
    - 'topn'           : B parmi les top_n plus proches de A OU l'inverse ;
    - 'mutual_topn'    : B parmi les top_n de A ET A parmi les top_n de B ;
    - 'threshold_topn' : 'topn' ET d <= max_dist (défaut ; sans max_dist,
                         équivaut à 'topn').
    Colonnes : drug_id_a, drug_id_b, kmeans_min_dist, kmeans_cluster_a/b,
    kmeans_weight_a/b, kmeans_k_a/b, kmeans_rank_a (rang de b dans les voisins
    de a, 1 = plus proche) et kmeans_rank_b.

    Si max_dist est fourni, ajoute kmeans_shared_weight_a (resp. _b) : fraction
    de la cohorte A dans des clusters éligibles situés à <= max_dist d'au moins
    un cluster éligible de B. Contrairement au poids du seul cluster apparié,
    cette masse reste interprétable quand k-means sur-segmente une indication
    en plusieurs clusters. `metric` et `min_cluster_weight` doivent être ceux
    passés à `cluster_distances`.
    """
    C = len(bank)
    finite = np.isfinite(dist)
    # Rang de chaque colonne dans chaque ligne (1 = plus proche)
    order = np.argsort(dist, axis=1, kind="stable")
    rank = np.empty((C, C), dtype=np.int32)
    rank[np.arange(C)[:, None], order] = np.arange(1, C + 1)[None, :]
    in_top = (rank <= top_n) & finite

    if rule == "threshold":
        if max_dist is None:
            raise ValueError("La règle 'threshold' requiert max_dist.")
        sel = dist <= max_dist
    elif rule in ("topn", "threshold_topn"):
        sel = in_top | in_top.T
        if rule == "threshold_topn" and max_dist is not None:
            sel &= dist <= max_dist
    elif rule == "mutual_topn":
        sel = in_top & in_top.T
        if max_dist is not None:
            sel &= dist <= max_dist
    else:
        raise ValueError(f"Règle inconnue : {rule}")

    ii, jj = np.nonzero(np.triu(sel & finite, k=1))
    # Orientation canonique : drug_id_a < drug_id_b
    ids = bank.drug_ids
    swap = ids[ii] > ids[jj]
    ia, ib = np.where(swap, jj, ii), np.where(swap, ii, jj)
    ca, cb = cl_a[ia, ib].astype(np.int64), cl_b[ia, ib].astype(np.int64)
    cols = {
        "drug_id_a": ids[ia],
        "drug_id_b": ids[ib],
        "kmeans_min_dist": dist[ia, ib],
        "kmeans_cluster_a": ca,
        "kmeans_cluster_b": cb,
        "kmeans_weight_a": bank.weights[ia, ca],
        "kmeans_weight_b": bank.weights[ib, cb],
        "kmeans_k_a": bank.k[ia],
        "kmeans_k_b": bank.k[ib],
        "kmeans_rank_a": rank[ia, ib],
        "kmeans_rank_b": rank[ib, ia],
    }
    if max_dist is not None and len(ia):
        elig = _eligible(bank, min_cluster_weight)
        D = _paired_cluster_distances(bank, ia, ib, metric)  # (P, K, K)
        close = (D <= max_dist) & elig[ia][:, :, None] & elig[ib][:, None, :]
        cols["kmeans_shared_weight_a"] = (bank.weights[ia] * close.any(2)).sum(1)
        cols["kmeans_shared_weight_b"] = (bank.weights[ib] * close.any(1)).sum(1)
    return pl.DataFrame(
        cols
    ).sort("kmeans_min_dist")


def threshold_from_reference(
    dist: np.ndarray,
    drug_ids: np.ndarray,
    reference_pairs: Iterable[tuple[int, int]],
    quantile: float = 0.5,
) -> float:
    """Seuil calibré sur des paires de référence (ex. intra-ATC4, Méthode 1) :

    quantile de d(A, B) sur ces paires. Aide au choix de max_dist, sans vérité
    terrain (question ouverte, voir le rapport de la Méthode 3).
    """
    pos = {int(c): i for i, c in enumerate(drug_ids)}
    vals = [
        dist[pos[a], pos[b]]
        for a, b in reference_pairs
        if a in pos and b in pos and np.isfinite(dist[pos[a], pos[b]])
    ]
    if not vals:
        raise ValueError("Aucune paire de référence présente dans la banque.")
    return float(np.quantile(vals, quantile))


# =============================================================================
# 5. Orchestration
# =============================================================================
def build_prototype_bank(
    drug_ids: Iterable[int],
    loader: EmbeddingLoader,
    whitener: MotorWhitener | None,
    k: int | Literal["auto", "bic"] = "bic",
    k_max: int = 5,
    min_cluster_patients: int = 50,
    max_fit_samples: int = 20000,
    n_init: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> PrototypeBank:
    """Blanchit chaque cohorte puis calcule ses prototypes (une cohorte en
    mémoire à la fois)."""
    protos, missing = [], 0
    drug_ids = list(drug_ids)
    for i, cid in enumerate(drug_ids):
        z = loader(cid)
        z = None if z is None else _finite(z)
        if z is None or z.shape[0] == 0:
            missing += 1
            continue
        Z = whitener.transform(z) if whitener is not None else np.asarray(
            z, dtype=np.float32
        )
        protos.append(
            fit_cohort_prototypes(
                Z, cid, k, k_max, min_cluster_patients, max_fit_samples,
                n_init, seed,
            )
        )
        if verbose and (i + 1) % 100 == 0:
            print(f"  prototypes : {i + 1:,}/{len(drug_ids):,} cohortes")
    if not protos:
        raise ValueError("Aucune cohorte avec représentations MOTOR.")
    if verbose and missing:
        print(f"  [Avertissement] {missing:,} cohortes sans représentation MOTOR.")
    meta = {
        "k": k, "k_max": k_max, "min_cluster_patients": min_cluster_patients,
        "max_fit_samples": max_fit_samples, "n_init": n_init, "seed": seed,
        "whitener": None if whitener is None else {
            "mode": whitener.mode, **whitener.meta
        },
    }
    return PrototypeBank.from_prototypes(protos, meta)
