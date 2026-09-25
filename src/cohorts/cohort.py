# src/cohorts/cohort.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from src.catalog.catalog import DrugItem

# Les 6 biomarqueurs cliniques cibles du protocole (Section 4.4)
CLINICAL_LAB_BIOMARKERS = ["ldl_c", "hba1c", "egfr", "alt", "crp", "sbp"]


class DrugCohort:
    """Gestionnaire clinique et phénotypique multi-sources d'une cohorte de patients

    pour une molécule ou une association médicamenteuse.

    Gère deux sous-cohortes complémentaires :
      1. Stanford EHR (Longitudinal) :
         - Suivi temporel à t0, t0 + 6 mois et t0 + 12 mois.
         - MOTOR baseline z0 à t0 (sans les prescriptions de t0).
         - Mesures réelles des 6 biomarqueurs de laboratoire (baseline, 6m, 12m).
         - Changements du profil RABIT (Delta RABIT 6m et 12m) après ablation
           du médicament d'intérêt dans l'historique EHR.
      2. UK Biobank (Protéomique Olink mesurée) :
         - Participants ayant débuté le médicament entre 3 et 24 mois avant la prise de sang.
         - MOTOR baseline z0 à t0.
         - Niveaux réels mesurés sur le panel Olink Explore (~3 000 protéines).

    Permet le calcul de prototypes cliniques k-Means et de la distance inter-cohortes
    min-linkage cosinus.
    """

    def __init__(
        self,
        drug_id: int,
        name: str | None = None,
        kind: Literal[
            "monotherapy", "fixed_combination", "combination"
        ] = "monotherapy",
        ingredient_concept_ids: list[int] | None = None,
        stanford_index: pl.DataFrame | None = None,
        ukb_index: pl.DataFrame | None = None,
    ):
        self.drug_id = int(drug_id)
        self.name = name or f"Drug_{self.drug_id}"
        self.kind = kind
        self.ingredient_concept_ids = ingredient_concept_ids or (
            [self.drug_id] if kind == "monotherapy" else []
        )

        # ---------------------------------------------------------------------
        # 1. Sous-cohorte Stanford EHR (Longitudinale)
        # DataFrame requis : [person_id, t0, t_6m, t_12m, ...]
        # ---------------------------------------------------------------------
        self.stanford_index = stanford_index
        self._stanford_motor_z0: np.ndarray | None = None  # (N_stanford, 768)
        self.stanford_labs: pl.DataFrame | None = None  # Labs réels à t0, 6m, 12m
        self._stanford_delta_rabit_6m: np.ndarray | None = (
            None  # (N_stanford, P)
        )
        self._stanford_delta_rabit_12m: np.ndarray | None = (
            None  # (N_stanford, P)
        )
        self._rabit_protein_names: list[str] | None = None

        # ---------------------------------------------------------------------
        # 2. Sous-cohorte UK Biobank (Olink mesuré)
        # DataFrame requis : [eid, t0, blood_draw_date, months_to_draw] (3 à 24m)
        # ---------------------------------------------------------------------
        self.ukb_index = ukb_index
        self._ukb_motor_z0: np.ndarray | None = None  # (N_ukb, 768)
        self._ukb_olink_measured: np.ndarray | None = None  # (N_ukb, P_olink)
        self._olink_protein_names: list[str] | None = None

        # ---------------------------------------------------------------------
        # 3. Prototypes cliniques
        # ---------------------------------------------------------------------
        # Centroïdes k-means dans l'espace blanchi (k, d) et poids associés
        # (fraction de la cohorte par cluster), cf. src/pairs/motor_pairs.py
        self._prototypes: np.ndarray | None = None  # (k, d)
        self._prototype_weights: np.ndarray | None = None  # (k,)

    # -------------------------------------------------------------------------
    # Constructeur d'usine
    # -------------------------------------------------------------------------
    @classmethod
    def from_drug_item(
        cls,
        drug_item: DrugItem,
        stanford_index: pl.DataFrame | None = None,
        ukb_index: pl.DataFrame | None = None,
    ) -> DrugCohort:
        """Instancie la cohorte directement depuis un DrugItem du DrugCatalog."""
        return cls(
            drug_id=drug_item.drug_id,
            name=drug_item.name,
            kind=drug_item.kind,
            ingredient_concept_ids=list(drug_item.ingredient_concept_ids),
            stanford_index=stanford_index,
            ukb_index=ukb_index,
        )

    # -------------------------------------------------------------------------
    # Propriétés descriptives
    # -------------------------------------------------------------------------
    @property
    def n_stanford(self) -> int:
        return len(self.stanford_index) if self.stanford_index is not None else 0

    @property
    def n_ukb(self) -> int:
        return len(self.ukb_index) if self.ukb_index is not None else 0

    @property
    def n_total(self) -> int:
        return self.n_stanford + self.n_ukb

    @property
    def is_combination(self) -> bool:
        return len(self.ingredient_concept_ids) > 1

    @property
    def stanford_motor_z0(self) -> np.ndarray | None:
        return self._stanford_motor_z0

    @property
    def stanford_delta_rabit_6m(self) -> np.ndarray | None:
        return self._stanford_delta_rabit_6m

    @property
    def stanford_delta_rabit_12m(self) -> np.ndarray | None:
        return self._stanford_delta_rabit_12m

    @property
    def ukb_motor_z0(self) -> np.ndarray | None:
        return self._ukb_motor_z0

    @property
    def ukb_olink_measured(self) -> np.ndarray | None:
        return self._ukb_olink_measured

    @property
    def prototypes(self) -> np.ndarray | None:
        return self._prototypes

    # -------------------------------------------------------------------------
    # Sauvegarde et Chargement Disque
    # -------------------------------------------------------------------------
    def save(self, output_base_dir: str | Path) -> Path:
        """Sauvegarde l'ensemble des métadonnées, index et tenseurs alignés de la cohorte."""
        base_dir = Path(output_base_dir)
        cohort_dir = base_dir / f"cohort_{self.drug_id}"
        cohort_dir.mkdir(parents=True, exist_ok=True)

        # 1. Métadonnées JSON
        metadata = {
            "drug_id": self.drug_id,
            "name": self.name,
            "kind": self.kind,
            "ingredient_concept_ids": self.ingredient_concept_ids,
            "n_stanford": self.n_stanford,
            "n_ukb": self.n_ukb,
            "has_stanford_z0": self._stanford_motor_z0 is not None,
            "has_stanford_labs": self.stanford_labs is not None,
            "has_stanford_rabit": self._stanford_delta_rabit_6m is not None,
            "has_ukb_z0": self._ukb_motor_z0 is not None,
            "has_ukb_olink": self._ukb_olink_measured is not None,
            "k_prototypes": (
                len(self._prototypes) if self._prototypes is not None else None
            ),
        }
        with open(cohort_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. Index tables
        if self.stanford_index is not None:
            self.stanford_index.write_parquet(
                cohort_dir / "stanford_index.parquet"
            )
        if self.ukb_index is not None:
            self.ukb_index.write_parquet(cohort_dir / "ukb_index.parquet")

        # 3. Données cliniques de laboratoire (Stanford)
        if self.stanford_labs is not None:
            self.stanford_labs.write_parquet(
                cohort_dir / "stanford_labs.parquet"
            )

        # 4. Tenseurs MOTOR et outcomes (Parquet / NPY)
        if self._stanford_motor_z0 is not None:
            np.save(
                cohort_dir / "stanford_motor_z0.npy", self._stanford_motor_z0
            )

        if self._stanford_delta_rabit_6m is not None:
            np.save(
                cohort_dir / "stanford_delta_rabit_6m.npy",
                self._stanford_delta_rabit_6m,
            )
        if self._stanford_delta_rabit_12m is not None:
            np.save(
                cohort_dir / "stanford_delta_rabit_12m.npy",
                self._stanford_delta_rabit_12m,
            )
        if self._rabit_protein_names:
            with open(cohort_dir / "rabit_proteins.json", "w") as f:
                json.dump(self._rabit_protein_names, f)

        if self._ukb_motor_z0 is not None:
            np.save(cohort_dir / "ukb_motor_z0.npy", self._ukb_motor_z0)

        if self._ukb_olink_measured is not None:
            np.save(
                cohort_dir / "ukb_olink_measured.npy", self._ukb_olink_measured
            )
        if self._olink_protein_names:
            with open(cohort_dir / "olink_proteins.json", "w") as f:
                json.dump(self._olink_protein_names, f)

        # 5. Prototypes
        if self._prototypes is not None:
            k = len(self._prototypes)
            np.save(cohort_dir / f"prototypes_k{k}.npy", self._prototypes)

        return cohort_dir

    @classmethod
    def from_disk(cls, cohort_dir: str | Path) -> DrugCohort:
        """Reconstitue intégralement une DrugCohort depuis son dossier persistant."""
        path = Path(cohort_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {path}")

        meta_file = path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json introuvable dans {path}")

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        # Instanciation de base
        cohort = cls(
            drug_id=meta["drug_id"],
            name=meta.get("name"),
            kind=meta.get("kind", "monotherapy"),
            ingredient_concept_ids=meta.get("ingredient_concept_ids", []),
        )

        # Rechargement des index
        st_index_file = path / "stanford_index.parquet"
        if st_index_file.exists():
            cohort.stanford_index = pl.read_parquet(st_index_file)

        ukb_index_file = path / "ukb_index.parquet"
        if ukb_index_file.exists():
            cohort.ukb_index = pl.read_parquet(ukb_index_file)

        # Rechargement des tables d'outcomes cliniques
        labs_file = path / "stanford_labs.parquet"
        alt_labs_file = path / "biomarkers.parquet"
        if labs_file.exists():
            cohort.stanford_labs = pl.read_parquet(labs_file)
        elif alt_labs_file.exists():
            cohort.stanford_labs = pl.read_parquet(alt_labs_file)

        # Rechargement des tenseurs
        if (path / "stanford_motor_z0.npy").exists():
            cohort._stanford_motor_z0 = np.load(
                path / "stanford_motor_z0.npy"
            ).astype(np.float32)

        if (path / "stanford_delta_rabit_6m.npy").exists():
            cohort._stanford_delta_rabit_6m = np.load(
                path / "stanford_delta_rabit_6m.npy"
            ).astype(np.float32)

        if (path / "stanford_delta_rabit_12m.npy").exists():
            cohort._stanford_delta_rabit_12m = np.load(
                path / "stanford_delta_rabit_12m.npy"
            ).astype(np.float32)

        if (path / "rabit_proteins.json").exists():
            with open(path / "rabit_proteins.json") as f:
                cohort._rabit_protein_names = json.load(f)

        if (path / "ukb_motor_z0.npy").exists():
            cohort._ukb_motor_z0 = np.load(path / "ukb_motor_z0.npy").astype(
                np.float32
            )

        if (path / "ukb_olink_measured.npy").exists():
            cohort._ukb_olink_measured = np.load(
                path / "ukb_olink_measured.npy"
            ).astype(np.float32)

        if (path / "olink_proteins.json").exists():
            with open(path / "olink_proteins.json") as f:
                cohort._olink_protein_names = json.load(f)

        # Prototypes
        proto_files = list(path.glob("prototypes_k*.npy"))
        if proto_files:
            cohort._prototypes = np.load(proto_files[0]).astype(np.float32)

        return cohort

    # -------------------------------------------------------------------------
    # Alignement des données de la sous-cohorte Stanford
    # -------------------------------------------------------------------------
    def load_stanford_motor_z0(
        self,
        source: str | Path | pl.DataFrame,
        id_cols: list[str] | None = None,
    ) -> np.ndarray:
        """Charge et aligne strictement la matrice MOTOR z0 (N, 768) sur stanford_index

        en utilisant la clé composite (person_id, t0) (Section 4.1).

        `source` : DataFrame, fichier parquet, ou dossier du magasin central
        produit par scripts/extract_motor_representations.py (fragments
        chunk_*.parquet). Deux formats de colonnes sont acceptés : une colonne
        `z` de type Array(Float32, 768) (magasin) ou des colonnes data_0..data_767
        (ancien format CSV/parquet RABIT).

        L'ordre et le nombre de lignes de stanford_index sont conservés : une
        inclusion sans représentation reçoit une ligne NaN (jamais de décalage
        silencieux entre z0 et l'index).
        """
        if self.stanford_index is None:
            raise ValueError(
                "stanford_index doit être initialisé avant d'aligner MOTOR z0."
            )

        keys = id_cols or ["person_id", "t0"]
        if isinstance(source, pl.DataFrame):
            df_reps = source
        else:
            path = Path(source)
            if path.is_dir():
                shards = sorted(path.glob("chunk_*.parquet"))
                if not shards:
                    raise FileNotFoundError(f"Aucun fragment chunk_*.parquet dans {path}")
                df_reps = pl.concat([pl.read_parquet(f) for f in shards])
            else:
                df_reps = pl.read_parquet(path)

        index_keys = self.stanford_index.select(
            pl.col(keys[0]).cast(pl.Int64), *[pl.col(k) for k in keys[1:]]
        ).with_row_index("_row")
        df_reps = df_reps.with_columns(pl.col(keys[0]).cast(pl.Int64))

        if "z" in df_reps.columns:
            matched = index_keys.join(
                df_reps.select(keys + ["z"]).unique(subset=keys, keep="first"),
                on=keys,
                how="inner",
            )
            values = matched["z"].to_numpy().astype(np.float32)
        else:
            feature_cols = [
                c
                for c in df_reps.columns
                if c.startswith("data_") or c.startswith("emb_")
            ]
            if not feature_cols:
                raise ValueError("Aucune colonne de représentation (z, data_*, emb_*).")
            matched = index_keys.join(
                df_reps.select(keys + feature_cols).unique(subset=keys, keep="first"),
                on=keys,
                how="inner",
            )
            values = matched.select(feature_cols).to_numpy().astype(np.float32)

        n = len(self.stanford_index)
        z0 = np.full((n, values.shape[1] if values.ndim == 2 and len(values) else 768),
                     np.nan, dtype=np.float32)
        if len(matched):
            z0[matched["_row"].to_numpy()] = values

        if len(matched) != n:
            print(
                f"[Avertissement] Cohorte {self.drug_id} (Stanford) : "
                f"{len(matched)}/{n} inclusions disposent d'un embedding MOTOR z0 "
                f"(lignes manquantes = NaN)."
            )

        self._stanford_motor_z0 = z0
        return self._stanford_motor_z0

    def load_stanford_rabit_deltas(
        self,
        parquet_6m: str | Path,
        parquet_12m: str | Path | None = None,
        id_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Aligne les deltas de profils synthétiques Delta RABIT à 6 mois et 12 mois

        calculés après ablation de la molécule de l'historique EHR.
        """
        if self.stanford_index is None:
            raise ValueError(
                "stanford_index doit être initialisé avant d'aligner les deltas RABIT."
            )

        keys = id_cols or ["person_id", "t0"]

        # 1. Delta 6 mois
        df_6m = pl.read_parquet(parquet_6m)
        protein_cols = [c for c in df_6m.columns if c not in keys]
        self._rabit_protein_names = protein_cols

        aligned_6m = self.stanford_index.select(keys).join(
            df_6m.select(keys + protein_cols), on=keys, how="inner"
        )
        self._stanford_delta_rabit_6m = (
            aligned_6m.select(protein_cols).to_numpy().astype(np.float32)
        )

        # 2. Delta 12 mois (si disponible)
        if parquet_12m is not None and Path(parquet_12m).exists():
            df_12m = pl.read_parquet(parquet_12m)
            aligned_12m = self.stanford_index.select(keys).join(
                df_12m.select(keys + protein_cols), on=keys, how="inner"
            )
            self._stanford_delta_rabit_12m = (
                aligned_12m.select(protein_cols).to_numpy().astype(np.float32)
            )

        return self._stanford_delta_rabit_6m, self._stanford_delta_rabit_12m

    # -------------------------------------------------------------------------
    # Alignement des données de la sous-cohorte UK Biobank
    # -------------------------------------------------------------------------
    def load_ukb_olink_data(
        self,
        parquet_olink: str | Path,
        id_col: str = "eid",
    ) -> np.ndarray:
        """Charge et aligne les taux réels de protéines mesurées au prélèvement Olink

        pour les participants ayant débuté la molécule 3 à 24 mois plus tôt.
        """
        if self.ukb_index is None:
            raise ValueError(
                "ukb_index doit être initialisé avant d'aligner les mesures Olink."
            )

        df_olink = pl.read_parquet(parquet_olink)
        protein_cols = [c for c in df_olink.columns if c != id_col]
        self._olink_protein_names = protein_cols

        aligned = self.ukb_index.select([id_col]).join(
            df_olink.select([id_col] + protein_cols), on=id_col, how="inner"
        )

        self._ukb_olink_measured = (
            aligned.select(protein_cols).to_numpy().astype(np.float32)
        )
        return self._ukb_olink_measured

    # -------------------------------------------------------------------------
    # Prototypes Cliniques Multi-Centroïdes (k-Means adaptatif sur z0)
    # -------------------------------------------------------------------------
    def compute_prototypes(
        self,
        k: int | Literal["auto", "bic"] = 3,
        source: Literal["stanford", "ukb"] = "stanford",
        whitener: MotorWhitener | None = None,
        force_recompute: bool = False,
        random_state: int = 42,
        k_max: int = 5,
        min_cluster_patients: int = 50,
        max_fit_samples: int = 20000,
    ) -> np.ndarray:
        """Calcule (ou renvoie, si déjà calculés) les prototypes k-means de la
        cohorte dans l'espace MOTOR blanchi (Section 4.2, Méthode 3).

        Enveloppe de `src.pairs.motor_pairs.fit_cohort_prototypes` pour une
        cohorte isolée ; le traitement de toutes les cohortes passe par
        `scripts/build_motor_pairs.py`. Les lignes NaN (inclusions sans
        représentation) sont écartées. Les centroïdes ne sont plus normalisés
        L2 : la métrique est choisie au moment du calcul des distances.
        Le `whitener` doit être le même pour toutes les cohortes comparées.
        """
        from src.pairs.motor_pairs import fit_cohort_prototypes

        if self._prototypes is not None and not force_recompute:
            return self._prototypes
        reps = (
            self._stanford_motor_z0
            if source == "stanford"
            else self._ukb_motor_z0
        )
        if reps is None:
            raise ValueError(
                f"Représentations MOTOR z0 non chargées pour la source '{source}'."
            )
        reps = reps[np.isfinite(reps).all(axis=1)]
        Z = whitener.transform(reps) if whitener is not None else reps
        protos = fit_cohort_prototypes(
            Z,
            self.drug_id,
            k=k,
            k_max=k_max,
            min_cluster_patients=min_cluster_patients,
            max_fit_samples=max_fit_samples,
            seed=random_state,
        )
        self._prototypes = protos.centroids
        self._prototype_weights = protos.weights
        return self._prototypes

    # -------------------------------------------------------------------------
    # Distance Inter-Cohortes (Min-Linkage)
    # -------------------------------------------------------------------------
    def min_cosine_distance(
        self, other: DrugCohort, min_cluster_weight: float = 0.0
    ) -> float:
        """Distance min-linkage cosinus entre prototypes (Section 4.2) :

            d_min(A, B) = min_{i,j éligibles} (1 - cos(mu_{A,i}, mu_{B,j})),

        dans [0, 2]. Un cluster est éligible si son poids >= min_cluster_weight
        (le cluster majoritaire l'est toujours). Pour le criblage de toutes les
        paires, préférer `src.pairs.motor_pairs.cluster_distances` (vectorisé,
        métrique de Mahalanobis débiaisée par défaut).
        """
        for c in (self, other):
            if c._prototypes is None:
                raise ValueError(
                    f"Prototypes non calculés pour {c.drug_id} ({c.name})."
                )

        def _eligible(c: DrugCohort) -> np.ndarray:
            P = c._prototypes / np.maximum(
                np.linalg.norm(c._prototypes, axis=1, keepdims=True), 1e-12
            )
            w = c._prototype_weights
            if w is None or min_cluster_weight <= 0:
                return P
            keep = w >= min_cluster_weight
            keep[np.argmax(w)] = True
            return P[keep]

        max_sim = float(np.max(_eligible(self) @ _eligible(other).T))
        return float(np.clip(1.0 - max_sim, 0.0, 2.0))

    # -------------------------------------------------------------------------
    # Export FEMR Prediction Times (Inférence MOTOR)
    # -------------------------------------------------------------------------
    def to_femr_prediction_times(
        self,
        source: Literal["stanford", "ukb"] = "stanford",
        timepoint: Literal["baseline_t0", "followup_6m", "followup_12m"] = "baseline_t0",
        anchor: str = "day_start",
    ) -> pl.DataFrame:
        """Génère les points d'inférence temporels requis par le moteur FEMR / MOTOR :

        (patient_id, prediction_time) avec granularité à la minute. Plusieurs
        instants par patient sont possibles (femr_compute_representations les
        accepte ; seul rabit_pipeline.py impose un patient par fichier).

        - 'baseline_t0' : date index t0, décalée selon `anchor` (voir
          src/cohorts/motor.py). Défaut 'day_start' = t0 00:00 : historique
          jusqu'à la fin de la veille, AUCUN événement du jour t0 (repli du
          protocole Section 4.1 : les diagnostics de t0 sont aussi exclus).
        - 'followup_6m' : date t0 + 6 mois (après ablation de la molécule d'intérêt).
        - 'followup_12m': date t0 + 12 mois (après ablation de la molécule d'intérêt).
        """
        if source == "stanford":
            if self.stanford_index is None:
                raise ValueError("stanford_index non initialisé.")
            df = self.stanford_index
            id_col = "person_id"

            if timepoint == "baseline_t0":
                time_col = "t0"
            elif timepoint == "followup_6m":
                time_col = "t_6m"
            elif timepoint == "followup_12m":
                time_col = "t_12m"
            else:
                raise ValueError(f"Timepoint inconnu : {timepoint}")

        elif source == "ukb":
            if self.ukb_index is None:
                raise ValueError("ukb_index non initialisé.")
            df = self.ukb_index
            id_col = "eid"
            time_col = "t0"
        else:
            raise ValueError(f"Source inconnue : {source}")

        from src.cohorts.motor import ANCHOR_OFFSETS_MIN

        if anchor not in ANCHOR_OFFSETS_MIN:
            raise ValueError(f"Ancrage inconnu : {anchor}")
        offset_min = ANCHOR_OFFSETS_MIN[anchor] if timepoint == "baseline_t0" else 0

        return (
            df.select(
                [
                    pl.col(id_col).cast(pl.Int64).alias("patient_id"),
                    (
                        pl.col(time_col).cast(pl.Datetime("us"))
                        + pl.duration(minutes=offset_min)
                    )
                    .dt.strftime("%Y-%m-%d %H:%M:00")
                    .alias("prediction_time"),
                ]
            )
            .unique(subset=["patient_id", "prediction_time"])
            .sort(["patient_id", "prediction_time"])
        )

    def __repr__(self) -> str:
        proto_str = (
            f", k_proto={len(self._prototypes)}"
            if self._prototypes is not None
            else ""
        )
        return (
            f"<DrugCohort id={self.drug_id} name='{self.name}' kind='{self.kind}' "
            f"(N_stanford={self.n_stanford:,}, N_ukb={self.n_ukb:,}{proto_str})>"
        )


# Le blanchiment de la Méthode 3 vit désormais dans src/pairs/motor_pairs.py ;
# ré-export pour compatibilité (`from src.cohorts.cohort import MotorWhitener`).
from src.pairs.motor_pairs import MotorWhitener  # noqa: E402,F401