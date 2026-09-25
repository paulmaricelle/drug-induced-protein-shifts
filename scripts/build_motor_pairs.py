# scripts/build_motor_pairs.py
"""Méthode 3 (Section 4.2) : paires candidates par prototypes k-means sur les
représentations MOTOR z0 blanchies.

Étapes (chacune mise en cache dans --out-dir, réutilisable avec --reuse) :
  1. whitener.npz   : blanchiment global ajusté sur un échantillon des cohortes ;
  2. prototypes.npz : banque de prototypes (centroïdes, poids, effectifs) ;
  3. distances.npz  : matrice min-linkage C x C et clusters appariés ;
  4. method3_pairs.parquet : paires retenues par la règle de décision ;
  5. (--register) fusion dans data/candidate_pairs.parquet (by_kmeans=True).

Exemples :
  python scripts/build_motor_pairs.py --min-n 100
  python scripts/build_motor_pairs.py --reuse --rule threshold_topn --calibrate-atc4 0.5
  python scripts/build_motor_pairs.py --reuse --max-dist 0.3 --register

Aucune donnée patient n'est affichée : uniquement des effectifs, des quantiles
de distances et des noms de molécules.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from src.catalog.catalog import DrugCatalog
from src.config import PathConfig
from src.pairs.motor_pairs import (
    MotorWhitener,
    PrototypeBank,
    build_prototype_bank,
    cluster_distances,
    fit_reference_whitener,
    npy_embedding_loader,
    select_pairs,
    store_embedding_loader,
    threshold_from_reference,
)
from src.pairs.pair import CandidatePair, PairRegistry


def parse_args():
    p = argparse.ArgumentParser(
        description="Méthode 3 : paires par k-means sur MOTOR z0 blanchi."
    )
    g = p.add_argument_group("Entrées")
    g.add_argument("--min-n", type=int, default=100,
                   help="Effectif minimal de la cohorte (défaut: 100)")
    g.add_argument("--source", choices=["store", "npy"], default="store",
                   help="Magasin central (défaut) ou fichiers .npy par cohorte")
    g.add_argument("--store-dir", type=Path,
                   default=ROOT_DIR / "data" / "motor" / "store_day_start",
                   help="Magasin de scripts/extract_motor_representations.py")
    g.add_argument("--npy-name", default="stanford_motor_z0.npy",
                   help="Nom du fichier .npy dans cohort_<id>/ (--source npy)")

    g = p.add_argument_group("Blanchiment")
    g.add_argument("--whiten", choices=["pca", "zca", "center", "none"],
                   default="pca")
    g.add_argument("--n-components", type=float, default=128,
                   help="Axes conservés : entier, ou fraction de variance si < 1"
                        " (défaut: 128)")
    g.add_argument("--shrinkage", type=float, default=0.0)
    g.add_argument("--whiten-per-cohort", type=int, default=200,
                   help="Patients tirés par cohorte pour l'ajustement (défaut: 200)")
    g.add_argument("--whiten-weighting",
                   choices=["balanced", "sqrt", "proportional"],
                   default="balanced")

    g = p.add_argument_group("k-means")
    g.add_argument("--k", default="3",
                   help="Entier, 'auto' (règle de taille) ou 'bic' (défaut: 3)")
    g.add_argument("--k-max", type=int, default=5)
    g.add_argument("--min-cluster-patients", type=int, default=50)
    g.add_argument("--max-fit-samples", type=int, default=20000)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Distances et règle de décision")
    g.add_argument("--metric", default="mahalanobis_debiased",
                   choices=["mahalanobis_debiased", "mahalanobis", "cosine"])
    g.add_argument("--min-cluster-weight", type=float, default=0.05,
                   help="Poids minimal d'un cluster pour porter une paire")
    g.add_argument("--rule", default="threshold_topn",
                   choices=["threshold", "topn", "threshold_topn", "mutual_topn"])
    g.add_argument("--top-n", type=int, default=10)
    g.add_argument("--max-dist", type=float, default=None)
    g.add_argument("--calibrate-atc4", type=float, default=None, metavar="Q",
                   help="max_dist = quantile Q des distances des paires intra-ATC4")

    g = p.add_argument_group("Sorties")
    g.add_argument("--out-dir", type=Path, default=ROOT_DIR / "data" / "motor_pairs")
    g.add_argument("--reuse", action="store_true",
                   help="Réutilise whitener / prototypes / distances existants")
    g.add_argument("--register", action="store_true",
                   help="Fusionne les paires dans data/candidate_pairs.parquet")
    return p.parse_args()


def main():
    args = parse_args()
    paths = PathConfig(is_sample=False)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    k = int(args.k) if args.k.isdigit() else args.k
    n_comp = int(args.n_components) if args.n_components >= 1 else args.n_components

    manifest = pl.read_parquet(paths.output_cohorts_dir / "manifest.parquet")
    eligible = manifest.filter(
        (pl.col("status") == "SAVED") & (pl.col("n_final_stanford") >= args.min_n)
    )
    ids = eligible["drug_id"].to_list()
    sizes = dict(zip(ids, eligible["n_final_stanford"].to_list()))
    info = {
        r["drug_id"]: (r["drug_name"], r["n_final_stanford"])
        for r in manifest.iter_rows(named=True)
    }
    print(f"Méthode 3 : {len(ids):,} cohortes éligibles (N >= {args.min_n}).")

    wh_f, bank_f, dist_f = (
        out / "whitener.npz", out / "prototypes.npz", out / "distances.npz"
    )
    loader = None

    def get_loader():
        nonlocal loader
        if loader is None:
            t = time.time()
            loader = (
                store_embedding_loader(paths.output_cohorts_dir, args.store_dir, ids)
                if args.source == "store"
                else npy_embedding_loader(paths.output_cohorts_dir, args.npy_name)
            )
            print(f"  chargeur prêt ({time.time() - t:.0f}s)")
        return loader

    # 1. Blanchiment
    if args.reuse and wh_f.exists():
        whitener = MotorWhitener.load(wh_f)
    elif args.whiten == "none":
        whitener = None
    else:
        t = time.time()
        whitener = fit_reference_whitener(
            ids, get_loader(), sizes=sizes,
            n_per_cohort=args.whiten_per_cohort,
            weighting=args.whiten_weighting, seed=args.seed,
            mode=args.whiten, n_components=n_comp, shrinkage=args.shrinkage,
        )
        whitener.save(wh_f)
        print(f"  blanchiment ajusté en {time.time() - t:.0f}s")
    if whitener is not None:
        print(f"  {whitener} ; variance expliquée ="
              f" {whitener.meta.get('explained_variance', float('nan')):.3f}")

    # 2. Prototypes
    if args.reuse and bank_f.exists():
        bank = PrototypeBank.load(bank_f)
    else:
        t = time.time()
        bank = build_prototype_bank(
            ids, get_loader(), whitener, k=k, k_max=args.k_max,
            min_cluster_patients=args.min_cluster_patients,
            max_fit_samples=args.max_fit_samples, seed=args.seed,
        )
        bank.save(bank_f)
        print(f"  prototypes calculés en {time.time() - t:.0f}s")
    kc = np.bincount(bank.k)
    print(f"  banque : {len(bank):,} cohortes ; distribution de k : "
          + ", ".join(f"k={i}: {c}" for i, c in enumerate(kc) if c))

    # 3. Distances min-linkage
    if args.reuse and dist_f.exists():
        f = np.load(dist_f)
        dist, cl_a, cl_b = f["dist"], f["cl_a"], f["cl_b"]
        if str(f["metric"]) != args.metric:
            print(f"  [Avertissement] distances réutilisées calculées avec"
                  f" metric={f['metric']} (demandé : {args.metric}) ;"
                  f" supprimer {dist_f} pour recalculer.")
    else:
        t = time.time()
        dist, cl_a, cl_b = cluster_distances(
            bank, metric=args.metric, min_cluster_weight=args.min_cluster_weight
        )
        np.savez_compressed(dist_f, dist=dist.astype(np.float32), cl_a=cl_a,
                            cl_b=cl_b, metric=np.array(args.metric))
        print(f"  distances calculées en {time.time() - t:.1f}s")
    iu = np.triu_indices(len(bank), 1)
    q = np.quantile(dist[iu][np.isfinite(dist[iu])], [0.01, 0.05, 0.25, 0.5, 0.75])
    print("  quantiles de d(A, B) [1%, 5%, 25%, 50%, 75%] : "
          + ", ".join(f"{v:.3f}" for v in q))

    # 4. Règle de décision
    max_dist = args.max_dist
    if args.calibrate_atc4 is not None:
        catalog = DrugCatalog.load(paths.catalog_path)
        catalog.register_combination_cohorts(paths.output_cohorts_dir)
        present = set(bank.drug_ids.tolist())
        ref = {
            (min(a, b), max(a, b))
            for a in present
            for b in catalog.get_atc4_family_ids(a)
            if b in present and a != b
        }
        max_dist = threshold_from_reference(
            dist, bank.drug_ids, ref, quantile=args.calibrate_atc4
        )
        print(f"  max_dist calibré sur {len(ref):,} paires intra-ATC4"
              f" (quantile {args.calibrate_atc4}) : {max_dist:.3f}")

    pairs = select_pairs(
        bank, dist, cl_a, cl_b, rule=args.rule, max_dist=max_dist,
        top_n=args.top_n, metric=args.metric,
        min_cluster_weight=args.min_cluster_weight,
    ).with_columns(
        pl.col("drug_id_a").map_elements(lambda x: info.get(x, ("",))[0],
                                         return_dtype=pl.Utf8).alias("drug_name_a"),
        pl.col("drug_id_b").map_elements(lambda x: info.get(x, ("",))[0],
                                         return_dtype=pl.Utf8).alias("drug_name_b"),
    )
    pairs.write_parquet(out / "method3_pairs.parquet")
    print(f"  -> {pairs.height:,} paires retenues (règle={args.rule},"
          f" top_n={args.top_n}, max_dist={max_dist}) :"
          f" {out / 'method3_pairs.parquet'}")

    # 5. Registre
    if args.register:
        catalog = DrugCatalog.load(paths.catalog_path)
        catalog.register_combination_cohorts(paths.output_cohorts_dir)
        pairs_file = paths.root_dir / "data" / "candidate_pairs.parquet"
        registry = PairRegistry.load_parquet(pairs_file)
        has_shared = "kmeans_shared_weight_a" in pairs.columns
        for r in pairs.iter_rows(named=True):
            registry.add_or_update(CandidatePair(
                drug_id_a=r["drug_id_a"],
                drug_id_b=r["drug_id_b"],
                stratum_concept_id=None,  # clé actuelle : (a, b, 0)
                by_kmeans=True,
                kmeans_min_dist=float(r["kmeans_min_dist"]),
                kmeans_cluster_a=r["kmeans_cluster_a"],
                kmeans_cluster_b=r["kmeans_cluster_b"],
                kmeans_weight_a=r["kmeans_weight_a"],
                kmeans_weight_b=r["kmeans_weight_b"],
                kmeans_shared_weight_a=r["kmeans_shared_weight_a"] if has_shared else None,
                kmeans_shared_weight_b=r["kmeans_shared_weight_b"] if has_shared else None,
                drug_name_a=r["drug_name_a"],
                drug_name_b=r["drug_name_b"],
                n_patients_a=info.get(r["drug_id_a"], ("", 0))[1],
                n_patients_b=info.get(r["drug_id_b"], ("", 0))[1],
            ))
        registry.annotate_ingredient_overlap(catalog)
        registry.save_parquet(pairs_file)

        # Aperçu du consensus au niveau (a, b), strates confondues
        df = registry.to_dataframe().group_by(["drug_id_a", "drug_id_b"]).agg(
            pl.col("by_indication").any(), pl.col("by_kmeans").any(),
            pl.col("by_atc4").any(),
        )
        # La clé est invariante par permutation mais l'orientation stockée peut
        # varier : on canonicalise avant de compter
        df = df.with_columns(
            pl.min_horizontal("drug_id_a", "drug_id_b").alias("u"),
            pl.max_horizontal("drug_id_a", "drug_id_b").alias("v"),
        ).group_by(["u", "v"]).agg(
            pl.col("by_indication").any(), pl.col("by_kmeans").any(),
            pl.col("by_atc4").any(),
        )
        print(f"  Registre (niveau (a, b), strates confondues) :"
              f" Méthode 3 = {df['by_kmeans'].sum():,},"
              f" Méthode 3 & Méthode 1 = {(df['by_kmeans'] & df['by_indication']).sum():,},"
              f" Méthode 3 & ATC4 = {(df['by_kmeans'] & df['by_atc4']).sum():,}")

    if pairs.height:
        print("\nPaires les plus proches :")
        print(pairs.select(
            "drug_name_a", "drug_name_b", "kmeans_min_dist",
            "kmeans_weight_a", "kmeans_weight_b",
        ).head(10))


if __name__ == "__main__":
    main()
