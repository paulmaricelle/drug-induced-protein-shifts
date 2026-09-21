from pathlib import Path
import sys
import numpy as np
import polars as pl

# Ancrage dynamique au sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from the_map.drugCatalog import DrugCatalog
    from the_map.config import PathConfig
except ImportError:
    from the_map.drugCatalog import DrugCatalog
    from the_map.config import PathConfig


def compute_group_stats(df: pl.DataFrame, group_col: str, min_patients: int = 10) -> None:
    # Filtre sur les molécules viables ayant un code de groupe renseigné
    filtered = df.filter(
        (pl.col("n_final_target") >= min_patients)
        & (pl.col(group_col).is_not_null())
        & (pl.col(group_col) != "")
    )

    # Agrégation : nombre de molécules viables par groupe
    grouped = (
        filtered.group_by(group_col)
        .agg([
            pl.len().alias("k_molecules"),
            pl.col("rxnorm_name").alias("drugs"),
            pl.col("n_final_target").sum().alias("total_patients"),
        ])
        .sort("k_molecules", descending=True)
    )

    k_counts = grouped["k_molecules"].to_numpy()
    if len(k_counts) == 0:
        print(f"Aucun groupe trouvé pour {group_col} avec >= {min_patients} patients.")
        return

    # Déciles (10% à 100%)
    deciles = np.arange(10, 101, 10)
    q_vals = np.percentile(k_counts, deciles)

    # Calcul des paires éligibles K * (K - 1) / 2
    pairs_per_group = [k * (k - 1) // 2 for k in k_counts if k >= 2]
    total_pairs = sum(pairs_per_group)
    viable_groups = (k_counts >= 2).sum()

    print(f"\n{'='*75}")
    print(f"ANALYSE DE REGROUPEMENT : {group_col.upper()} (Seuil: >= {min_patients} patients/molécule)")
    print(f"{'='*75}")
    print(f"Total molécules viables classées : {filtered.height:,}")
    print(f"Total groupes distincts          : {len(k_counts):,}")
    print(f"Groupes avec >= 2 molécules      : {viable_groups:,} ({(viable_groups/len(k_counts))*100:.1f}%)")
    print(f"Nombre total de paires (A, B)    : {total_pairs:,}")
    print(f"Moyenne molécules / groupe       : {k_counts.mean():.2f}")
    print(f"Écart-type                       : {k_counts.std():.2f}")
    print("-" * 75)
    print("DÉCILES DU NOMBRE DE MOLÉCULES PAR GROUPE :")
    print(f"  Min  : {k_counts.min():>4}")
    for d, val in zip(deciles, q_vals):
        print(f"  D{d:<3}: {val:>4.1f}")
    print(f"  Max  : {k_counts.max():>4}")

    print("-" * 75)
    print(f"RÉPARTITION PAR TAILLE DE GROUPE (K) :")
    for cutoff in [1, 2, 3, 5, 10, 15]:
        n_grp = (k_counts >= cutoff).sum()
        n_pairs = sum(k * (k - 1) // 2 for k in k_counts if k >= cutoff)
        print(f"  K >= {cutoff:<2} molécules : {n_grp:>4} groupes ({n_grp/len(k_counts)*100:>5.1f}%) -> {n_pairs:>5,} paires potentielles")

    print("-" * 75)
    print(f"TOP 10 DES PLUS GRANDS GROUPES {group_col.upper()} :")
    for i, row in enumerate(grouped.head(10).iter_rows(named=True), 1):
        k = row["k_molecules"]
        p = k * (k - 1) // 2
        drug_preview = ", ".join(row["drugs"][:3]) + (", ..." if k > 3 else "")
        print(f"  {i:>2}. {row[group_col]:<8} : {k:>2} molécules ({p:>3} paires, {row['total_patients']:>6,} pts) -> [{drug_preview}]")
    print("=" * 75)


def main(min_patients: int = 10) -> None:
    manifest_path = Path("data/cohorts/manifest.parquet")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste manquant : {manifest_path}")

    paths = PathConfig(is_sample=False)
    catalog = DrugCatalog.from_parquet(paths.catalog_path, paths.mapping_path)

    df_manifest = pl.read_parquet(manifest_path)

    # Récupération de l'ATC4 et de l'ATC3 directement depuis DrugCatalog
    cids = df_manifest["rxnorm_concept_id"].to_list()
    atc4_list = [catalog.get(cid).atc4 if cid in catalog else None for cid in cids]
    atc3_list = [catalog.get(cid).atc3 if cid in catalog else None for cid in cids]

    df_merged = df_manifest.with_columns([
        pl.Series("atc3_resolved", atc3_list),
        pl.Series("atc4_resolved", atc4_list),
    ])

    compute_group_stats(df_merged, group_col="atc4_resolved", min_patients=min_patients)
    compute_group_stats(df_merged, group_col="atc3_resolved", min_patients=min_patients)


if __name__ == "__main__":
    min_pts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(min_patients=min_pts)