# scripts/build_candidate_pairs.py
import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.catalog.catalog import DrugCatalog
from src.config import PathConfig
from src.pairs.indication_pairs import extract_shared_indication_pairs
from src.pairs.pair import CandidatePair, PairRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Génération et sélection des paires comparatrices."
    )
    parser.add_argument(
        "--method1-indications",
        action="store_true",
        help="Exécute la Méthode 1 : indications et diagnostics partagés à t0",
    )
    parser.add_argument(
        "--intra-atc4",
        action="store_true",
        help="Génère directement les paires intra-classe ATC4",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="Effectif minimal de la cohorte pour être éligible (défaut: 100)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Affiche une synthèse des paires actuellement dans le registre",
    )
    return parser.parse_args()


def generate_intra_atc4_pairs(
    catalog: DrugCatalog, paths: PathConfig, min_n: int, registry: PairRegistry
):
    manifest = pl.read_parquet(paths.output_cohorts_dir / "manifest.parquet")
    eligible = manifest.filter(
        (pl.col("status") == "SAVED") & (pl.col("n_final_stanford") >= min_n)
    )
    eligible_ids = set(eligible["drug_id"].to_list())

    print(
        f"Génération des paires intra-ATC4 pour {len(eligible_ids):,} molécules"
        f" (N >= {min_n})..."
    )
    n_added = 0
    for drug_id in eligible_ids:
        item = catalog.get(drug_id)
        if not item or not item.atc4:
            continue
        family = catalog.get_atc4_family_ids(drug_id)
        for comp_id in family:
            if comp_id in eligible_ids and drug_id < comp_id:
                comp_item = catalog.get(comp_id)
                pair = CandidatePair(
                    drug_id_a=drug_id,
                    drug_id_b=comp_id,
                    stratum_concept_id=None,
                    stratum_name=f"Classe ATC4: {item.atc4}",
                    by_atc4=True,
                    drug_name_a=item.name,
                    drug_name_b=comp_item.name if comp_item else "",
                    n_patients_a=manifest.filter(pl.col("drug_id") == drug_id)[
                        "n_final_stanford"
                    ][0],
                    n_patients_b=manifest.filter(pl.col("drug_id") == comp_id)[
                        "n_final_stanford"
                    ][0],
                )
                registry.add_or_update(pair)
                n_added += 1
    print(f"  -> {n_added:,} paires intra-ATC4 enregistrées.")


def main():
    args = parse_args()
    paths = PathConfig(is_sample=False)
    catalog = DrugCatalog.load(paths.catalog_path)

    # Rattacher automatiquement les bi-thérapies de facto
    catalog.register_de_facto_cohorts(paths.output_cohorts_dir)

    pairs_file = paths.root_dir / "data" / "candidate_pairs.parquet"
    registry = PairRegistry.load_parquet(pairs_file)

    if args.intra_atc4:
        generate_intra_atc4_pairs(catalog, paths, args.min_n, registry)

    if args.method1_indications:
        extract_shared_indication_pairs(
            catalog=catalog,
            paths=paths,
            min_cohort_size=args.min_n,
            registry=registry,
        )

    # Toujours recalculé : les registres antérieurs n'ont pas ces colonnes
    registry.annotate_ingredient_overlap(catalog)
    registry.save_parquet(pairs_file)

    if args.audit or not (args.method1_indications or args.intra_atc4):
        print("\n" + "=" * 80)
        print(f"AUDIT DU REGISTRE DES PAIRES ({pairs_file})")
        print("=" * 80)
        print(f"Total paires candidates indexées : {len(registry):,}")
        df = registry.to_dataframe()
        if len(df) > 0:
            p_ind = df.filter(pl.col("by_indication")).height
            p_atc = df.filter(pl.col("by_atc4")).height
            p_both = df.filter(
                pl.col("by_indication") & pl.col("by_atc4")
            ).height
            print(
                f" - Paires issues des diagnostics (Méthode 1) :"
                f" {p_ind:,}"
            )
            print(
                f" - Paires intra-ATC4                         :"
                f" {p_atc:,}"
            )
            print(
                f" - Paires consensus (Méthode 1 & ATC4)       :"
                f" {p_both:,}"
            )
            p_addon = df.filter(pl.col("is_add_on")).height
            p_same = df.filter(pl.col("same_ingredients")).height
            print(
                f" - Paires add-on (A+B vs A)                  :"
                f" {p_addon:,}"
            )
            print(
                f" - Paires mêmes ingrédients (fixe/de facto)  :"
                f" {p_same:,}"
            )
            print("\nExemples de paires candidates :")
            cols = [
                "drug_name_a",
                "drug_name_b",
                "stratum_name",
                "n_patients_a",
                "n_patients_b",
            ]
            print(df.select([c for c in cols if c in df.columns]).head(10))
        print("=" * 80)


if __name__ == "__main__":
    main()