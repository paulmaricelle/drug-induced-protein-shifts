# scripts/build_cache.py
import argparse
import sys
import time
from pathlib import Path
import duckdb
import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.catalog.catalog import DrugCatalog
from src.config import PathConfig


def ensure_mapping_file(paths: PathConfig) -> None:
    """Génère data/ingredient_to_prescriptions.parquet depuis DrugCatalog s'il est absent."""
    if paths.mapping_path.exists():
        return

    print("Génération de ingredient_to_prescriptions.parquet depuis le DrugCatalog...")
    paths.mapping_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = DrugCatalog.load(paths.catalog_path)

    records = []
    for item in catalog:
        is_mono = item.kind == "monotherapy"
        for rx_id in item.descendant_concept_ids:
            records.append(
                {
                    "drug_concept_id": int(rx_id),
                    "ingredient_id": int(item.drug_id),
                    "is_monotherapy": is_mono,
                }
            )

    df_mapping = pl.DataFrame(
        records,
        schema={
            "drug_concept_id": pl.Int64,
            "ingredient_id": pl.Int64,
            "is_monotherapy": pl.Boolean,
        },
    )
    df_mapping.write_parquet(paths.mapping_path)
    print(f"✓ Mapping sauvegardé : {paths.mapping_path} ({len(df_mapping):,} codes)")


def build_cache(sample_ratio: float = 1.0) -> None:
    start_time = time.time()
    paths = PathConfig(is_sample=(sample_ratio < 1.0))
    ensure_mapping_file(paths)

    target_dir = paths.cache_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    obs_glob = str(paths.omop_dir / "observation_period" / "*.csv.zst")
    drug_glob = str(paths.omop_dir / "drug_exposure" / "*.csv.zst")
    map_file = str(paths.mapping_path)

    obs_out = paths.observation_period_parquet
    drug_out = paths.drug_exposure_parquet
    drug_tmp = drug_out.with_suffix(".tmp.parquet")

    print(f"\n{'='*75}")
    mode_str = (
        f"ÉCHANTILLON {int(sample_ratio * 100)}%"
        if sample_ratio < 1.0
        else "PRODUCTION (100% EXHAUSTIF)"
    )
    print(f"CONSTRUCTION DU CACHE LOCAL [{mode_str}]")
    print(f"Destination : {target_dir}")
    print(f"{'='*75}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=16;")
    con.execute("PRAGMA max_memory='48GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    # 1. observation_period
    print("\n--- 1. Extraction de observation_period ---")
    t0_step = time.time()
    filter_clause = (
        f"WHERE (abs(hash(person_id)) % 100) < {int(sample_ratio * 100)}"
        if sample_ratio < 1.0
        else ""
    )

    obs_query = f"""
        CREATE OR REPLACE TEMP TABLE base_cohort AS
        SELECT 
            person_id::BIGINT AS person_id,
            MIN(TRY_CAST(observation_period_start_date AS DATE)) AS obs_start,
            MAX(TRY_CAST(observation_period_end_date AS DATE)) AS obs_end
        FROM read_csv('{obs_glob}', auto_detect=true)
        {filter_clause}
        GROUP BY person_id;
    """
    con.execute(obs_query)
    n_patients = con.execute("SELECT COUNT(*) FROM base_cohort").fetchone()[0]
    print(f"✓ Patients retenus : {n_patients:,} (en {time.time() - t0_step:.1f}s)")
    con.execute(f"COPY base_cohort TO '{obs_out}' (FORMAT PARQUET);")

    # 2. drug_exposure (filtré par SEMI JOIN sur le mapping)
    print("\n--- 2. Extraction filtrée de drug_exposure ---")
    t0_step = time.time()

    drug_query = f"""
        COPY (
            SELECT 
                de.person_id::BIGINT AS person_id,
                de.drug_concept_id::BIGINT AS drug_concept_id,
                TRY_CAST(de.drug_exposure_start_date AS DATE) AS exp_date,
                m.ingredient_id::BIGINT AS ingredient_id,
                m.is_monotherapy::BOOLEAN AS is_monotherapy,
                -- Informations connues à t0 pour classer une prescription ponctuelle
                TRY_CAST(de.drug_exposure_end_date AS DATE) AS exp_end_date,
                TRY_CAST(de.drug_type_concept_id AS BIGINT) AS drug_type_concept_id,
                TRY_CAST(de.route_concept_id AS BIGINT) AS route_concept_id,
                TRY_CAST(de.refills AS INTEGER) AS refills
            FROM read_csv('{drug_glob}', auto_detect=true, union_by_name=true) de
            SEMI JOIN base_cohort bc 
                ON de.person_id = bc.person_id
            JOIN read_parquet('{map_file}') m 
            ON de.drug_concept_id = m.drug_concept_id
            WHERE TRY_CAST(de.drug_exposure_start_date AS DATE) IS NOT NULL
        ) TO '{drug_tmp}' (FORMAT PARQUET);
    """
    # Écriture dans un fichier temporaire : le cache existant reste valide en cas d'échec
    con.execute(drug_query)
    drug_tmp.replace(drug_out)
    n_records = con.execute(f"SELECT COUNT(*) FROM read_parquet('{drug_out}')").fetchone()[0]
    print(f"✓ Expositions indexées : {n_records:,} (en {time.time() - t0_step:.1f}s)")

    print(f"\n{'='*75}")
    print(f"Cache généré en {(time.time() - start_time)/60:.2f} minutes.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Création du cache local STARR OMOP.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Ratio d'échantillonnage (ex: 0.2 pour 20%% benchmark, 1.0 pour 100%% production).",
    )
    args = parser.parse_args()
    build_cache(sample_ratio=args.ratio)