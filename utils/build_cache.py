import argparse
import sys
import time
from pathlib import Path
import duckdb

# Ajout dynamique de la racine au sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PathConfig

# Chemin vers les tables distantes STARR OMOP
OMOP_DIR = Path(
    "../../../remote/private/starr_omop_deid/ro/STARR_OMOP_tables/"
    "som-rit-phi-starr-prod.starr_omop_cdm54_confidential_lite_2026_07_22"
)


def build_cache(sample_ratio: float = 1.0) -> None:
    """
    Génère le cache Parquet local optimisé pour l'extraction de cohortes.
    
    :param sample_ratio: 1.0 pour l'exhaustivité (100%), ou float < 1.0 (ex. 0.2 pour 20%).
    """
    start_time = time.time()
    paths = PathConfig()

    is_sample = sample_ratio < 1.0
    # Cible le dossier approprié (data/cache_benchmark si échantillon, sinon data/cache_full)
    target_dir = (
        paths.cache_dir if is_sample else Path("data/cache_full")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    obs_glob = str(OMOP_DIR / "observation_period" / "*.csv.zst")
    drug_glob = str(OMOP_DIR / "drug_exposure" / "*.csv.zst")
    map_file = str(paths.mapping_path)

    obs_out = target_dir / "sampled_observation_period.parquet"
    drug_out = target_dir / "sampled_drug_exposure.parquet"

    print(f"\n{'='*75}")
    mode_str = f"ÉCHANTILLON {int(sample_ratio * 100)}%" if is_sample else "PRODUCTION (100% EXHAUSTIF)"
    print(f"CONSTRUCTION DU CACHE LOCAL [{mode_str}]")
    print(f"Dossier de destination : {target_dir}")
    print(f"{'='*75}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=16;")
    con.execute("PRAGMA max_memory='48GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Table d'observation et cohorte de patients
    # -------------------------------------------------------------------------
    print("\n--- 1. Extraction de observation_period ---")
    t0_step = time.time()

    if is_sample:
        hash_threshold = int(sample_ratio * 100)
        filter_clause = f"WHERE (abs(hash(person_id)) % 100) < {hash_threshold}"
    else:
        filter_clause = ""

    obs_query = f"""
        CREATE OR REPLACE TEMP TABLE base_cohort AS
        SELECT 
            person_id,
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
    print(f"✓ Fichier sauvegardé : {obs_out}")

    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Table drug_exposure filtrée et formatée
    # -------------------------------------------------------------------------
    print("\n--- 2. Extraction filtrée de drug_exposure avec calcul de end_date ---")
    t0_step = time.time()

    drug_query = f"""
        COPY (
            SELECT 
                de.person_id,
                de.drug_concept_id,
                TRY_CAST(de.drug_exposure_start_date AS DATE) AS exp_date,
                COALESCE(
                    TRY_CAST(de.drug_exposure_end_date AS DATE),
                    TRY_CAST(de.drug_exposure_start_date AS DATE) + COALESCE(TRY_CAST(de.days_supply AS INTEGER), 30)
                ) AS end_date
            FROM read_csv('{drug_glob}', auto_detect=true) de
            SEMI JOIN base_cohort bc 
                   ON de.person_id = bc.person_id
            SEMI JOIN read_parquet('{map_file}') m 
                   ON de.drug_concept_id = m.drug_concept_id
            WHERE TRY_CAST(de.drug_exposure_start_date AS DATE) IS NOT NULL
        ) TO '{drug_out}' (FORMAT PARQUET);
    """
    con.execute(drug_query)
    
    n_records = con.execute(f"SELECT COUNT(*) FROM read_parquet('{drug_out}')").fetchone()[0]
    print(f"✓ Expositions indexées : {n_records:,} (en {time.time() - t0_step:.1f}s)")
    print(f"✓ Fichier sauvegardé : {drug_out}")

    elapsed = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"Cache généré avec succès en {elapsed/60:.2f} minutes.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Création du cache Parquet local STARR OMOP.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Ratio d'échantillonnage des patients (ex: 0.2 pour 20%%, 1.0 pour la totalité). Défaut: 1.0",
    )
    args = parser.parse_args()
    build_cache(sample_ratio=args.ratio)