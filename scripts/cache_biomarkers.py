#!/usr/bin/env python3
"""Génération du cache local Parquet pour les 6 biomarqueurs cibles.

Scanne les 5000 shards .csv.zst de STARR OMOP et filtre exclusivement sur les
concepts LDL, HbA1c, eGFR, ALT, CRP et SBP.
"""

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import duckdb
from src.config import PathConfig

# Concepts cibles du protocole
# scripts/cache_biomarkers.py (extrait de TARGET_CONCEPTS)

# Concepts cibles du protocole
TARGET_CONCEPTS = [
    # LDL Cholesterol
    3028288,
    3027597,  # Ajout : concept LDL ciblé par extract_biomarkers.py
    3027114,
    3013444,
    3025809,
    # HbA1c
    3004410,
    3005673,
    40762352,
    # eGFR
    3049187,
    3053283,
    3030354,
    40764999,
    # ALT
    3006923,
    3000676,
    # CRP
    3020460,
    3010156,
    3007461,
    # Systolic BP
    3004249,
    3012888,
    3034219,
]

concepts_sql = ", ".join(str(c) for c in TARGET_CONCEPTS)


def build_biomarkers_cache():
  paths = PathConfig(is_sample=False)
  input_pattern = (
      "/remote/private/starr_omop_deid/ro/STARR_OMOP_tables/"
      "som-rit-phi-starr-prod.starr_omop_cdm54_confidential_lite_2026_07_22/"
      "measurement/*.csv.zst"
  )
  output_parquet = paths.cache_dir / "biomarkers_measurements.parquet"

  print("=" * 80)
  print("CRÉATION DU CACHE LOCAL DES BIOMARQUEURS (STARR)")
  print(f"Source distale : {input_pattern}")
  print(f"Sortie locale  : {output_parquet}")
  print(f"Concepts cibles: {len(TARGET_CONCEPTS)} codes OMOP")
  print("=" * 80)

  con = duckdb.connect()
  con.execute("PRAGMA threads=16;")
  con.execute("PRAGMA memory_limit='48GB';")

  start_time = time.time()

  # Lecture tolérante des 5000 shards et écriture en streaming
  query = f"""
    COPY (
        SELECT 
            TRY_CAST(person_id AS BIGINT) AS person_id,
            TRY_CAST(measurement_date AS DATE) AS measurement_date,
            TRY_CAST(measurement_concept_id AS BIGINT) AS measurement_concept_id,
            TRY_CAST(value_as_number AS DOUBLE) AS value_as_number
        FROM read_csv(
            '{input_pattern}',
            header = true,
            quote = '"',
            escape = '"',
            null_padding = true,
            ignore_errors = true
        )
        WHERE TRY_CAST(measurement_concept_id AS BIGINT) IN ({concepts_sql})
          AND TRY_CAST(value_as_number AS DOUBLE) IS NOT NULL
    ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

  print("\nExtraction et écriture en streaming Parquet en cours...")
  con.execute(query)

  elapsed = time.time() - start_time
  size_mb = output_parquet.stat().st_size / (1024 * 1024)

  # Métriques de volumétrie
  n_rows = con.execute(
      f"SELECT COUNT(*) FROM read_parquet('{output_parquet}')"
  ).fetchone()[0]

  print(f"\nCache créé avec succès en {elapsed/60:.2f} minutes !")
  print(f"  * Lignes retenues : {n_rows:,}")
  print(f"  * Poids du fichier: {size_mb:.1f} Mo")


if __name__ == "__main__":
  build_biomarkers_cache()