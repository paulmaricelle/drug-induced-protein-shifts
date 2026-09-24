#!/usr/bin/env python3
"""Génération du cache local Parquet pour les 6 biomarqueurs cibles.

Scanne les shards .csv.zst de STARR OMOP et filtre exclusivement sur les
concepts LDL, HbA1c, eGFR, ALT, CRP et SBP.
"""

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import duckdb
from src.cohorts.biomarkers import all_concept_ids
from src.config import PathConfig

# Concepts cibles du protocole : source unique src/cohorts/biomarkers.py
TARGET_CONCEPTS = all_concept_ids()
concepts_sql = ", ".join(str(c) for c in TARGET_CONCEPTS)


def build_biomarkers_cache():
  paths = PathConfig(is_sample=False)
  input_pattern = str(paths.omop_dir / "measurement" / "*.csv.zst")
  output_parquet = paths.cache_dir / "biomarkers_measurements.parquet"
  tmp_parquet = output_parquet.with_suffix(".tmp.parquet")

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
            TRY_CAST(value_as_number AS DOUBLE) AS value_as_number,
            -- Unité indispensable à l'harmonisation (CRP en mg/dL ou mg/L, etc.)
            TRY_CAST(unit_concept_id AS BIGINT) AS unit_concept_id
        FROM read_csv(
            '{input_pattern}',
            header = true,
            quote = '"',
            escape = '"',
            null_padding = true,
            ignore_errors = true,
            union_by_name = true
        )
        WHERE TRY_CAST(measurement_concept_id AS BIGINT) IN ({concepts_sql})
          AND TRY_CAST(value_as_number AS DOUBLE) IS NOT NULL
    ) TO '{tmp_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

  print("\nExtraction et écriture en streaming Parquet en cours...")
  con.execute(query)
  # Remplacement atomique : l'ancien cache reste valide en cas d'échec
  tmp_parquet.replace(output_parquet)

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