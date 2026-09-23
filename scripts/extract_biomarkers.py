#!/usr/bin/env python3
"""Extraction massive et vectorisée des 6 biomarqueurs cliniques en passe unique.

Jointure globale entre l'ensemble des cohortes (N >= min_n) et le cache local
biomarkers_measurements.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import polars as pl
from tqdm import tqdm
from src.config import PathConfig


def run_batch_biomarker_extraction(min_n: int = 1) -> None:
    paths = PathConfig(is_sample=False)
    manifest_path = paths.output_cohorts_dir / "manifest.parquet"
    cache_bio = paths.cache_dir / "biomarkers_measurements.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")
    if not cache_bio.exists():
        raise FileNotFoundError(f"Cache local des biomarqueurs introuvable : {cache_bio}")

    print("=" * 90)
    print("EXTRACTION MASSIVE EN PASSE UNIQUE DES 6 BIOMARQUEURS (STARR)")
    print(f"Filtre cohortes : N >= {min_n} patients")
    print(f"Source mesures  : {cache_bio}")
    print("=" * 90)

    # 1. Sélection des cohortes cibles
    manifest = pl.read_parquet(manifest_path)
    eligible_manifest = manifest.filter(
        (pl.col("status") == "SAVED") & (pl.col("n_final_stanford") >= min_n)
    )
    eligible_ids = sorted(eligible_manifest["drug_id"].to_list())
    print(f"\n[1/5] Cohortes sélectionnées : {len(eligible_ids):,} / {manifest.height:,}")

    # 2. Consolidation rapide de tous les index patients avec typage strict
    print("[2/5] Chargement et indexation de toutes les inclusions t0...")
    t_start = time.time()
    index_frames = []

    for cid in tqdm(eligible_ids, desc="Lecture des stanford_index"):
        idx_p = paths.output_cohorts_dir / f"cohort_{cid}" / "stanford_index.parquet"
        if idx_p.exists():
            df_idx = pl.read_parquet(idx_p).select([
                pl.col("person_id").cast(pl.Int64),
                pl.col("t0").cast(pl.Date),
                pl.col("t_6m").cast(pl.Date),
                pl.col("t_12m").cast(pl.Date),
                pl.lit(cid, dtype=pl.Int64).alias("drug_id"),
            ])
            index_frames.append(df_idx)

    if not index_frames:
        print("[-] Aucun fichier stanford_index.parquet trouvé. Arrêt.")
        return

    all_targets_df = pl.concat(index_frames)
    print(f"      -> {all_targets_df.height:,} inclusions chargées en {time.time() - t_start:.1f}s")

    # 3. Moteur DuckDB pour la jointure temporelle globale
    con = duckdb.connect()
    con.execute("PRAGMA threads=16;")
    con.execute("PRAGMA memory_limit='40GB';")

    con.register("all_targets", all_targets_df)

    print("\n[3/5] Exécution de la passe unique de jointure sur 416M de mesures...")
    t_join = time.time()

    query = f"""
    WITH filtered_measurements AS (
        SELECT 
            person_id,
            measurement_date,
            CASE 
                WHEN measurement_concept_id IN (3028288, 3027597) THEN 'ldl'
                WHEN measurement_concept_id IN (3004410, 3005673, 40762352) THEN 'hba1c'
                WHEN measurement_concept_id IN (3049187, 3053283, 3030354, 40764999) THEN 'egfr'
                WHEN measurement_concept_id IN (3006923, 3000676) THEN 'alt'
                WHEN measurement_concept_id IN (3020460, 3010156, 3007461) THEN 'crp'
                WHEN measurement_concept_id IN (3004249) THEN 'sbp'
            END AS biomarker,
            value_as_number AS val
        FROM read_parquet('{cache_bio}')
        WHERE (
            (measurement_concept_id IN (3028288, 3027597) AND value_as_number BETWEEN 10.0 AND 400.0)
            OR (measurement_concept_id IN (3004410, 3005673, 40762352) AND value_as_number BETWEEN 3.0 AND 20.0)
            OR (measurement_concept_id IN (3049187, 3053283, 3030354, 40764999) AND value_as_number BETWEEN 3.0 AND 180.0)
            OR (measurement_concept_id IN (3006923, 3000676) AND value_as_number BETWEEN 2.0 AND 2000.0)
            OR (measurement_concept_id IN (3020460, 3010156, 3007461) AND value_as_number BETWEEN 0.05 AND 300.0)
            OR (measurement_concept_id IN (3004249) AND value_as_number BETWEEN 60.0 AND 260.0)
        )
    ),
    candidate_windows AS (
        SELECT 
            c.drug_id,
            c.person_id,
            m.biomarker,
            m.val,
            CASE 
                WHEN m.measurement_date >= (c.t0 - INTERVAL '90 days') AND m.measurement_date <= c.t0 THEN 'baseline'
                WHEN m.measurement_date >= (c.t0 + INTERVAL '120 days') AND m.measurement_date <= (c.t0 + INTERVAL '240 days') THEN 'm6'
                WHEN m.measurement_date >= (c.t0 + INTERVAL '300 days') AND m.measurement_date <= (c.t0 + INTERVAL '420 days') THEN 'm12'
            END AS window_name,
            CASE 
                WHEN m.measurement_date >= (c.t0 - INTERVAL '90 days') AND m.measurement_date <= c.t0 THEN ABS(c.t0 - m.measurement_date)
                WHEN m.measurement_date >= (c.t0 + INTERVAL '120 days') AND m.measurement_date <= (c.t0 + INTERVAL '240 days') THEN ABS(c.t_6m - m.measurement_date)
                WHEN m.measurement_date >= (c.t0 + INTERVAL '300 days') AND m.measurement_date <= (c.t0 + INTERVAL '420 days') THEN ABS(c.t_12m - m.measurement_date)
            END AS dist_to_target,
            m.measurement_date
        FROM all_targets c
        JOIN filtered_measurements m ON c.person_id = m.person_id
        WHERE m.measurement_date >= (c.t0 - INTERVAL '90 days')
          AND m.measurement_date <= (c.t0 + INTERVAL '420 days')
    ),
    best_per_window AS (
        SELECT 
            drug_id,
            person_id,
            biomarker,
            window_name,
            val,
            ROW_NUMBER() OVER (
                PARTITION BY drug_id, person_id, biomarker, window_name 
                ORDER BY dist_to_target ASC, measurement_date DESC
            ) AS rn
        FROM candidate_windows
        WHERE window_name IS NOT NULL
    )
    SELECT 
        drug_id,
        person_id,
        biomarker,
        MAX(CASE WHEN window_name = 'baseline' THEN val END) AS baseline_val,
        MAX(CASE WHEN window_name = 'm6' THEN val END) AS m6_val,
        MAX(CASE WHEN window_name = 'm12' THEN val END) AS m12_val
    FROM best_per_window
    WHERE rn = 1
    GROUP BY drug_id, person_id, biomarker;
    """

    con.execute(f"CREATE TEMP TABLE final_pairs AS {query}")
    print(f"      -> Jointure temporelle achevée en {(time.time() - t_join)/60:.2f} minutes.")

    # 4. Calcul vectoriel des deltas et transfert vers Polars
    print("\n[4/5] Calcul des deltas causaux et transfert vers Polars...")
    df_results = con.execute("""
        SELECT 
            drug_id,
            person_id,
            biomarker,
            baseline_val,
            m6_val,
            m12_val,
            (m6_val - baseline_val) AS delta_6m,
            (m12_val - baseline_val) AS delta_12m,
            (baseline_val IS NOT NULL AND m6_val IS NOT NULL) AS has_pair_6m,
            (baseline_val IS NOT NULL AND m12_val IS NOT NULL) AS has_pair_12m
        FROM final_pairs
    """).pl()

    print(f"      -> {df_results.height:,} lignes de biomarqueurs extraites.")

    # 5. Écriture par lot dans chaque répertoire de cohorte
    print("\n[5/5] Écriture des fichiers biomarkers.parquet par cohorte...")
    cohorts_written = 0

    for sub_df in tqdm(df_results.partition_by("drug_id"), desc="Écriture disque"):
        cid = sub_df["drug_id"][0]
        out_dir = paths.output_cohorts_dir / f"cohort_{cid}"
        if out_dir.exists():
            sub_df.drop("drug_id").write_parquet(out_dir / "biomarkers.parquet")
            cohorts_written += 1

    # Synthèse globale exportée
    summary_df = (
        df_results.group_by(["drug_id", "biomarker"])
        .agg([
            pl.col("baseline_val").is_not_null().sum().alias("n_baseline"),
            pl.col("m6_val").is_not_null().sum().alias("n_m6"),
            pl.col("m12_val").is_not_null().sum().alias("n_m12"),
            pl.col("has_pair_6m").sum().alias("n_complete_pair_6m"),
            pl.col("has_pair_12m").sum().alias("n_complete_pair_12m"),
        ])
        .sort(["drug_id", "biomarker"])
    )

    summary_path = paths.output_cohorts_dir / "biomarkers_summary.parquet"
    summary_df.write_parquet(summary_path)

    print("=" * 90)
    print(f"TRAITEMENT COMPLET EN {(time.time() - t_start)/60:.2f} MINUTES !")
    print(f"  * Cohortes enrichies avec biomarkers.parquet : {cohorts_written:,}")
    print(f"  * Synthèse globale exportée dans            : {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction globale des biomarqueurs.")
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Seuil minimal d'inclusions dans le manifeste (défaut: 1).",
    )
    args = parser.parse_args()

    run_batch_biomarker_extraction(min_n=args.min_n)