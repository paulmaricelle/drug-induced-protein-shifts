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
from src.cohorts.biomarkers import BIOMARKERS, FEMALE, MALE, concept_table_records
from src.config import PathConfig

# Fenêtre de baseline [t0 - N j, t0] : mesure la plus proche de t0. La baseline n'est pas
# requise (issue = valeur à 6/12 mois) ; elle sert de covariable pronostique si présente.
BASELINE_DAYS = 180


def run_batch_biomarker_extraction(min_n: int = 1) -> None:
    paths = PathConfig(is_sample=False)
    manifest_path = paths.output_cohorts_dir / "manifest.parquet"
    cache_bio = paths.cache_dir / "biomarkers_measurements.parquet"
    demo_parquet = paths.cache_dir / "person_demographics.parquet"
    EGFR_LO, EGFR_HI = next(b.derived_range for b in BIOMARKERS if b.name == "egfr")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")
    if not demo_parquet.exists():
        raise FileNotFoundError(f"Démographie introuvable : {demo_parquet} (lancer cache_biomarkers.py)")
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
    # Règles d'harmonisation (concept x unité), cf. src/cohorts/biomarkers.py
    con.register("lab_rules", pl.DataFrame(concept_table_records(), schema={
        "concept_id": pl.Int64, "biomarker": pl.Utf8, "priority": pl.Int64,
        "lo": pl.Float64, "hi": pl.Float64, "offset": pl.Float64,
        "unit_concept_id": pl.Int64, "factor": pl.Float64, "excluded": pl.Boolean,
    }))

    print("\n[3/5] Exécution de la passe unique de jointure sur les mesures...")
    t_join = time.time()

    query = f"""
    WITH harmonized AS (
        -- Règle spécifique à l'unité si elle existe, sinon règle par défaut du concept
        SELECT
            m.person_id,
            m.measurement_date,
            COALESCE(ru.biomarker, rd.biomarker) AS biomarker,
            COALESCE(ru.priority, rd.priority) AS priority,
            COALESCE(ru.factor, rd.factor) * m.value_as_number
              + COALESCE(ru.offset, rd.offset) AS val,
            COALESCE(ru.excluded, rd.excluded) AS excluded,
            COALESCE(ru.lo, rd.lo) AS lo,
            COALESCE(ru.hi, rd.hi) AS hi
        FROM read_parquet('{cache_bio}') m
        LEFT JOIN lab_rules ru
          ON ru.concept_id = m.measurement_concept_id
         AND ru.unit_concept_id = m.unit_concept_id
        LEFT JOIN lab_rules rd
          ON rd.concept_id = m.measurement_concept_id
         AND rd.unit_concept_id IS NULL
        WHERE COALESCE(ru.concept_id, rd.concept_id) IS NOT NULL
    ),
    -- Une valeur par (patient, biomarqueur, jour) : concept prioritaire, moyenne des répétitions
    filtered_measurements AS (
        SELECT person_id, measurement_date, biomarker, AVG(val) AS val
        FROM (
            SELECT *, MIN(priority) OVER (
                PARTITION BY person_id, biomarker, measurement_date) AS best_priority
            FROM harmonized
            WHERE NOT excluded AND val BETWEEN lo AND hi
        )
        WHERE priority = best_priority
        GROUP BY 1, 2, 3
    ),
    -- eGFR CKD-EPI 2021 recalculé depuis la créatinine (adultes, sexe renseigné)
    derived_measurements AS (
        SELECT f.person_id, f.measurement_date, f.biomarker,
            CASE WHEN f.biomarker = 'egfr' THEN
                142.0
                * POW(LEAST(f.val / k, 1.0), a)
                * POW(GREATEST(f.val / k, 1.0), -1.200)
                * POW(0.9938, age)
                * CASE WHEN female THEN 1.012 ELSE 1.0 END
            ELSE f.val END AS val
        FROM (
            SELECT f.*,
                (d.gender_concept_id = {FEMALE}) AS female,
                CASE WHEN d.gender_concept_id = {FEMALE} THEN 0.7 ELSE 0.9 END AS k,
                CASE WHEN d.gender_concept_id = {FEMALE} THEN -0.241 ELSE -0.302 END AS a,
                DATE_DIFF('day', d.birth_date, f.measurement_date) / 365.25 AS age,
                d.gender_concept_id
            FROM filtered_measurements f
            LEFT JOIN read_parquet('{demo_parquet}') d ON d.person_id = f.person_id
        ) f
        WHERE f.biomarker <> 'egfr'
           OR (f.gender_concept_id IN ({FEMALE}, {MALE}) AND f.age >= 18)
    ),
    final_measurements AS (
        SELECT * FROM derived_measurements
        WHERE biomarker <> 'egfr' OR val BETWEEN {EGFR_LO} AND {EGFR_HI}
    ),
    candidate_windows AS (
        SELECT 
            c.drug_id,
            c.person_id,
            m.biomarker,
            m.val,
            CASE 
                WHEN m.measurement_date >= (c.t0 - INTERVAL '{BASELINE_DAYS} days') AND m.measurement_date <= c.t0 THEN 'baseline'
                WHEN m.measurement_date >= (c.t0 + INTERVAL '120 days') AND m.measurement_date <= (c.t0 + INTERVAL '240 days') THEN 'm6'
                WHEN m.measurement_date >= (c.t0 + INTERVAL '300 days') AND m.measurement_date <= (c.t0 + INTERVAL '420 days') THEN 'm12'
            END AS window_name,
            CASE 
                WHEN m.measurement_date >= (c.t0 - INTERVAL '{BASELINE_DAYS} days') AND m.measurement_date <= c.t0 THEN ABS(c.t0 - m.measurement_date)
                WHEN m.measurement_date >= (c.t0 + INTERVAL '120 days') AND m.measurement_date <= (c.t0 + INTERVAL '240 days') THEN ABS(c.t_6m - m.measurement_date)
                WHEN m.measurement_date >= (c.t0 + INTERVAL '300 days') AND m.measurement_date <= (c.t0 + INTERVAL '420 days') THEN ABS(c.t_12m - m.measurement_date)
            END AS dist_to_target,
            m.measurement_date
        FROM all_targets c
        JOIN final_measurements m ON c.person_id = m.person_id
        WHERE m.measurement_date >= (c.t0 - INTERVAL '{BASELINE_DAYS} days')
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
    print("\n[5/5] Écriture des fichiers stanford_labs.parquet par cohorte...")
    cohorts_written = 0
    # Purge préalable : une cohorte sans aucune mesure ne doit pas garder un fichier périmé
    for cid in eligible_ids:
        for name in ("stanford_labs.parquet", "biomarkers.parquet"):
            (paths.output_cohorts_dir / f"cohort_{cid}" / name).unlink(missing_ok=True)

    for sub_df in tqdm(df_results.partition_by("drug_id"), desc="Écriture disque"):
        cid = sub_df["drug_id"][0]
        out_dir = paths.output_cohorts_dir / f"cohort_{cid}"
        if out_dir.exists():
            sub_df.drop("drug_id").write_parquet(out_dir / "stanford_labs.parquet")
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
    print(f"  * Cohortes enrichies (stanford_labs.parquet) : {cohorts_written:,}")
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