# src/pairs/indication_pairs.py
from __future__ import annotations

from pathlib import Path
import time
import duckdb
import polars as pl

from src.catalog.catalog import DrugCatalog
from src.config import PathConfig
from src.pairs.pair import CandidatePair, PairRegistry


def extract_shared_indication_pairs(
    catalog: DrugCatalog,
    paths: PathConfig,
    min_cohort_size: int = 100,
    registry: PairRegistry | None = None,
) -> PairRegistry:
    """Implémente la Méthode 1 (Section 4.2) : identifie les paires candidates

    partageant un diagnostic majeur à ou avant t0 dans les dossiers
    hospitaliers.
    """
    registry = registry or PairRegistry()
    manifest_path = paths.output_cohorts_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")

    # Index des noms et effectifs du manifeste pour un fallback parfait sur les bi-thérapies
    manifest = pl.read_parquet(manifest_path)
    manifest_info = {
        row["drug_id"]: (row["drug_name"], row["n_final_stanford"])
        for row in manifest.iter_rows(named=True)
    }

    eligible = manifest.filter(
        (pl.col("status") == "SAVED")
        & (pl.col("n_final_stanford") >= min_cohort_size)
    )
    eligible_ids = set(eligible["drug_id"].to_list())
    print(
        f"Méthode 1 : Analyse de {len(eligible_ids):,} cohortes éligibles (N >="
        f" {min_cohort_size})..."
    )

    # 1. Consolidation des index temporels t0
    index_frames = []
    for cid in eligible_ids:
        idx_p = (
            paths.output_cohorts_dir / f"cohort_{cid}" / "stanford_index.parquet"
        )
        if idx_p.exists():
            df = pl.read_parquet(idx_p).select([
                pl.col("person_id").cast(pl.Int64),
                pl.col("t0").cast(pl.Date),
                pl.lit(cid, dtype=pl.Int64).alias("drug_id"),
            ])
            index_frames.append(df)

    if not index_frames:
        return registry

    all_inclusions = pl.concat(index_frames)

    # 2. DuckDB : Jointure sur condition_occurrence
    con = duckdb.connect()
    con.execute("PRAGMA threads=16;")
    con.execute("PRAGMA max_memory='40GB';")
    con.register("inclusions", all_inclusions)

    cond_glob = str(paths.omop_dir / "condition_occurrence" / "*.csv.zst")
    vocab_parquet = "/remote/shared/collab/omop-vocabularies/v20250227/CONCEPT.parquet"

    print("Recherche des diagnostics à ou avant t0 (fenêtre [t0 - 30j, t0])...")
    t0_start = time.time()

    query = f"""
    WITH raw_conditions AS (
        SELECT 
            TRY_CAST(co.person_id AS BIGINT) AS person_id,
            TRY_CAST(co.condition_concept_id AS BIGINT) AS condition_concept_id,
            TRY_CAST(co.condition_start_date AS DATE) AS cond_date
        FROM read_csv(
            '{cond_glob}',
            header = true,
            quote = '"',
            escape = '"',
            null_padding = true,
            ignore_errors = true
        ) co
        SEMI JOIN inclusions inc ON TRY_CAST(co.person_id AS BIGINT) = inc.person_id
        WHERE TRY_CAST(co.condition_concept_id AS BIGINT) IS NOT NULL
          AND TRY_CAST(co.condition_concept_id AS BIGINT) != 0
    ),
    -- Diagnostics enregistrés entre -30j et t0
    cohort_conditions AS (
        SELECT 
            inc.drug_id,
            inc.person_id,
            rc.condition_concept_id
        FROM inclusions inc
        JOIN raw_conditions rc 
          ON inc.person_id = rc.person_id
         AND rc.cond_date >= (inc.t0 - INTERVAL '30 days')
         AND rc.cond_date <= inc.t0
        GROUP BY inc.drug_id, inc.person_id, rc.condition_concept_id
    ),
    global_condition_prevalence AS (
        SELECT 
            condition_concept_id,
            COUNT(DISTINCT person_id)::FLOAT / (SELECT COUNT(DISTINCT person_id) FROM inclusions) AS global_prev
        FROM cohort_conditions
        GROUP BY condition_concept_id
        HAVING COUNT(DISTINCT person_id) >= 50
    ),
    prevalence_per_drug AS (
        SELECT 
            cc.drug_id,
            cc.condition_concept_id,
            COUNT(DISTINCT cc.person_id)::FLOAT / man.n_final_stanford AS prevalence,
            man.n_final_stanford AS total_drug_patients,
            (COUNT(DISTINCT cc.person_id)::FLOAT / man.n_final_stanford) / gcp.global_prev AS lift
        FROM cohort_conditions cc
        JOIN read_parquet('{manifest_path}') man ON cc.drug_id = man.drug_id
        JOIN global_condition_prevalence gcp ON cc.condition_concept_id = gcp.condition_concept_id
        GROUP BY cc.drug_id, cc.condition_concept_id, man.n_final_stanford, gcp.global_prev
        HAVING (COUNT(DISTINCT cc.person_id)::FLOAT / man.n_final_stanford) >= 0.15
           AND ((COUNT(DISTINCT cc.person_id)::FLOAT / man.n_final_stanford) / gcp.global_prev) >= 2.0
    ),
    top_indications AS (
        SELECT 
            p.*,
            c.concept_name,
            c.concept_class_id
        FROM prevalence_per_drug p
        JOIN read_parquet('{vocab_parquet}') c ON p.condition_concept_id = c.concept_id
        WHERE c.concept_name NOT ILIKE '%abnormal%'
          AND c.concept_name NOT ILIKE '%finding%'
          AND c.concept_name NOT IN ('Illness', 'Chronic pain', 'Dyspnea', 'Fatigue', 'Edema', 'Chest pain')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY p.drug_id ORDER BY p.lift DESC, p.prevalence DESC) <= 3
    ),
    -- Filtre de spécificité : une vraie indication regroupe entre 2 et 35 molécules alternatives
    specific_strata AS (
        SELECT condition_concept_id
        FROM top_indications
        GROUP BY condition_concept_id
        HAVING COUNT(DISTINCT drug_id) BETWEEN 2 AND 35
    )
    SELECT 
        p1.drug_id AS drug_id_a,
        p2.drug_id AS drug_id_b,
        p1.condition_concept_id AS stratum_concept_id,
        p1.concept_name AS stratum_name,
        p1.prevalence AS prevalence_a,
        p2.prevalence AS prevalence_b,
        p1.total_drug_patients AS n_patients_a,
        p2.total_drug_patients AS n_patients_b
    FROM top_indications p1
    JOIN top_indications p2 
      ON p1.condition_concept_id = p2.condition_concept_id
     AND p1.drug_id < p2.drug_id
    SEMI JOIN specific_strata s 
      ON p1.condition_concept_id = s.condition_concept_id;
    """
    
    res_df = con.execute(query).pl()
    print(
        f"  -> {len(res_df):,} paires candidates extraites en"
        f" {time.time() - t0_start:.1f}s."
    )

    # 3. Enregistrement avec résolution propre des noms
    # 3. Enregistrement avec résolution propre des noms et du consensus ATC4
    for row in res_df.iter_rows(named=True):
        aid, bid = row["drug_id_a"], row["drug_id_b"]
        item_a = catalog.get(aid)
        item_b = catalog.get(bid)

        name_a = (
            item_a.name
            if item_a
            else manifest_info.get(aid, (f"Drug_{aid}", 0))[0]
        )
        name_b = (
            item_b.name
            if item_b
            else manifest_info.get(bid, (f"Drug_{bid}", 0))[0]
        )

        # Détection immédiate du partage de classe ATC4
        is_same_atc4 = bool(
            item_a
            and item_b
            and item_a.atc4
            and item_b.atc4
            and str(item_a.atc4)[:5].upper() == str(item_b.atc4)[:5].upper()
        )

        pair = CandidatePair(
            drug_id_a=aid,
            drug_id_b=bid,
            stratum_concept_id=row["stratum_concept_id"],
            stratum_name=row["stratum_name"],
            by_indication=True,
            by_atc4=is_same_atc4,  # <-- Flag consensus positionné ici !
            prevalence_a=row["prevalence_a"],
            prevalence_b=row["prevalence_b"],
            drug_name_a=name_a,
            drug_name_b=name_b,
            n_patients_a=row["n_patients_a"],
            n_patients_b=row["n_patients_b"],
        )
        registry.add_or_update(pair)

    return registry