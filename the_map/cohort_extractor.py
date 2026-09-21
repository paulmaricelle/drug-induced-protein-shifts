from __future__ import annotations

from typing import Optional
import duckdb
import polars as pl

from the_map.config import PathConfig, ProtocolConfig
from the_map.drugCatalog import DrugCatalog
from the_map.drugCohort import DrugCohort


class CohortExtractor:
    """Moteur d'extraction opérant strictement par concept_id entiers."""

    def __init__(
        self,
        catalog: DrugCatalog,
        paths: PathConfig,
        protocol: ProtocolConfig,
        con: Optional[duckdb.DuckDBPyConnection] = None,
    ):
        self.catalog = catalog
        self.paths = paths
        self.protocol = protocol
        self.con = con or duckdb.connect()

        self.con.execute("PRAGMA threads=16;")
        self.con.execute("PRAGMA preserve_insertion_order=false;")

    def _build_sql(self, target_id: int) -> str:
        p = self.protocol
        paths = self.paths

        return f"""
        WITH 
        -- 1. Codes cliniques cibles : STRICTEMENT MONOTHÉRAPIE
        target_prescriptions AS (
            SELECT drug_concept_id::BIGINT AS drug_concept_id
            FROM read_parquet('{paths.mapping_path}')
            WHERE ingredient_id = {target_id}
              AND is_monotherapy = TRUE
        ),

        -- 2. Codes de la famille ATC4 : TOUTES FORMULATIONS (Washout de classe)
        class_prescriptions AS (
            SELECT m.drug_concept_id::BIGINT AS drug_concept_id
            FROM read_parquet('{paths.mapping_path}') m
            JOIN same_atc4_ingredients sci ON m.ingredient_id = sci.ingredient_id
        ),

        -- 3. Détection des expositions cibles pour les candidats
        target_exposures AS (
            SELECT 
                de.person_id::BIGINT AS person_id,
                de.exp_date,
                de.end_date
            FROM read_parquet('{paths.drug_exposure_parquet}') de
            JOIN target_prescriptions tp ON de.drug_concept_id = tp.drug_concept_id
        ),

        candidate_t0 AS (
            SELECT 
                person_id,
                MIN(exp_date) AS t0,
                (MIN(exp_date) + INTERVAL '{p.obs_post_days} days')::DATE AS t_endpoint,
                COUNT(*)::INTEGER AS total_target_prescriptions,
                MAX(end_date) AS max_target_end_date
            FROM target_exposures
            GROUP BY person_id
        ),

        -- 4. Étape 1 : Fenêtres d'observation requises dans STARR
        step_1_obs_valid AS (
            SELECT 
                c.*,
                (c.t0 - po.obs_start)::INTEGER AS history_days_prior
            FROM candidate_t0 c
            JOIN read_parquet('{paths.observation_period_parquet}') po ON c.person_id = po.person_id
            WHERE (c.t0 - po.obs_start) >= {p.obs_pre_days}
              AND (po.obs_end - c.t0) >= {p.obs_post_days}
        ),

        -- 5. Étape 2 : Washout ATC4 strict dans [t0 - 365j, t0[
        prior_atc4_exposures AS (
            SELECT DISTINCT de.person_id::BIGINT AS person_id
            FROM read_parquet('{paths.drug_exposure_parquet}') de
            JOIN class_prescriptions cp ON de.drug_concept_id = cp.drug_concept_id
            JOIN step_1_obs_valid s ON de.person_id = s.person_id
            WHERE de.exp_date >= (s.t0 - INTERVAL '{p.washout_days} days')
              AND de.exp_date < s.t0
        ),

        step_2_atc4_washout_valid AS (
            SELECT *
            FROM step_1_obs_valid
            WHERE person_id NOT IN (SELECT person_id FROM prior_atc4_exposures)
        ),

        -- 6. Étape 3 : Co-médications incidentes (toutes formulations : mono + poly)
        other_exposures AS (
            SELECT 
                de.person_id::BIGINT AS person_id,
                m.ingredient_id::BIGINT AS ingredient_id,
                de.exp_date
            FROM read_parquet('{paths.drug_exposure_parquet}') de
            JOIN read_parquet('{paths.mapping_path}') m ON de.drug_concept_id = m.drug_concept_id
            WHERE m.ingredient_id NOT IN (SELECT ingredient_id FROM same_atc4_ingredients)
              AND de.person_id IN (SELECT person_id FROM step_2_atc4_washout_valid)
        ),

        comed_profile AS (
            SELECT 
                oe.person_id,
                oe.ingredient_id,
                MAX(CASE WHEN oe.exp_date < (s.t0 - INTERVAL '{p.stability_pre_days} days') THEN oe.exp_date ELSE NULL END) AS last_baseline_date,
                COUNT(CASE WHEN oe.exp_date >= (s.t0 - INTERVAL '{p.stability_pre_days} days') 
                            AND oe.exp_date <= (s.t0 + INTERVAL '{p.stability_post_days} days') THEN 1 ELSE NULL END)::INTEGER AS n_in_stability_window
            FROM other_exposures oe
            JOIN step_2_atc4_washout_valid s ON oe.person_id = s.person_id
            GROUP BY oe.person_id, oe.ingredient_id
        ),

        violations_incident_drugs AS (
            SELECT DISTINCT person_id
            FROM comed_profile
            WHERE n_in_stability_window > 0
              AND (
                last_baseline_date IS NULL
                OR 
                last_baseline_date < (
                    (SELECT t0 FROM step_2_atc4_washout_valid WHERE step_2_atc4_washout_valid.person_id = comed_profile.person_id) 
                    - INTERVAL '{p.stability_pre_days + p.max_chronic_gap_days} days'
                )
              )
        ),

        step_3_comed_stable AS (
            SELECT *
            FROM step_2_atc4_washout_valid
            WHERE person_id NOT IN (SELECT person_id FROM violations_incident_drugs)
        ),

        -- 7. Étape 4 : Adhérence confirmée
        step_4_adherent AS (
            SELECT 
                person_id,
                t0,
                t_endpoint,
                total_target_prescriptions,
                history_days_prior
            FROM step_3_comed_stable
            WHERE total_target_prescriptions >= 2
               OR max_target_end_date >= (t0 + INTERVAL '{p.min_treatment_coverage_days} days')
        )

        -- 0 = PATIENTS FINAUX, 1 = COMPTES D'ATTRITION
        SELECT 0::TINYINT AS record_type, person_id, t0, t_endpoint, total_target_prescriptions, history_days_prior, NULL::BIGINT AS n_patients
        FROM step_4_adherent
        UNION ALL
        SELECT 1::TINYINT, NULL, NULL, NULL, NULL, NULL, COUNT(*)::BIGINT FROM candidate_t0
        UNION ALL
        SELECT 1::TINYINT, NULL, NULL, NULL, NULL, NULL, COUNT(*)::BIGINT FROM step_1_obs_valid
        UNION ALL
        SELECT 1::TINYINT, NULL, NULL, NULL, NULL, NULL, COUNT(*)::BIGINT FROM step_2_atc4_washout_valid
        UNION ALL
        SELECT 1::TINYINT, NULL, NULL, NULL, NULL, NULL, COUNT(*)::BIGINT FROM step_3_comed_stable
        UNION ALL
        SELECT 1::TINYINT, NULL, NULL, NULL, NULL, NULL, COUNT(*)::BIGINT FROM step_4_adherent;
        """

    def extract_cohort(self, target_id: int) -> DrugCohort:
        if not isinstance(target_id, int):
            raise TypeError(f"target_id doit être un entier OMOP concept_id (reçu: {type(target_id)} = {target_id}).")

        item = self.catalog.get(target_id)

        # Ingrédients ATC4 frères (strictement des entiers)
        family_ids = list(self.catalog.get_atc4_family_ids(target_id))
        self.con.register("same_atc4_ingredients", pl.DataFrame({"ingredient_id": family_ids}, schema={"ingredient_id": pl.Int64}))

        sql = self._build_sql(target_id)
        res_df = self.con.sql(sql).pl()

        patients = res_df.filter(pl.col("record_type") == 0).select([
            "person_id", "t0", "t_endpoint", "total_target_prescriptions", "history_days_prior"
        ])

        attr_counts = res_df.filter(pl.col("record_type") == 1)["n_patients"].to_list()

        steps = [
            "1. Initiateurs bruts",
            f"2. Observation STARR ({self.protocol.obs_pre_days}j pre / {self.protocol.obs_post_days}j post)",
            f"3. Washout ATC4 strict ({self.protocol.washout_days}j)",
            f"4. Co-médications stables ([-{self.protocol.stability_pre_days}j, +{self.protocol.stability_post_days}j])",
            f"5. Adhérence confirmée (>=2 prises OU couvr. >={self.protocol.min_treatment_coverage_days}j)"
        ]

        attrition_df = pl.DataFrame({
            "etape": steps,
            "step_index": list(range(1, len(steps) + 1)),
            "n_patients": attr_counts
        }).with_columns([
            (pl.col("n_patients") / attr_counts[0] * 100).round(2).alias("pct_retenu_initial"),
            (pl.col("n_patients") / pl.col("n_patients").shift(1) * 100).round(2).alias("pct_passage_etape")
        ])

        return DrugCohort(
            target_item=item,
            target_patients=patients,
            attrition_summary=attrition_df
        )