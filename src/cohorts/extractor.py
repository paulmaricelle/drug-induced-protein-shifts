# src/cohorts/extractor.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
import duckdb
import polars as pl

from src.catalog.catalog import DrugCatalog
from src.cohorts.cohort import DrugCohort
from src.config import PathConfig, ProtocolConfig


class CohortExtractor:
  """Moteur d'extraction haute performance en RAM (DuckDB) pour Target Trial Emulation."""

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
    self.con = con or duckdb.connect(":memory:")

    self.con.execute("PRAGMA threads=16;")
    self.con.execute("PRAGMA max_memory='48GB';")
    self.con.execute("PRAGMA preserve_insertion_order=false;")

    self.de_facto_pairs: dict[tuple[int, int], list[dict]] = {}
    self._init_duckdb_cache()

  def _init_duckdb_cache(self) -> None:
    paths = self.paths
    if not paths.observation_period_parquet.exists():
      raise FileNotFoundError(
          f"Cache introuvable dans {paths.cache_dir}.\n"
          f"Exécutez : python3 scripts/build_cache.py --ratio 1.0"
      )

    # Détection d'un cache obsolète (généré par une ancienne version de build_cache.py)
    required_cols = {
        "person_id", "drug_concept_id", "exp_date", "ingredient_id",
        "is_monotherapy",
    }
    found_cols = {
        row[0]
        for row in self.con.execute(
            "SELECT name FROM parquet_schema(?)",
            [str(paths.drug_exposure_parquet)],
        ).fetchall()
    }
    missing = required_cols - found_cols
    if missing:
      ratio = "0.2" if paths.is_sample else "1.0"
      raise RuntimeError(
          f"Cache obsolète : colonnes {sorted(missing)} absentes de"
          f" {paths.drug_exposure_parquet}.\n"
          f"Régénérez-le : python3 scripts/build_cache.py --ratio {ratio}"
      )

    print(
        f"Chargement du cache OMOP ({paths.cache_dir.name}) en mémoire"
        " DuckDB..."
    )
    self.con.execute(f"""
            CREATE OR REPLACE TABLE observation_period AS 
            SELECT person_id, obs_start, obs_end
            FROM read_parquet('{paths.observation_period_parquet}');
        """)

    self.con.execute(f"""
            CREATE OR REPLACE TABLE drug_exposure AS 
            SELECT person_id, drug_concept_id, exp_date, ingredient_id, is_monotherapy
            FROM read_parquet('{paths.drug_exposure_parquet}');
        """)

    self.con.execute("CREATE INDEX idx_exp_ing ON drug_exposure(ingredient_id);")
    self.con.execute("CREATE INDEX idx_exp_person ON drug_exposure(person_id);")
    print("-> Cache DuckDB prêt.")

  def get_atc4_comparators(self, target_id: int) -> list[int]:
    """Résout la liste des molécules du catalogue partageant la même classe ATC4 (5 caractères)."""
    if hasattr(self.catalog, "get_atc4_family_ids"):
      res = self.catalog.get_atc4_family_ids(target_id)
      if res:
        return [int(x) for x in res if int(x) != target_id]

    target_item = self.catalog.get(target_id)
    if target_item is None:
      return []

    target_atc = (
        getattr(target_item, "atc4", None)
        or getattr(target_item, "atc", None)
        or getattr(target_item, "atc3", None)
    )
    if not target_atc:
      return []

    target_prefix = str(target_atc)[:5].upper()
    comparators = []
    for item in self.catalog:
      if item.drug_id == target_id:
        continue
      item_atc = (
          getattr(item, "atc4", None)
          or getattr(item, "atc", None)
          or getattr(item, "atc3", None)
      )
      if item_atc and str(item_atc)[:5].upper() == target_prefix:
        comparators.append(int(item.drug_id))

    return comparators

  def _build_sql(self, target_id: int) -> str:
    p = self.protocol

    return f"""
        WITH 
        -- 1. Date candidate t0 = Première exposition à la molécule cible sous forme MONOTHÉRAPIE
        target_exposures AS (
            SELECT person_id, exp_date
            FROM drug_exposure
            WHERE ingredient_id = {target_id}
              AND is_monotherapy = TRUE
        ),
        candidate_t0 AS (
            SELECT 
                person_id,
                MIN(exp_date) AS t0,
                (MIN(exp_date) + INTERVAL '{p.obs_post_days} days')::DATE AS t_6m,
                (MIN(exp_date) + INTERVAL '{p.followup_12m_days} days')::DATE AS t_12m
            FROM target_exposures
            GROUP BY person_id
        ),

        -- 2. Critères d'observation : >= 365j avant t0 et >= 182j après t0
        step_1_obs_valid AS (
            SELECT 
                c.person_id,
                c.t0,
                c.t_6m,
                c.t_12m,
                (c.t0 - po.obs_start)::INTEGER AS history_days_prior,
                (po.obs_end - c.t0)::INTEGER AS follow_up_days,
                ((po.obs_end - c.t0) >= {p.followup_12m_days})::BOOLEAN AS has_12m_followup
            FROM candidate_t0 c
            JOIN observation_period po 
              ON c.person_id = po.person_id
             AND c.t0 >= po.obs_start 
             AND c.t0 <= po.obs_end
            WHERE (c.t0 - po.obs_start) >= {p.obs_pre_days}
              AND (po.obs_end - c.t0) >= {p.obs_post_days}
        ),

        -- 3. Historique d'exposition durant l'année de baseline [t0 - 365j, t0[
        baseline_exposures AS (
            SELECT DISTINCT de.person_id, de.ingredient_id
            FROM drug_exposure de
            JOIN step_1_obs_valid s ON de.person_id = s.person_id
            WHERE de.exp_date >= (s.t0 - INTERVAL '{p.washout_days} days')
              AND de.exp_date < s.t0
        ),

        -- Wash-out strict (365j) : exclusion si prise antérieure de la cible OU d'un comparateur ATC4
        prior_washout_violations AS (
            SELECT DISTINCT b.person_id
            FROM baseline_exposures b
            WHERE b.ingredient_id = {target_id}
               OR b.ingredient_id IN (SELECT ingredient_id FROM comparator_ingredients)
        ),
        step_2_washout_valid AS (
            SELECT s.*
            FROM step_1_obs_valid s
            WHERE s.person_id NOT IN (SELECT person_id FROM prior_washout_violations)
        ),

        -- 4. Prescriptions délivrées le jour t0
        prescriptions_at_t0 AS (
            SELECT 
                de.person_id,
                de.ingredient_id,
                de.is_monotherapy,
                s.t0,
                s.t_6m,
                s.t_12m,
                s.history_days_prior,
                s.follow_up_days,
                s.has_12m_followup
            FROM drug_exposure de
            JOIN step_2_washout_valid s 
              ON de.person_id = s.person_id 
             AND de.exp_date = s.t0
        ),

        -- Distinction clinique fondamentale :
        -- Une molécule à t0 est NOUVELLE si elle n'a jamais été vue en baseline [t0 - 365j, t0[
        new_initiations_at_t0 AS (
            SELECT p.*
            FROM prescriptions_at_t0 p
            LEFT JOIN baseline_exposures b 
                   ON p.person_id = b.person_id 
                  AND p.ingredient_id = b.ingredient_id
            WHERE b.ingredient_id IS NULL
        ),

        -- Décompte des NOUVEAUX principes actifs initiés à t0
        patient_new_counts AS (
            SELECT 
                person_id,
                COUNT(DISTINCT ingredient_id)::INTEGER AS n_new_drugs,
                BOOL_AND(is_monotherapy) AS all_monotherapies
            FROM new_initiations_at_t0
            GROUP BY person_id
        ),

        -- Monothérapie retenue : 1 seule NOUVELLE molécule initiée à t0 (la cible) sous forme mono
        -- (les renouvellements d'ordonnances chroniques pré-existantes n'excluent plus le patient)
        step_3_pure_mono AS (
            SELECT s.*
            FROM step_2_washout_valid s
            JOIN patient_new_counts c 
              ON s.person_id = c.person_id
            WHERE c.n_new_drugs = 1 
              AND c.all_monotherapies = TRUE
        ),

        -- Bi-thérapie de facto : exactement 2 NOUVELLES molécules initiées le même jour
        co_initiations_de_facto AS (
            SELECT n.*
            FROM new_initiations_at_t0 n
            JOIN patient_new_counts c 
              ON n.person_id = c.person_id
            WHERE c.n_new_drugs = 2 
              AND c.all_monotherapies = TRUE
        )

        -- RECORD 0 : Cohorte Monothérapie
        SELECT 
            0::TINYINT AS record_type,
            person_id,
            t0,
            t_6m,
            t_12m,
            history_days_prior,
            follow_up_days,
            has_12m_followup,
            NULL::BIGINT AS other_ingredient_id,
            NULL::BIGINT AS n_patients
        FROM step_3_pure_mono

        -- RECORD 1 : Comptes d'attrition
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM candidate_t0
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_1_obs_valid
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_2_washout_valid
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_3_pure_mono

        -- RECORD 2 : Paires de facto
        UNION ALL
        SELECT 
            2::TINYINT,
            person_id,
            t0,
            t_6m,
            t_12m,
            history_days_prior,
            follow_up_days,
            has_12m_followup,
            ingredient_id AS other_ingredient_id,
            NULL::BIGINT
        FROM co_initiations_de_facto
        WHERE ingredient_id != {target_id};
        """

  def extract_cohort(
      self,
      target_id: int,
      comparator_ids: list[int] | None = None,
  ) -> DrugCohort | None:
    item = self.catalog.get(target_id)
    if item is None:
      raise KeyError(f"Molécule {target_id} introuvable dans le DrugCatalog.")

    # Résolution automatique des comparateurs ATC4 si non spécifiés
    if comparator_ids is None:
      comparator_ids = self.get_atc4_comparators(target_id)

    self.con.register(
        "comparator_ingredients",
        pl.DataFrame(
            {"ingredient_id": comparator_ids},
            schema={"ingredient_id": pl.Int64},
        ),
    )

    sql = self._build_sql(target_id)
    res_df = self.con.sql(sql).pl()

    # Enregistrement des paires de facto
    co_init_df = res_df.filter(pl.col("record_type") == 2)
    if len(co_init_df) > 0:
      for row in co_init_df.iter_rows(named=True):
        other_id = int(row["other_ingredient_id"])
        pair_key = (min(target_id, other_id), max(target_id, other_id))
        self.de_facto_pairs.setdefault(pair_key, []).append(row)

    # Monothérapie
    stanford_index = (
        res_df.filter(pl.col("record_type") == 0)
        .select([
            "person_id",
            "t0",
            "t_6m",
            "t_12m",
            "history_days_prior",
            "follow_up_days",
            "has_12m_followup",
        ])
        .sort(["person_id", "t0"])
    )

    if len(stanford_index) == 0:
      return None

    return DrugCohort.from_drug_item(
        drug_item=item, stanford_index=stanford_index
    )

  def export_valid_de_facto_cohorts(
      self, min_size: int | None = None
  ) -> list[DrugCohort]:
    valid_cohorts = []
    threshold = (
        min_size if min_size is not None else self.protocol.min_de_facto_size
    )

    for (ing_a, ing_b), pts in self.de_facto_pairs.items():
      unique_pts = {p["person_id"]: p for p in pts}
      if len(unique_pts) >= threshold:
        item_a = self.catalog.get(ing_a)
        item_b = self.catalog.get(ing_b)
        name_a = item_a.name if item_a else str(ing_a)
        name_b = item_b.name if item_b else str(ing_b)

        df_pts = (
            pl.DataFrame(list(unique_pts.values()))
            .select([
                "person_id",
                "t0",
                "t_6m",
                "t_12m",
                "history_days_prior",
                "follow_up_days",
                "has_12m_followup",
            ])
            .sort(["person_id", "t0"])
        )

        # ID 64-bit déterministe et sans collision
        hash_digest = hashlib.md5(f"{ing_a}_{ing_b}".encode()).hexdigest()
        combo_id = int(hash_digest[:12], 16) % (10**12) + 9_000_000_000_000

        combo_cohort = DrugCohort(
            drug_id=combo_id,
            name=f"{name_a} + {name_b} (De Facto)",
            kind="de_facto_combination",
            ingredient_concept_ids=[ing_a, ing_b],
            stanford_index=df_pts,
        )
        valid_cohorts.append(combo_cohort)

    return valid_cohorts