# src/cohorts/extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import duckdb
import polars as pl

from src.catalog.catalog import DrugCatalog, combo_drug_id
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

    # (ing_a, ing_b) -> lignes de co-initiation, annotées par la molécule source
    self.de_facto_pairs: dict[tuple[int, int], list[dict]] = {}
    # (ing_a, ing_b) -> stanford_index de la passe comprimé combiné
    self.fixed_pairs: dict[tuple[int, int], pl.DataFrame] = {}
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
    """Comparateurs du wash-out : classe(s) ATC4 de chaque ingrédient de la cible."""
    return self.catalog.get_comparator_ids(target_id)

  def _build_sql(self, target_id: int, kind: str) -> str:
    """Requête TTE commune aux monothérapies et aux bi-thérapies fixes.

    Tables enregistrées au préalable :
      - target_ingredients   : ingrédients de la cible (1 pour mono, 2 pour fixe)
      - comparator_ingredients : comparateurs ATC4 (wash-out de classe)
      - target_codes         : codes prescrits de la bi-thérapie fixe (kind fixe)
    """
    p = self.protocol
    is_fixed = kind == "fixed_combination"

    if is_fixed:
      # t0 = première dispensation du comprimé combiné
      target_filter = (
          "drug_concept_id IN (SELECT drug_concept_id FROM target_codes)"
      )
      # Seuls les 2 ingrédients du comprimé sont nouveaux à t0
      final_filter = "c.n_new_drugs = 2 AND c.n_new_targets = 2"
    else:
      # t0 = première exposition à la molécule cible sous forme MONOTHÉRAPIE
      target_filter = f"ingredient_id = {target_id} AND is_monotherapy = TRUE"
      # 1 seule NOUVELLE molécule initiée à t0 (la cible) sous forme mono
      # (les renouvellements d'ordonnances chroniques pré-existantes n'excluent pas le patient)
      final_filter = "c.n_new_drugs = 1 AND c.all_monotherapies = TRUE"

    # Bi-thérapies de facto : collectées uniquement depuis les passes monothérapie
    de_facto_select = "" if is_fixed else f"""
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
        WHERE ingredient_id != {target_id}"""

    return f"""
        WITH 
        -- 1. Date candidate t0
        target_exposures AS (
            SELECT DISTINCT person_id, exp_date
            FROM drug_exposure
            WHERE {target_filter}
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

        -- Wash-out strict (365j) : exclusion si prise antérieure d'un ingrédient cible
        -- (sous toute forme) OU d'un comparateur ATC4
        prior_washout_violations AS (
            SELECT DISTINCT b.person_id
            FROM baseline_exposures b
            WHERE b.ingredient_id IN (SELECT ingredient_id FROM target_ingredients)
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
                COUNT(DISTINCT ingredient_id) FILTER (
                    WHERE ingredient_id IN (SELECT ingredient_id FROM target_ingredients)
                )::INTEGER AS n_new_targets,
                BOOL_AND(is_monotherapy) AS all_monotherapies
            FROM new_initiations_at_t0
            GROUP BY person_id
        ),

        step_3_final AS (
            SELECT s.*
            FROM step_2_washout_valid s
            JOIN patient_new_counts c 
              ON s.person_id = c.person_id
            WHERE {final_filter}
        ),

        -- Bi-thérapie de facto : exactement 2 NOUVELLES molécules mono initiées le même jour
        co_initiations_de_facto AS (
            SELECT n.*
            FROM new_initiations_at_t0 n
            JOIN patient_new_counts c 
              ON n.person_id = c.person_id
            WHERE c.n_new_drugs = 2 
              AND c.all_monotherapies = TRUE
        )

        -- RECORD 0 : Cohorte cible
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
        FROM step_3_final

        -- RECORD 1 : Comptes d'attrition
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM candidate_t0
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_1_obs_valid
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_2_washout_valid
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT FROM step_3_final
        {de_facto_select};
        """

  def extract_cohort(
      self,
      target_id: int,
      comparator_ids: list[int] | None = None,
  ) -> DrugCohort | None:
    """Extrait la cohorte d'une monothérapie ou d'une bi-thérapie fixe du catalogue.

    La cohorte fixe n'est pas une cohorte finale : elle est conservée en
    mémoire et fusionnée par export_combination_cohorts().
    """
    item = self.catalog.get(target_id)
    if item is None:
      raise KeyError(f"Molécule {target_id} introuvable dans le DrugCatalog.")
    if item.kind not in ("monotherapy", "fixed_combination"):
      raise ValueError(
          f"{item.name} ({item.kind}) : les de facto sont exportées via"
          " export_combination_cohorts()."
      )

    # Résolution automatique des comparateurs ATC4 si non spécifiés
    if comparator_ids is None:
      comparator_ids = self.get_atc4_comparators(target_id)

    ingredient_ids = list(item.ingredient_concept_ids) or [target_id]
    self.con.register(
        "target_ingredients",
        pl.DataFrame({"ingredient_id": ingredient_ids}, schema={"ingredient_id": pl.Int64}),
    )
    self.con.register(
        "comparator_ingredients",
        pl.DataFrame({"ingredient_id": comparator_ids}, schema={"ingredient_id": pl.Int64}),
    )
    self.con.register(
        "target_codes",
        pl.DataFrame(
            {"drug_concept_id": sorted(item.descendant_concept_ids)},
            schema={"drug_concept_id": pl.Int64},
        ),
    )

    sql = self._build_sql(target_id, item.kind)
    res_df = self.con.sql(sql).pl()

    # Enregistrement des co-initiations de facto, annotées par la passe source
    co_init_df = res_df.filter(pl.col("record_type") == 2)
    for row in co_init_df.iter_rows(named=True):
      other_id = int(row["other_ingredient_id"])
      pair_key = (min(target_id, other_id), max(target_id, other_id))
      row["source_id"] = target_id
      self.de_facto_pairs.setdefault(pair_key, []).append(row)

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

    if item.kind == "fixed_combination" and len(stanford_index) > 0:
      pair_key = tuple(sorted(item.ingredient_concept_ids))
      self.fixed_pairs[pair_key] = stanford_index

    if len(stanford_index) == 0:
      return None

    return DrugCohort.from_drug_item(
        drug_item=item, stanford_index=stanford_index
    )

  def export_combination_cohorts(
      self, min_size: int | None = None
  ) -> list[DrugCohort]:
    """Construit une cohorte unique (kind combination) par couple d'ingrédients.

    Fusionne deux sources, tracées par patient dans la colonne combo_source :
      - fixed    : patients de la passe comprimé combiné (extract_cohort) ;
      - de_facto : co-initiations captées par les passes des DEUX ingrédients
                   au même t0 (wash-out des classes ATC4 de A ET de B). Les deux
                   monothérapies doivent avoir été extraites dans la même exécution.
    Un patient présent dans les deux sources garde le t0 le plus précoce.
    Le seuil min_size ne s'applique qu'aux couples sans comprimé combiné
    (les cohortes fixes sont conservées dès N >= 1, comme les monothérapies).
    """
    threshold = (
        min_size if min_size is not None else self.protocol.min_de_facto_size
    )
    index_cols = [
        "person_id",
        "t0",
        "t_6m",
        "t_12m",
        "history_days_prior",
        "follow_up_days",
        "has_12m_followup",
    ]

    valid_cohorts = []
    for ing_a, ing_b in sorted(set(self.de_facto_pairs) | set(self.fixed_pairs)):
      parts = []

      pts = self.de_facto_pairs.get((ing_a, ing_b), [])
      if pts:
        sources: dict[tuple[int, object], set[int]] = {}
        rows: dict[tuple[int, object], dict] = {}
        for r in pts:
          key = (r["person_id"], r["t0"])
          sources.setdefault(key, set()).add(r["source_id"])
          rows.setdefault(key, r)
        valid_keys = [k for k, s in sources.items() if {ing_a, ing_b} <= s]
        if valid_keys:
          parts.append(
              pl.DataFrame([rows[k] for k in valid_keys])
              .select(index_cols)
              .with_columns(pl.lit("de_facto").alias("combo_source"))
          )

      fixed_df = self.fixed_pairs.get((ing_a, ing_b))
      if fixed_df is not None:
        parts.append(
            fixed_df.select(index_cols).with_columns(
                pl.lit("fixed").alias("combo_source")
            )
        )

      if not parts:
        continue
      df_pts = (
          pl.concat(parts, how="vertical_relaxed")
          .sort(["person_id", "t0", "combo_source"], descending=[False, False, True])
          .unique(subset=["person_id"], keep="first", maintain_order=True)
      )
      if fixed_df is None and len(df_pts) < threshold:
        continue

      item_a = self.catalog.get(ing_a)
      item_b = self.catalog.get(ing_b)
      if item_a is None or item_b is None:
        continue

      combo_item = self.catalog.get_combination("combination", ing_a, ing_b)
      if combo_item is None:
        fixed_item = self.catalog.get_combination(
            "fixed_combination", ing_a, ing_b
        )
        combo_item = self.catalog.build_combination_item(
            "combination", item_a, item_b,
            descendant_concept_ids=(
                set(fixed_item.descendant_concept_ids) if fixed_item else None
            ),
        )
      assert combo_item.drug_id == combo_drug_id("combination", ing_a, ing_b)

      valid_cohorts.append(
          DrugCohort.from_drug_item(drug_item=combo_item, stanford_index=df_pts)
      )

    return valid_cohorts
