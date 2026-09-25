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
        "is_monotherapy", "exp_end_date", "drug_type_concept_id",
        "route_concept_id", "refills",
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
            SELECT person_id, drug_concept_id, exp_date, ingredient_id, is_monotherapy,
                   exp_end_date, drug_type_concept_id, route_concept_id, refills
            FROM read_parquet('{paths.drug_exposure_parquet}');
        """)

    self.con.execute("CREATE INDEX idx_exp_ing ON drug_exposure(ingredient_id);")
    self.con.execute("CREATE INDEX idx_exp_person ON drug_exposure(person_id);")
    self._register_ingredient_flags()
    print("-> Cache DuckDB prêt.")

  def _register_ingredient_flags(self) -> None:
    """Table ing_flags : périmètre du compteur et agents procéduraux, par ingrédient.

    in_scope : ingrédient du catalogue, ou ATC4 hors des groupes exclus
    (protocol.excluded_atc_prefixes). Les ingrédients sans ATC hors catalogue
    (excipients, solutés, extraits) sont hors périmètre.
    """
    p = self.protocol
    atc = {}
    names = {}
    if self.paths.atc4_path.exists():
      for row in pl.read_parquet(self.paths.atc4_path).iter_rows(named=True):
        atc[int(row["ingredient_id"])] = str(row["atc4_code"])
        names[int(row["ingredient_id"])] = str(row["ingredient_name"]).lower()
    catalog_ids = {i.drug_id for i in self.catalog if i.kind == "monotherapy"}
    for i in self.catalog:
      if i.kind == "monotherapy" and i.atc4:
        atc[i.drug_id] = i.atc4
        names.setdefault(i.drug_id, i.name.lower())

    records = []
    for ing_id in set(atc) | catalog_ids:
      code = atc.get(ing_id, "")
      in_scope = ing_id in catalog_ids or (
          bool(code) and not code.startswith(p.excluded_atc_prefixes)
      )
      procedural = bool(code) and code.startswith(p.procedural_atc_prefixes)
      procedural |= names.get(ing_id, "") in p.procedural_ingredient_names
      records.append({
          "ingredient_id": ing_id, "in_scope": in_scope, "procedural": procedural,
      })
    self.con.register(
        "ing_flags",
        pl.DataFrame(records, schema={
            "ingredient_id": pl.Int64, "in_scope": pl.Boolean, "procedural": pl.Boolean,
        }),
    )

  def get_atc4_comparators(self, target_id: int) -> list[int]:
    """Comparateurs du wash-out : classe(s) ATC4 de chaque ingrédient de la cible."""
    return self.catalog.get_comparator_ids(target_id)

  # Type d'enregistrement OMOP "EHR order" : seule source d'une durée prescrite fiable
  _EHR_ORDER_TYPE = 32833

  def _counted_expr(self) -> str:
    """Condition SQL : une nouvelle initiation non cible incrémente-t-elle le compteur ?"""
    p = self.protocol
    parts = []
    if p.counter_scope == "systemic":
      parts.append("in_scope")
    elif p.counter_scope != "all":
      raise ValueError(f"counter_scope inconnu : {p.counter_scope}")
    if p.short_order_max_days > 0:
      parts.append("NOT is_short")
    if p.ignore_local_routes:
      parts.append("NOT is_local")
    if p.ignore_procedural:
      parts.append("NOT procedural")
    return " AND ".join(parts) or "TRUE"

  def _build_sql(self, target_id: int, kind: str) -> str:
    """Requête TTE commune aux monothérapies et aux bi-thérapies fixes.

    t0 = première date d'exposition à la cible qui satisfait TOUS les critères :
    fenêtres d'observation, wash-out (aucune exposition à la cible ni à ses
    comparateurs ATC4 dans les 365 j précédents) et comptage des nouvelles
    initiations du jour (cf. _counted_expr).

    Tables enregistrées au préalable :
      - target_ingredients   : ingrédients de la cible (1 pour mono, 2 pour fixe)
      - comparator_ingredients : comparateurs ATC4 (wash-out de classe)
      - target_codes         : codes prescrits de la bi-thérapie fixe (kind fixe)
      - ing_flags            : périmètre du compteur (cf. _register_ingredient_flags)
    """
    p = self.protocol
    is_fixed = kind == "fixed_combination"
    local_routes = ", ".join(str(r) for r in p.local_route_concept_ids) or "NULL"
    short_days = max(p.short_order_max_days, 0)

    ext = p.persistence_extension
    pers_lo, pers_hi = p.persistence_window_days

    if is_fixed:
      # Dispensations du comprimé combiné
      target_filter = (
          "drug_concept_id IN (SELECT drug_concept_id FROM target_codes)"
      )
      # Seuls les 2 ingrédients du comprimé sont de nouvelles initiations comptées
      final_tpl = "n_new{s} = 2 AND n_new_targets{s} = 2"
    else:
      # Expositions à la molécule cible sous forme MONOTHÉRAPIE
      target_filter = f"ingredient_id = {target_id} AND is_monotherapy = TRUE"
      # 1 seule nouvelle initiation comptée (la cible), sous forme mono
      final_tpl = "n_new{s} = 1 AND all_mono{s}"
    final_filter = final_tpl.format(s="")
    final_filter_x = final_tpl.format(s="_x")

    # Persistance d'une co-initiation : réexposition au même ingrédient dans la fenêtre
    pers_cte = f"""
        persistent AS (
            SELECT DISTINCT n.person_id, n.d, n.ingredient_id
            FROM new_ingredients n
            JOIN drug_exposure de
              ON de.person_id = n.person_id AND de.ingredient_id = n.ingredient_id
             AND de.exp_date >= (n.d + INTERVAL '{pers_lo} days')
             AND de.exp_date <= (n.d + INTERVAL '{pers_hi} days')
        ),""" if ext else ""
    pers_expr = "(p.ingredient_id IS NOT NULL)" if ext else "FALSE"
    pers_join = """
                LEFT JOIN persistent p
                  ON p.person_id = n.person_id AND p.d = n.d
                 AND p.ingredient_id = n.ingredient_id""" if ext else ""

    # Bi-thérapies de facto : collectées uniquement depuis les passes monothérapie
    de_facto_select = "" if is_fixed else f"""
        -- RECORD 2 : Co-initiations de facto (toutes dates éligibles, la fusion
        -- retient la plus précoce commune aux deux passes, règle principale d'abord)
        UNION ALL
        SELECT
            2::TINYINT,
            c.person_id,
            c.d,
            (c.d + INTERVAL '{p.obs_post_days} days')::DATE,
            (c.d + INTERVAL '{p.followup_12m_days} days')::DATE,
            (c.d - w.obs_start)::INTEGER,
            (w.obs_end - c.d)::INTEGER,
            ((w.obs_end - c.d) >= {p.followup_12m_days})::BOOLEAN,
            c.ingredient_id AS other_ingredient_id,
            NULL::BIGINT,
            CASE WHEN pd.n_new = 2 AND pd.all_mono AND c.counted
                 THEN 't0_info' ELSE 'persistence' END
        FROM counted c
        JOIN per_d pd ON pd.person_id = c.person_id AND pd.d = c.d
        JOIN wo_ok w ON w.person_id = c.person_id AND w.d = c.d
        WHERE NOT c.is_target
          AND ((pd.n_new = 2 AND pd.all_mono AND c.counted)
               OR (pd.n_new_x = 2 AND pd.all_mono_x AND c.counted_x))"""

    return f"""
        WITH
        -- 1. Dates candidates : expositions à la cible
        target_exposures AS (
            SELECT DISTINCT person_id, exp_date AS d
            FROM drug_exposure
            WHERE {target_filter}
        ),
        -- 2. Fenêtres d'observation : >= 365 j avant et >= 182 j après la date
        obs_ok AS (
            SELECT t.person_id, t.d, po.obs_start, po.obs_end
            FROM target_exposures t
            JOIN observation_period po
              ON t.person_id = po.person_id
             AND t.d >= po.obs_start AND t.d <= po.obs_end
            WHERE (t.d - po.obs_start) >= {p.obs_pre_days}
              AND (po.obs_end - t.d) >= {p.obs_post_days}
        ),
        -- 3. Wash-out : aucune exposition à un ingrédient cible (toute forme)
        -- ni à un comparateur ATC4 dans les {p.washout_days} j précédant la date
        washout_set AS (
            SELECT DISTINCT person_id, exp_date
            FROM drug_exposure
            WHERE ingredient_id IN (SELECT ingredient_id FROM target_ingredients)
               OR ingredient_id IN (SELECT ingredient_id FROM comparator_ingredients)
        ),
        lagged AS (
            SELECT person_id, exp_date,
                   LAG(exp_date) OVER (PARTITION BY person_id ORDER BY exp_date) AS prev
            FROM washout_set
        ),
        wo_ok AS (
            SELECT o.*
            FROM obs_ok o
            JOIN lagged l ON l.person_id = o.person_id AND l.exp_date = o.d
            WHERE l.prev IS NULL OR (o.d - l.prev) >= {p.washout_days}
        ),

        -- 4. Nouvelles initiations du jour : ingrédient absent de [d - 365 j, d[
        at_d AS (
            SELECT w.person_id, w.d, de.ingredient_id, de.is_monotherapy,
                   de.drug_type_concept_id, de.route_concept_id,
                   de.exp_end_date, de.refills
            FROM wo_ok w
            JOIN drug_exposure de ON de.person_id = w.person_id AND de.exp_date = w.d
        ),
        seen_before AS (
            SELECT DISTINCT a.person_id, a.d, a.ingredient_id
            FROM (SELECT DISTINCT person_id, d, ingredient_id FROM at_d) a
            JOIN drug_exposure de
              ON de.person_id = a.person_id AND de.ingredient_id = a.ingredient_id
             AND de.exp_date >= (a.d - INTERVAL '{p.washout_days} days')
             AND de.exp_date < a.d
        ),
        new_rows AS (
            SELECT a.* FROM at_d a
            ANTI JOIN seen_before s
              ON s.person_id = a.person_id AND s.d = a.d
             AND s.ingredient_id = a.ingredient_id
        ),
        -- Classement par ingrédient, avec la seule information disponible à t0
        new_ingredients AS (
            SELECT person_id, d, ingredient_id,
                BOOL_AND(is_monotherapy) AS all_mono,
                -- Ponctuel : uniquement des ordonnances de 1..{short_days} j sans renouvellement
                BOOL_AND(COALESCE(
                    drug_type_concept_id = {self._EHR_ORDER_TYPE}
                    AND (exp_end_date - d) BETWEEN 1 AND {short_days}
                    AND COALESCE(refills, 0) = 0, FALSE)) AS is_short,
                -- Local : uniquement des voies non systémiques
                BOOL_AND(COALESCE(route_concept_id IN ({local_routes}), FALSE)) AS is_local
            FROM new_rows
            GROUP BY 1, 2, 3
        ),{pers_cte}
        counted AS (
            SELECT n.*,
                   -- Extension : co-initiation comptée seulement si elle persiste
                   n.counted AND (n.is_target OR n.is_persistent) AS counted_x
            FROM (
                SELECT n.*,
                       (n.ingredient_id IN (SELECT ingredient_id FROM target_ingredients)) AS is_target,
                       (n.ingredient_id IN (SELECT ingredient_id FROM target_ingredients))
                         OR ({self._counted_expr()}) AS counted
                FROM (
                    SELECT n.*, COALESCE(f.in_scope, FALSE) AS in_scope,
                           COALESCE(f.procedural, FALSE) AS procedural,
                           {pers_expr} AS is_persistent
                    FROM new_ingredients n
                    LEFT JOIN ing_flags f ON f.ingredient_id = n.ingredient_id{pers_join}
                ) n
            ) n
        ),
        per_d AS (
            SELECT person_id, d,
                COUNT(*) FILTER (WHERE counted)::INTEGER AS n_new,
                COUNT(*) FILTER (WHERE counted AND is_target)::INTEGER AS n_new_targets,
                COALESCE(BOOL_AND(all_mono) FILTER (WHERE counted), TRUE) AS all_mono,
                COUNT(*) FILTER (WHERE counted_x)::INTEGER AS n_new_x,
                COUNT(*) FILTER (WHERE counted_x AND is_target)::INTEGER AS n_new_targets_x,
                COALESCE(BOOL_AND(all_mono) FILTER (WHERE counted_x), TRUE) AS all_mono_x
            FROM counted
            GROUP BY 1, 2
        ),

        -- 5. t0 = première date éligible selon les règles connues à t0 ('t0_info') ;
        -- à défaut, première date éligible selon l'extension persistance
        t0_main AS (
            SELECT person_id, MIN(d) AS t0
            FROM per_d
            WHERE {final_filter}
            GROUP BY 1
        ),
        t0_ext AS (
            SELECT person_id, MIN(d) AS t0
            FROM per_d
            WHERE {"TRUE" if ext else "FALSE"} AND {final_filter_x}
              AND person_id NOT IN (SELECT person_id FROM t0_main)
            GROUP BY 1
        ),
        t0_sel AS (
            SELECT person_id, t0, 't0_info' AS t0_rule FROM t0_main
            UNION ALL
            SELECT person_id, t0, 'persistence' AS t0_rule FROM t0_ext
        ),
        step_final AS (
            SELECT t.person_id, t.t0, t.t0_rule,
                (t.t0 + INTERVAL '{p.obs_post_days} days')::DATE AS t_6m,
                (t.t0 + INTERVAL '{p.followup_12m_days} days')::DATE AS t_12m,
                (t.t0 - w.obs_start)::INTEGER AS history_days_prior,
                (w.obs_end - t.t0)::INTEGER AS follow_up_days,
                ((w.obs_end - t.t0) >= {p.followup_12m_days})::BOOLEAN AS has_12m_followup
            FROM t0_sel t
            JOIN wo_ok w ON w.person_id = t.person_id AND w.d = t.t0
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
            NULL::BIGINT AS n_patients,
            t0_rule
        FROM step_final

        -- RECORD 1 : Attrition (patients ayant au moins une date passant l'étape)
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(DISTINCT person_id)::BIGINT, NULL::VARCHAR FROM target_exposures
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(DISTINCT person_id)::BIGINT, NULL::VARCHAR FROM obs_ok
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(DISTINCT person_id)::BIGINT, NULL::VARCHAR FROM wo_ok
        UNION ALL
        SELECT 1::TINYINT, NULL::BIGINT, NULL::DATE, NULL::DATE, NULL::DATE, NULL::INTEGER, NULL::INTEGER, NULL::BOOLEAN, NULL::BIGINT, COUNT(*)::BIGINT, NULL::VARCHAR FROM step_final
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
            "t0_rule",
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
        "t0_rule",
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
          # Règle la plus faible retenue si les deux passes diffèrent
          if key not in rows or r["t0_rule"] == "persistence":
            rows[key] = r
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
      # Un essai par patient : règle principale d'abord, puis t0 le plus précoce
      df_pts = (
          pl.concat(parts, how="vertical_relaxed")
          .with_columns((pl.col("t0_rule") != "t0_info").alias("_rule_rank"))
          .sort(["person_id", "_rule_rank", "t0", "combo_source"],
                descending=[False, False, False, True])
          .unique(subset=["person_id"], keep="first", maintain_order=True)
          .drop("_rule_rank")
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
