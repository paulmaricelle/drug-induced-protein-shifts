#!/usr/bin/env python3
"""Audit contradictoire et validation empirique des cohortes extraites.

Stress-tests TTE : wash-out de 365 jours, fenêtres d'observation,
co-initiations à t0 et plausibilité clinique (âge/sexe).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import duckdb
import polars as pl
from src.catalog.catalog import DrugCatalog
from src.config import PathConfig, ProtocolConfig


def run_adversarial_audit(
    test_drugs: list[int] | None = None,
) -> None:
  paths = PathConfig(is_sample=False)
  protocol = ProtocolConfig()
  manifest_path = paths.output_cohorts_dir / "manifest.parquet"

  if not manifest_path.exists():
    print(f"[ERREUR] Manifeste introuvable dans {paths.output_cohorts_dir}")
    return

  print("=" * 85)
  print("AUDIT CONTRADICTOIRE DES COHORTES TARGET TRIAL (STARR)")
  print(f"Répertoire des cohortes : {paths.output_cohorts_dir}")
  print("=" * 85)

  manifest = pl.read_parquet(manifest_path)
  catalog = DrugCatalog.load(paths.catalog_path)

  # Initialisation du moteur DuckDB
  con = duckdb.connect()
  con.execute("PRAGMA threads=8;")
  con.execute("PRAGMA memory_limit='32GB';")

  # 1. Enregistrement des tables du cache
  con.execute(f"""
        CREATE VIEW drug_exposure AS 
        SELECT * FROM read_parquet('{paths.drug_exposure_parquet}');
        
        CREATE VIEW observation_period AS 
        SELECT * FROM read_parquet('{paths.observation_period_parquet}');
    """)

  # Détection robuste des noms de colonnes dans le cache
  de_cols = [
      col[0] for col in con.execute("DESCRIBE drug_exposure").fetchall()
  ]
  date_col = "exp_date" if "exp_date" in de_cols else "drug_exposure_start_date"

  op_cols = [
      col[0] for col in con.execute("DESCRIBE observation_period").fetchall()
  ]
  start_col = (
      "obs_start" if "obs_start" in op_cols else "observation_period_start_date"
  )
  end_col = (
      "obs_end" if "obs_end" in op_cols else "observation_period_end_date"
  )

  # 2. Résolution de la table person dans paths.omop_dir
  person_path = None
  for candidate_name in [
      "person.parquet",
      "person.csv",
      "PERSON.parquet",
      "PERSON.csv",
  ]:
    cand = paths.omop_dir / candidate_name
    if cand.exists():
      person_path = cand
      break

  if person_path is not None:
    if person_path.suffix == ".parquet":
      con.execute(
          f"CREATE VIEW person AS SELECT * FROM read_parquet('{person_path}');"
      )
    else:
      con.execute(
          "CREATE VIEW person AS SELECT * FROM"
          f" read_csv_auto('{person_path}');"
      )
    print(f"[SOURCE] Table person chargée : {person_path}")
  else:
    print(
        "[-] Table person non trouvée dans omop_dir. Plausibilité clinique"
        " ignorée."
    )

  # -------------------------------------------------------------------------
  # TEST 0 : Synchronisation Disque vs Manifeste
  # -------------------------------------------------------------------------
  print("\n>>> TEST 0 : Synchronisation Disque vs Manifeste")
  mismatch_count = 0
  saved_cohorts = manifest.filter(pl.col("status") == "SAVED")

  for row in saved_cohorts.iter_rows(named=True):
    cid = row["drug_id"]
    cohort_folder = paths.output_cohorts_dir / f"cohort_{cid}"
    index_file = cohort_folder / "stanford_index.parquet"

    if not index_file.exists():
      print(f"  [ÉCHEC] Dossier absent sur disque : cohort_{cid}")
      mismatch_count += 1
      continue

    actual_len = pl.scan_parquet(index_file).select(pl.len()).collect().item()
    if actual_len != row["n_final_stanford"]:
      print(
          f"  [ÉCHEC] Cohorte {cid} ({row['drug_name']}) : Manifeste ="
          f" {row['n_final_stanford']} vs Disque = {actual_len}"
      )
      mismatch_count += 1

  if mismatch_count == 0:
    print(
        f"  [OK] 100% des {saved_cohorts.height:,} cohortes enregistrées"
        " concordent avec le disque."
    )

  # Sélection des 5 monothérapies majeures avec leurs identifiants OMOP exacts
  benchmark_ids = test_drugs or [
      1503297,  # Metformin
      1545958,  # Atorvastatin
      1332418,  # Amlodipine
      1367500,  # Losartan
      1308216,  # Lisinopril
  ]

  for target_id in benchmark_ids:
    item = catalog.get(target_id)
    name = item.name if item else f"Molécule {target_id}"
    index_file = (
        paths.output_cohorts_dir
        / f"cohort_{target_id}"
        / "stanford_index.parquet"
    )

    if not index_file.exists():
      print(f"\n[SKIP] Cohorte {name} (ID: {target_id}) absente du disque.")
      continue

    print(f"\n{'='*85}")
    print(f"AUDIT PROFOND : {name.upper()} (drug_id = {target_id})")
    print(f"{'='*85}")

    cohort_df = pl.read_parquet(index_file)
    n_patients = cohort_df.height
    print(f"Taille de la cohorte : {n_patients:,} patients")

    con.register("current_cohort", cohort_df)

    # ---------------------------------------------------------------------
    # TEST 1 : Unicité stricte des patients
    # ---------------------------------------------------------------------
    dups = n_patients - cohort_df["person_id"].n_unique()
    if dups > 0:
      print(
          f"  [CRITIQUE - ÉCHEC] {dups:,} doublons de person_id détectés dans la"
          " cohorte !"
      )
    else:
      print("  [OK] Unicité patient : 0 doublon (1 essai par patient).")

    # ---------------------------------------------------------------------
    # TEST 2 : Respect des fenêtres d'observation protocolaires
    # ---------------------------------------------------------------------
    obs_violations = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE c.t0 < op.{start_col} OR c.t0 > op.{end_col}) AS t0_out_of_bounds,
                COUNT(*) FILTER (WHERE (c.t0 - op.{start_col}) < {protocol.obs_pre_days}) AS baseline_too_short,
                COUNT(*) FILTER (WHERE (op.{end_col} - c.t0) < {protocol.obs_post_days}) AS followup_too_short,
                COUNT(*) FILTER (WHERE c.has_12m_followup != ((op.{end_col} - c.t0) >= 365)) AS flag_12m_mismatch
            FROM current_cohort c
            JOIN observation_period op ON c.person_id = op.person_id
            WHERE c.t0 >= op.{start_col} AND c.t0 <= op.{end_col}
        """).pl()

    v_bounds = obs_violations["t0_out_of_bounds"][0]
    v_base = obs_violations["baseline_too_short"][0]
    v_post = obs_violations["followup_too_short"][0]
    v_flag = obs_violations["flag_12m_mismatch"][0]

    if v_bounds + v_base + v_post + v_flag > 0:
      print("  [CRITIQUE - ÉCHEC] Violations des fenêtres d'observation :")
      print(f"     - t0 hors période d'observation      : {v_bounds}")
      print(f"     - Baseline < {protocol.obs_pre_days}j : {v_base}")
      print(f"     - Suivi < {protocol.obs_post_days}j    : {v_post}")
      print(f"     - Désaccord flag 12 mois              : {v_flag}")
    else:
      print(
          f"  [OK] Fenêtres observationnelles : 100% conformes (>="
          f" {protocol.obs_pre_days}j baseline, >= {protocol.obs_post_days}j"
          " post-t0)."
      )

    # ---------------------------------------------------------------------
    # TEST 3 : Fuite de Wash-out (Prevalent-user bias sur la cible)
    # ---------------------------------------------------------------------
    washout_leak = con.execute(f"""
            SELECT COUNT(DISTINCT c.person_id) AS n_leaks
            FROM current_cohort c
            JOIN drug_exposure de ON c.person_id = de.person_id
            WHERE de.ingredient_id = {target_id}
              AND de.{date_col} >= (c.t0 - INTERVAL '{protocol.washout_days} days')
              AND de.{date_col} < c.t0
        """).pl()["n_leaks"][0]

    if washout_leak > 0:
      pct_leak = (washout_leak / n_patients) * 100
      print(
          f"  [CRITIQUE - ÉCHEC] Fuite de wash-out : {washout_leak:,} patients"
          f" ({pct_leak:.2f}%) sous traitement dans l'année précédant t0 !"
      )
    else:
      print(
          f"  [OK] Wash-out molécule ({protocol.washout_days}j) : 0"
          " contamination."
      )

    # ---------------------------------------------------------------------
    # TEST 4 : Wash-out des comparateurs de même classe ATC4
    # ---------------------------------------------------------------------
    comparators = []
    if hasattr(catalog, "get_atc4_family_ids"):
      comparators = [
          c
          for c in catalog.get_atc4_family_ids(target_id)
          if c != target_id
      ]

    if comparators:
      comp_str = ", ".join(str(c) for c in comparators)
      comp_leaks = con.execute(f"""
                SELECT COUNT(DISTINCT c.person_id) AS n_leaks
                FROM current_cohort c
                JOIN drug_exposure de ON c.person_id = de.person_id
                WHERE de.ingredient_id IN ({comp_str})
                  AND de.{date_col} >= (c.t0 - INTERVAL '{protocol.washout_days} days')
                  AND de.{date_col} < c.t0
            """).pl()["n_leaks"][0]

      if comp_leaks > 0:
        pct_comp = (comp_leaks / n_patients) * 100
        print(
            f"  [CRITIQUE - ÉCHEC] Fuite wash-out ATC4 : {comp_leaks:,} patients"
            f" ({pct_comp:.2f}%) sous molécule comparatrice en baseline !"
        )
      else:
        print(
            f"  [OK] Wash-out ATC4 ({len(comparators)} molécules comparatrices)"
            " : 0 contamination."
        )
    else:
      print("  [INFO] Aucun comparateur ATC4 renseigné dans le catalogue.")

    # ---------------------------------------------------------------------
    # TEST 5 : Pureté de la monothérapie à t0
    # ---------------------------------------------------------------------
    co_initiations = con.execute(f"""
            WITH baseline_history AS (
                SELECT DISTINCT de.person_id, de.ingredient_id
                FROM drug_exposure de
                JOIN current_cohort c ON de.person_id = c.person_id
                WHERE de.{date_col} >= (c.t0 - INTERVAL '{protocol.washout_days} days')
                  AND de.{date_col} < c.t0
            ),
            t0_prescriptions AS (
                SELECT DISTINCT de.person_id, de.ingredient_id
                FROM drug_exposure de
                JOIN current_cohort c ON de.person_id = c.person_id
                WHERE de.{date_col} = c.t0
                  AND de.ingredient_id != {target_id}
            )
            SELECT COUNT(DISTINCT t0.person_id) AS n_concurrent_initiations
            FROM t0_prescriptions t0
            LEFT JOIN baseline_history b 
                   ON t0.person_id = b.person_id 
                  AND t0.ingredient_id = b.ingredient_id
            WHERE b.ingredient_id IS NULL
        """).pl()["n_concurrent_initiations"][0]

    pct_co = (co_initiations / n_patients) * 100
    if co_initiations > 0:
      print(
          f"  [ATTENTION] Co-initiations aiguës à t0 : {co_initiations:,}"
          f" ({pct_co:.2f}%) patients ont initié une autre molécule le même"
          " jour."
      )
    else:
      print("  [OK] Monothérapie à t0 : 0 co-initiation concurrente détectée.")

    # ---------------------------------------------------------------------
    # TEST 6 : Plausibilité clinique (Âge et Sexe)
    # ---------------------------------------------------------------------
    if person_path is not None:
      p_cols = [col[0] for col in con.execute("DESCRIBE person").fetchall()]
      yob_col = "year_of_birth" if "year_of_birth" in p_cols else "YEAR_OF_BIRTH"
      gender_col = (
          "gender_concept_id"
          if "gender_concept_id" in p_cols
          else "GENDER_CONCEPT_ID"
      )

      demo_stats = con.execute(f"""
                SELECT 
                    ROUND(AVG(EXTRACT(YEAR FROM c.t0) - p.{yob_col}), 1) AS age_moyen,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(YEAR FROM c.t0) - p.{yob_col}), 1) AS age_median,
                    COUNT(*) FILTER (WHERE (EXTRACT(YEAR FROM c.t0) - p.{yob_col}) < 18) AS n_pediatrique,
                    COUNT(*) FILTER (WHERE (EXTRACT(YEAR FROM c.t0) - p.{yob_col}) > 110 OR (EXTRACT(YEAR FROM c.t0) - p.{yob_col}) < 0) AS n_age_aberrant,
                    ROUND(COUNT(*) FILTER (WHERE p.{gender_col} = 8532) * 100.0 / COUNT(*), 1) AS pct_femmes
                FROM current_cohort c
                JOIN person p ON c.person_id = p.person_id
            """).pl()

      print("  [PLAISIBILITÉ CLINIQUE]")
      print(
          f"     * Âge moyen : {demo_stats['age_moyen'][0]} ans | Médian :"
          f" {demo_stats['age_median'][0]} ans"
      )
      print(
          "     * Moins de 18 ans :"
          f" {demo_stats['n_pediatrique'][0]:,} ({demo_stats['n_pediatrique'][0]/n_patients*100:.2f}%)"
      )
      print(
          "     * Âges aberrants (<0 ou >110 ans) :"
          f" {demo_stats['n_age_aberrant'][0]}"
      )
      print(f"     * Ratio Femmes (OMOP 8532)        : {demo_stats['pct_femmes'][0]}%")

    # ---------------------------------------------------------------------
    # TEST 7 : Distribution temporelle des index t0
    # ---------------------------------------------------------------------
    yearly_dist = con.execute("""
            SELECT 
                EXTRACT(YEAR FROM t0)::INT AS annee,
                COUNT(*) AS n_inclusions
            FROM current_cohort
            GROUP BY 1
            ORDER BY 1
        """).pl()

    min_yr = yearly_dist["annee"].min()
    max_yr = yearly_dist["annee"].max()
    pic_yr = yearly_dist.sort("n_inclusions", descending=True)["annee"][0]
    print(
        f"  [CALENDRIER D'INCLUSION] t0 s'étend de {min_yr} à {max_yr} (Pic"
        f" annuel en {pic_yr})"
    )

  print(f"\n{'='*85}")
  print("AUDIT CONTRADICTOIRE TERMINÉ.")
  print("=" * 85)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Audit contradictoire des cohortes TTE."
  )
  parser.add_argument(
      "--drugs",
      nargs="+",
      type=int,
      default=None,
      help="Liste d'identifiants OMOP à auditer spécifiquement.",
  )
  args = parser.parse_args()

  run_adversarial_audit(test_drugs=args.drugs)