# scripts/extract_cohorts.py
import argparse
import sys
import time
from pathlib import Path
import polars as pl
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from src.catalog.catalog import DrugCatalog
from src.cohorts.cohort import DrugCohort
from src.cohorts.extractor import CohortExtractor
from src.config import PathConfig, ProtocolConfig


def run_batch_extraction(
    is_sample: bool = False,
    resume: bool = True,
    limit: int | None = None,
) -> None:
  paths = PathConfig(is_sample=is_sample)
  protocol = ProtocolConfig()
  paths.output_cohorts_dir.mkdir(parents=True, exist_ok=True)

  mode_label = (
      "ÉCHANTILLON (data/cache_benchmark)"
      if is_sample
      else "PRODUCTION (data/cache_full)"
  )
  print(f"\n{'='*80}")
  print("EXTRACTION DU PROTOCOLE TARGET TRIAL EMULATION")
  print(f"Mode                 : {mode_label}")
  print(f"Dossier de sortie    : {paths.output_cohorts_dir}")
  print(f"Seuil de-facto min   : {protocol.min_de_facto_size}")
  print(f"{'='*80}\n")

  # 1. Chargement du catalogue et instanciation du moteur DuckDB
  print("1. Chargement du DrugCatalog...")
  catalog = DrugCatalog.load(paths.catalog_path)
  extractor = CohortExtractor(catalog=catalog, paths=paths, protocol=protocol)

  # 2. Molécules cibles (monothérapies)
  monotherapies = [item for item in catalog if item.kind == "monotherapy"]
  target_ids = [item.drug_id for item in monotherapies]

  if limit is not None:
    target_ids = target_ids[:limit]
    print(f"Limitation active : premières {limit} molécules sélectionnées.")

  # 3. Gestion de la reprise
  already_extracted_ids = set()
  if resume:
    for folder in paths.output_cohorts_dir.glob("cohort_*"):
      if folder.is_dir() and (
          (folder / "stanford_index.parquet").exists()
          or (folder / "metadata.json").exists()
      ):
        try:
          cid = int(folder.name.replace("cohort_", ""))
          already_extracted_ids.add(cid)
        except ValueError:
          continue
    if already_extracted_ids:
      print(
          f"Mode reprise activé : {len(already_extracted_ids):,} cohortes déjà"
          " existantes ignorées."
      )

  queue_ids = [cid for cid in target_ids if cid not in already_extracted_ids]
  print(f"Cohortes monothérapies à traiter : {len(queue_ids):,}\n")

  manifest_records = []
  errors = []
  start_total = time.time()

  # 4. Boucle d'extraction vectorisée
  pbar = tqdm(queue_ids, desc="Extraction monothérapies", unit="molécule")
  for cid in pbar:
    item = catalog.get(cid)
    pbar.set_postfix_str(f"ID {cid} ({item.name[:12]})")

    try:
      t0_extract = time.time()
      cohort = extractor.extract_cohort(cid)
      elapsed = time.time() - t0_extract

      if cohort is not None and cohort.n_stanford > 0:
        cohort.save(paths.output_cohorts_dir)
        n_final = cohort.n_stanford
        n_12m = cohort.stanford_index.filter(pl.col("has_12m_followup")).height
        status = "SAVED"
      else:
        n_final = 0
        n_12m = 0
        status = "ZERO_PATIENT"

      manifest_records.append({
          "drug_id": cid,
          "drug_name": item.name,
          "kind": item.kind,
          "n_final_stanford": n_final,
          "n_with_12m_followup": n_12m,
          "extraction_time_s": round(elapsed, 3),
          "status": status,
      })

    except Exception as e:
      errors.append({"drug_id": cid, "error": str(e)})
      pbar.write(f"[ERREUR] ID {cid} ({item.name}): {e}")

  # 5. Extraction des bi-thérapies de facto
  print(
      f"\n2. Détection et export des bi-thérapies de facto (seuil N >="
      f" {protocol.min_de_facto_size})..."
  )
  de_facto_cohorts = extractor.export_valid_de_facto_cohorts(
      min_size=protocol.min_de_facto_size
  )
  for combo_cohort in de_facto_cohorts:
    combo_cohort.save(paths.output_cohorts_dir)
    n_12m = combo_cohort.stanford_index.filter(
        pl.col("has_12m_followup")
    ).height

    manifest_records.append({
        "drug_id": combo_cohort.drug_id,
        "drug_name": combo_cohort.name,
        "kind": combo_cohort.kind,
        "n_final_stanford": combo_cohort.n_stanford,
        "n_with_12m_followup": n_12m,
        "extraction_time_s": 0.0,
        "status": "SAVED",
    })

  total_time = time.time() - start_total

  # 6. Synthèse et export du manifeste
  if manifest_records:
    new_manifest_df = pl.DataFrame(manifest_records)
    manifest_path = paths.output_cohorts_dir / "manifest.parquet"
    csv_path = paths.output_cohorts_dir / "manifest.csv"

    if manifest_path.exists() and resume:
      existing = pl.read_parquet(manifest_path)
      full_manifest = pl.concat([existing, new_manifest_df]).unique(
          subset=["drug_id"], keep="last"
      )
    else:
      full_manifest = new_manifest_df

    full_manifest.sort("n_final_stanford", descending=True).write_parquet(
        manifest_path
    )
    full_manifest.sort("n_final_stanford", descending=True).write_csv(csv_path)

    n_saved = full_manifest.filter(pl.col("status") == "SAVED").height
    n_ge_100 = full_manifest.filter(pl.col("n_final_stanford") >= 100).height
    n_ge_50 = full_manifest.filter(pl.col("n_final_stanford") >= 50).height

    print(f"\n{'='*80}")
    print("SYNTHÈSE DE L'EXTRACTION :")
    print(f"Temps total d'exécution       : {total_time/60:.2f} minutes")
    print(f"Cohortes sauvegardées (N >= 1): {n_saved:,}")
    print(f"   -> dont N >= 100           : {n_ge_100:,}")
    print(f"   -> dont N >= 50            : {n_ge_50:,}")
    print(f"Manifestes exportés           : {manifest_path} (.parquet & .csv)")
    if errors:
      print(f"Nombre d'erreurs              : {len(errors)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Extraction en batch des cohortes du DrugCatalog."
  )
  parser.add_argument(
      "--full",
      action="store_true",
      help="Exécute sur le cache complet de production (data/cache_full).",
  )
  parser.add_argument(
      "--no-resume",
      action="store_true",
      help="Recalcule toutes les cohortes pour reconstruire le registre complet.",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=None,
      help="Limite le nombre de molécules à traiter.",
  )

  args = parser.parse_args()

  run_batch_extraction(
      is_sample=not args.full,
      resume=not args.no_resume,
      limit=args.limit,
  )