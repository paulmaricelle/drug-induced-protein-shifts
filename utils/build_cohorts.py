import argparse
import sys
import time
from pathlib import Path
import polars as pl
from tqdm import tqdm

# Ancrage dynamique du sys.path sur la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from the_map.config import PathConfig, ProtocolConfig
from the_map.drugCatalog import DrugCatalog
from the_map.cohort_extractor import CohortExtractor


def run_batch_extraction(
    is_sample: bool = False,
    min_patients: int = 10,
    resume: bool = True,
    limit: int | None = None,
) -> None:
    paths = PathConfig(is_sample=is_sample)
    protocol = ProtocolConfig()

    paths.output_cohorts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXTRACTION AUTOMATISÉE DES COHORTES")
    print(f"Mode : {'ÉCHANTILLON (data/cache_benchmark)' if is_sample else 'PRODUCTION COMPLÈTE (data/cache_full)'}")
    print(f"Dossier de sortie : {paths.output_cohorts_dir}")
    print(f"{'='*80}\n")

    # 1. Chargement unique du catalogue et du mapping
    print("Chargement du catalogue et des mappings...")
    catalog = DrugCatalog.from_parquet(
        catalog_path=paths.catalog_path,
        mapping_path=paths.mapping_path,
    )
    extractor = CohortExtractor(catalog=catalog, paths=paths, protocol=protocol)

    # 2. Identification des molécules cibles valides
    # On filtre les molécules qui disposent d'au moins un code de prescription monothérapie
    all_concept_ids = catalog.concept_ids
    target_ids = [
        cid for cid in all_concept_ids 
        if len(catalog.get_prescription_codes(cid)) > 0
    ]

    print(f"Molécules dans le catalogue : {len(all_concept_ids):,}")
    print(f"Molécules avec prescriptions OMOP réelles : {len(target_ids):,}")

    if limit is not None:
        target_ids = target_ids[:limit]
        print(f"Limitation appliquée : premières {limit} molécules sélectionnées.")

    # 3. Détection des cohortes déjà extraites (évite de recalculer si interruption)
    already_extracted_ids = set()
    if resume:
        for folder in paths.output_cohorts_dir.glob("cohort_*"):
            if folder.is_dir() and (folder / "target_patients.parquet").exists():
                try:
                    cid = int(folder.name.replace("cohort_", ""))
                    already_extracted_ids.add(cid)
                except ValueError:
                    continue
        if already_extracted_ids:
            print(f"Mode reprise activé : {len(already_extracted_ids):,} cohortes déjà existantes seront ignorées.")

    queue_ids = [cid for cid in target_ids if cid not in already_extracted_ids]
    print(f"Cohortes à extraire : {len(queue_ids):,}\n")

    # 4. Boucle d'extraction vectorisée
    manifest_records = []
    errors = []

    start_total = time.time()

    pbar = tqdm(queue_ids, desc="Extraction des cohortes", unit="molécule")
    for cid in pbar:
        item = catalog.get(cid)
        pbar.set_postfix_str(f"ID {cid} ({item.name[:12]})")

        try:
            t0_extract = time.time()
            cohort = extractor.extract_cohort(cid)
            elapsed = time.time() - t0_extract

            raw_candidates = 0
            if cohort.attrition_summary.height > 0:
                raw_candidates = cohort.attrition_summary["n_patients"][0]

            retention_rate = (
                (cohort.n_target / raw_candidates * 100) if raw_candidates > 0 else 0.0
            )

            # Persistance uniquement si l'effectif final est exploitable pour l'inférence
            # Dans la boucle d'extraction :
            if cohort.n_target > 0:
                cohort.save(paths.output_cohorts_dir)
                status = "SAVED"
            else:
                status = "ZERO_PATIENT"

            manifest_records.append({
                "rxnorm_concept_id": cid,
                "rxnorm_name": item.name,
                "atc3": item.atc3,
                "n_raw_candidates": raw_candidates,
                "n_final_target": cohort.n_target,
                "retention_rate_pct": round(retention_rate, 2),
                "extraction_time_s": round(elapsed, 2),
                "status": status,
            })

        except Exception as e:
            errors.append({"rxnorm_concept_id": cid, "error": str(e)})
            pbar.write(f"[ERREUR] ID {cid} ({item.name}): {e}")

    total_time = time.time() - start_total

    # 5. Synthèse et export du manifeste d'indexation
    if manifest_records:
        new_manifest_df = pl.DataFrame(manifest_records)
        manifest_path = paths.output_cohorts_dir / "manifest.parquet"

        if manifest_path.exists() and resume:
            existing_manifest = pl.read_parquet(manifest_path)
            full_manifest = pl.concat([existing_manifest, new_manifest_df]).unique(
                subset=["rxnorm_concept_id"], keep="last"
            )
        else:
            full_manifest = new_manifest_df

        full_manifest.sort("n_final_target", descending=True).write_parquet(manifest_path)
        csv_path = paths.output_cohorts_dir / "manifest.csv"
        full_manifest.sort("n_final_target", descending=True).write_csv(csv_path)

        n_saved = full_manifest.filter(pl.col("status") == "SAVED").height
        print(f"\n{'='*80}")
        print(f"SYNTHÈSE DU RUN :")
        print(f"Temps total : {total_time/60:.1f} minutes")
        print(f"Cohortes enregistrées {n_saved:,}")
        print(f"Manifeste mis à jour : {manifest_path} (et {csv_path})")
        if errors:
            print(f"Erreurs rencontrées : {len(errors)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction globale des cohortes du catalogue.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Exécute sur le cache complet de production (data/cache_full). Si omis, tourne sur data/cache_benchmark.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recalcule toutes les cohortes même si un dossier existe déjà.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre de molécules à traiter (utile pour debug/validation).",
    )

    args = parser.parse_args()

    run_batch_extraction(
        is_sample=False,
        min_patients=args.min_patients,
        resume=not args.no_resume,
        limit=args.limit,
    )