# scripts/sync_defacto_to_catalog.py
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.catalog.catalog import DrugCatalog
from src.config import PathConfig


def sync_defacto_combinations():
    paths = PathConfig(is_sample=False)
    catalog_file = paths.catalog_path
    cohorts_dir = paths.output_cohorts_dir

    print(f"Chargement du DrugCatalog depuis {catalog_file}...")
    loaded = DrugCatalog.load(catalog_file)
    initial_count = len(loaded)

    # Purge des de facto existants : le catalogue reflète exactement les cohortes sur disque
    catalog = loaded.without_kind("de_facto_combination")
    n_purged = initial_count - len(catalog)

    print(f"Analyse des dossiers de bi-thérapies de facto dans {cohorts_dir}...")
    n_added = catalog.register_de_facto_cohorts(cohorts_dir)

    print(f"\nBilan de synchronisation :")
    print(f"  * Items initiaux dans le catalogue : {initial_count:,}")
    print(f"  * Anciennes de facto retirées       : {n_purged:,}")
    print(f"  * Bi-thérapies de facto intégrées   : {n_added:,}")
    print(f"  * Total final du catalogue         : {len(catalog):,}")

    # Sauvegarde sur disque pour figer l'état
    catalog.save(catalog_file)
    print("✓ Catalogue mis à jour et scellé avec succès.")


if __name__ == "__main__":
    sync_defacto_combinations()
