# scripts/sync_defacto_to_catalog.py
import json
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.catalog.catalog import DrugCatalog, DrugItem
from src.config import PathConfig


def sync_defacto_combinations():
    paths = PathConfig(is_sample=False)
    catalog_file = paths.catalog_path
    cohorts_dir = paths.output_cohorts_dir

    print(f"Chargement du DrugCatalog depuis {catalog_file}...")
    catalog = DrugCatalog.load(catalog_file)
    initial_count = len(catalog)

    n_added = 0
    n_skipped = 0

    # Scan de tous les dossiers de bi-thérapies de facto extraits
    defacto_folders = sorted(list(cohorts_dir.glob("cohort_9*")))
    print(f"Analyse de {len(defacto_folders):,} dossiers de bi-thérapies de facto...")

    for folder in defacto_folders:
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        combo_id = int(meta["drug_id"])
        if combo_id in catalog._items:
            continue

        ing_ids = meta.get("ingredient_concept_ids", [])
        if len(ing_ids) != 2:
            continue

        item_a = catalog.get(ing_ids[0])
        item_b = catalog.get(ing_ids[1])

        if not item_a or not item_b:
            n_skipped += 1
            continue

        # Additivité stricte des cibles : u_{A+B} = u_A + u_B (Section 3.3)
        u_combo = None
        if item_a.target_vector is not None and item_b.target_vector is not None:
            u_combo = item_a.target_vector + item_b.target_vector

        # Embedding textuel combiné normalisé
        text_combo = None
        if item_a.text_embedding is not None and item_b.text_embedding is not None:
            text_combo = (item_a.text_embedding + item_b.text_embedding) / 2.0
            norm = np.linalg.norm(text_combo)
            if norm > 0:
                text_combo = text_combo / norm

        combo_item = DrugItem(
            drug_id=combo_id,
            name=meta.get("name", f"{item_a.name} + {item_b.name} (De Facto)"),
            kind="de_facto_combination",
            ingredient_concept_ids=ing_ids,
            ingredient_names=[item_a.name, item_b.name],
            atc4=None,
            descendant_concept_ids=set(),
            targets=item_a.targets + item_b.targets,
            target_vector=u_combo,
            text_embedding=text_combo,
        )

        catalog.add_item(combo_item)
        n_added += 1

    print(f"\nBilan de synchronisation :")
    print(f"  * Items initiaux dans le catalogue : {initial_count:,}")
    print(f"  * Bi-thérapies de facto intégrées   : {n_added:,}")
    print(f"  * Non rattachées (ingrédient absent): {n_skipped}")
    print(f"  * Total final du catalogue         : {len(catalog):,}")

    # Sauvegarde sur disque pour figer l'état
    catalog.save(catalog_file)
    print("✓ Catalogue mis à jour et scellé avec succès.")


if __name__ == "__main__":
    sync_defacto_combinations()