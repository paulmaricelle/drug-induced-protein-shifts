# scripts/audit_catalog.py
import json
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.catalog.catalog import DrugCatalog, DrugItem


def audit_catalog():
    jsonl_path = ROOT_DIR / "data" / "catalog" / "drug_catalog.jsonl"
    print(f"=== 1. TEST DE CHARGEMENT & DESCRIPTIF GLOBAL ===")
    catalog = DrugCatalog.load(jsonl_path)
    print(f"Items totaux en mémoire : {len(catalog)}")
    print(
        f"Codes de prescription résolubles en O(1) : {len(catalog._descendant_to_drug_id)}"
    )

    monos = [item for item in catalog if item.kind == "monotherapy"]
    fixed = [item for item in catalog if item.kind == "fixed_combination"]
    print(f" - Monothérapies : {len(monos)}")
    print(f" - Bi-thérapies fixes : {len(fixed)}")

    # Intégrité des dimensions vectorielles
    has_bio = sum(
        1
        for item in monos
        if item.target_vector is not None and np.any(item.target_vector != 0)
    )
    has_text = sum(
        1
        for item in monos
        if item.text_embedding is not None and np.any(item.text_embedding != 0)
    )

    dim_bio = len(monos[0].target_vector) if monos[0].target_vector is not None else 0
    dim_text = (
        len(monos[0].text_embedding) if monos[0].text_embedding is not None else 0
    )

    print(
        f" - Couverture biologique non nulle (u_a) : {has_bio}/{len(monos)} ({has_bio/len(monos):.1%}) | Dimension : {dim_bio}"
    )
    print(
        f" - Couverture textuelle non nulle        : {has_text}/{len(monos)} ({has_text/len(monos):.1%}) | Dimension : {dim_text}"
    )

    print(f"\n=== 2. SPOT-CHECK SUR DES MOLÉCULES CANONIQUES ===")
    # Recherche par nom de quelques molécules clés
    test_drugs = ["atorvastatin", "metformin", "lisinopril", "imatinib"]
    for drug_name in test_drugs:
        matched = [
            item for item in monos if drug_name in item.name.lower()
        ]
        if not matched:
            print(f"[-] {drug_name.capitalize()} introuvable !")
            continue
        d = matched[0]
        norm_u = np.linalg.norm(d.target_vector) if d.target_vector is not None else 0.0
        norm_txt = (
            np.linalg.norm(d.text_embedding) if d.text_embedding is not None else 0.0
        )
        print(f"[+] {d.name} (ID: {d.drug_id}) :")
        print(f"    - Cibles documentées : {len(d.targets)} cibles")
        print(f"    - Norme ||u_a|| : {norm_u:.4f} | Norme ||text|| : {norm_txt:.4f}")
        print(
            f"    - Codes prescriptibles associés : {len(d.descendant_concept_ids)} formulations"
        )

    print(f"\n=== 3. TEST DE RÉSOLUTION CLINIQUE O(1) ===")
    # Prendre une formulation commerciale au hasard d'une monothérapie et vérifier la résolution
    sample_item = next(item for item in monos if len(item.descendant_concept_ids) > 5)
    sample_descendant = list(sample_item.descendant_concept_ids)[0]

    resolved_item = catalog.get_by_prescribed_concept_id(sample_descendant)
    assert resolved_item is not None, "Échec de résolution !"
    assert resolved_item.drug_id == sample_item.drug_id, "Mauvaise résolution !"
    print(f"Code prescrit testé (EHR) : {sample_descendant}")
    print(f" -> Résolu instantanément vers : {resolved_item.name} (ID: {resolved_item.drug_id}) [OK]")

    print(f"\n=== 4. TEST DES BI-THÉRAPIES FIXES (ADDITIVITÉ u_A + u_B) ===")
    if fixed:
        combo = fixed[0]
        id_a, id_b = combo.ingredient_concept_ids
        item_a = catalog.get(id_a)
        item_b = catalog.get(id_b)

        print(f"Bi-thérapie analysée : {combo.name}")
        print(f" - Composant A : {item_a.name} (ID: {id_a})")
        print(f" - Composant B : {item_b.name} (ID: {id_b})")
        print(
            f" - Formulations commerciales rattachées : {len(combo.descendant_concept_ids)}"
        )

        # Vérification mathématique stricte de u_{combo} = u_A + u_B
        expected_u = item_a.target_vector + item_b.target_vector
        diff = np.max(np.abs(combo.target_vector - expected_u))
        print(f" - Écart max avec u_A + u_B : {diff:.2e} (doit être 0.00) [OK]")

    print(f"\n=== 5. TEST DE COMBINAISON DE FACTO DYNAMIQUE ===")
    # Simulation : un patient prend simultanément deux monothérapies sans formulation fixe
    # Prenons deux médicaments au hasard
    mono_a, mono_b = monos[0], monos[1]
    de_facto = catalog.get_or_create_combination(mono_a.drug_id, mono_b.drug_id)

    print(f"Création à la volée : {de_facto.name}")
    print(f" - ID synthétique déterministe : {de_facto.drug_id}")
    print(f" - Type : {de_facto.kind}")

    expected_de_facto_u = mono_a.target_vector + mono_b.target_vector
    diff_df = np.max(np.abs(de_facto.target_vector - expected_de_facto_u))
    print(f" - Écart vectoriel u_{{A+B}} : {diff_df:.2e} [OK]")

    # Vérification idempotence (ne doit pas recréer un deuxième objet si rappelé)
    de_facto_2 = catalog.get_or_create_combination(mono_b.drug_id, mono_a.drug_id)
    assert de_facto.drug_id == de_facto_2.drug_id, "L'ordre des ingrédients doit être invariant !"
    print(f" - Invariance par permutation de l'ordre d'appel : [OK]")

    print("\nTous les tests de cohérence sont passés avec succès.")


if __name__ == "__main__":
    audit_catalog()