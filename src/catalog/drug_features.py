# src/catalog/drug_features.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import polars as pl


def parse_action_type(action_type: str | None) -> float:
    """Attribue la direction biologique sigma_ap (-1, +1, ou 0)."""
    if action_type is None:
        return 0.0
    act = str(action_type).upper()
    if any(
        k in act
        for k in [
            "INHIBITOR",
            "ANTAGONIST",
            "BLOCKER",
            "SUPPRESSOR",
            "NEGATIVE",
            "INVERSE",
            "DEGRADER",
        ]
    ):
        return -1.0
    if any(
        k in act
        for k in [
            "ACTIVATOR",
            "AGONIST",
            "OPENER",
            "STIMULATOR",
            "POSITIVE",
            "REPLACEMENT",
        ]
    ):
        return 1.0
    return 0.0


def resolve_direction(row: dict) -> float:
    """Résout sigma_ap en priorité via la colonne 'direction' de ChEMBL, sinon via 'action_type'."""
    d = row.get("direction")
    if d is not None:
        try:
            val = float(d)
            if val in (-1.0, 1.0):
                return val
        except (ValueError, TypeError):
            pass
    return parse_action_type(row.get("action_type"))


def compute_target_potency_score(
    min_affinity_nm: float | None,
    cutoff_nm: float = 10_000.0,
    default_nm: float = 100.0,
) -> float:
    """Calcule un score d'affinité pharmacologique continue pour w_ap (Section 3.3).

    - Si l'affinité mesurée dépasse le seuil clinique (10 uM) : score = 0 (hors d'atteinte).
    - Si l'affinité est connue : score = 9.0 - log10(affinity_nm)
        * 1 nM   -> score = 9.0
        * 100 nM -> score = 7.0
        * 10 uM  -> score = 5.0
    - Si l'affinité n'est pas chiffrée mais que le mécanisme est validé : score par défaut (100 nM).
    """
    if min_affinity_nm is not None and not np.isnan(min_affinity_nm):
        aff = float(min_affinity_nm)
        if aff <= 0:
            aff = 0.1  # Plafond de liaison sub-nanomolaire
        if aff > cutoff_nm:
            return 0.0  # Élimination des liaisons supraphysiologiques in vitro
        return max(0.0, 9.0 - float(np.log10(aff)))

    # Mécanisme validé sans constante répertoriée dans ChEMBL affinities
    return max(0.0, 9.0 - float(np.log10(default_nm)))


def build_theoretical_drug_catalog(
    chembl_mapping_df: pl.DataFrame,
    vp_parquet_path: Path,
    output_catalog_path: Path,
    affinity_cutoff_nm: float = 10_000.0,
) -> pl.DataFrame:
    """Compile le catalogue théorique complet en projetant vp vers ua

    selon la formule du protocole : u_a = sum_p w_ap * sigma_ap * v_p.
    Les poids w_ap conservent leur magnitude pharmacologique absolue
    (pas de division par la somme des scores).
    """
    output_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Chargement de la matrice des cibles depuis {vp_parquet_path.name}...")

    # 1. Standardisation des noms de colonnes
    rename_map = {}
    if "concept_id" in chembl_mapping_df.columns:
        rename_map["concept_id"] = "ingredient_concept_id"
    if "concept_name" in chembl_mapping_df.columns:
        rename_map["concept_name"] = "ingredient_name"
    if "drug_chembl_id" in chembl_mapping_df.columns:
        rename_map["drug_chembl_id"] = "chembl_id"

    df = chembl_mapping_df.rename(rename_map)

    # 2. Charger vp (798, 1590) sous forme de dictionnaire {uniprot_id: ndarray}
    vp_df = pl.read_parquet(vp_parquet_path)
    dim_vp = len(vp_df["vp"][0])
    target_vectors = {
        row["uniprot_id"]: np.array(row["vp"], dtype=np.float32)
        for row in vp_df.iter_rows(named=True)
    }
    print(f"  -> {len(target_vectors)} vecteurs de cibles (dimension {dim_vp}).")

    # 3. Liste unique des ingrédients actifs
    all_ingredients = (
        df.select(["ingredient_concept_id", "ingredient_name"])
        .unique()
        .sort("ingredient_concept_id")
    )
    print(
        f"Calcul des vecteurs u_a pour {len(all_ingredients)} principes actifs avec pondération w_ap..."
    )

    # Grouper les interactions par ingrédient
    interactions = df.filter(pl.col("uniprot_id").is_not_null()).to_dicts()
    drug_targets: dict[int, list[dict]] = {}
    for inter in interactions:
        ing_id = inter["ingredient_concept_id"]
        drug_targets.setdefault(ing_id, []).append(inter)

    # 4. Projection matricielle pour chaque principe actif
    records = []
    n_off_targets_removed = 0

    for ing in all_ingredients.iter_rows(named=True):
        ing_id = ing["ingredient_concept_id"]
        name = ing["ingredient_name"]

        target_list = drug_targets.get(ing_id, [])

        # Filtrer sur les cibles valides dans v_p
        candidate_targets = [
            t
            for t in target_list
            if t.get("uniprot_id")
            and t["uniprot_id"].split("-")[0] in target_vectors
        ]

        # Calcul des scores de puissance et filtrage clinique
        valid_targets = []
        scores = []
        for t in candidate_targets:
            score = compute_target_potency_score(
                t.get("min_affinity_nm"), cutoff_nm=affinity_cutoff_nm
            )
            if score > 0.0:
                valid_targets.append(t)
                scores.append(score)
            else:
                n_off_targets_removed += 1

        if not valid_targets:
            # Molécule sans cible protéique active
            ua = np.zeros(dim_vp, dtype=np.float32)
            targets_struct = []
            chembl_ids = list(
                {t["chembl_id"] for t in target_list if t.get("chembl_id")}
            )
            mechanisms = []
            uids = []
        else:
            # Poids absolus non normalisés : w_ap conserve l'échelle d'affinité
            weights = [float(s) for s in scores]

            ua = np.zeros(dim_vp, dtype=np.float32)
            targets_struct = []
            uids = []
            mechanisms = []
            chembl_ids = set()

            for t, w in zip(valid_targets, weights):
                clean_uid = t["uniprot_id"].split("-")[0]
                sigma_ap = resolve_direction(t)
                v_p = target_vectors[clean_uid]

                eff_sigma = sigma_ap if sigma_ap != 0.0 else -1.0
                ua += float(w) * eff_sigma * v_p

                uids.append(clean_uid)
                if t.get("mechanism_of_action"):
                    mechanisms.append(t["mechanism_of_action"])
                if t.get("chembl_id"):
                    chembl_ids.add(t["chembl_id"])

                targets_struct.append(
                    {
                        "uniprot_id": clean_uid,
                        "gene_symbol": "",
                        "kd_nm": (
                            float(t["min_affinity_nm"])
                            if t.get("min_affinity_nm") is not None
                            else None
                        ),
                        "direction": int(eff_sigma),
                        "potency_weight": float(w),
                    }
                )

            chembl_ids = list(chembl_ids)

        records.append(
            {
                "ingredient_concept_id": ing_id,
                "ingredient_name": name,
                "chembl_id": chembl_ids[0] if chembl_ids else None,
                "n_targets": len(valid_targets),
                "target_uids": sorted(list(set(uids))),
                "mechanisms": sorted(list(set(mechanisms))),
                "targets_json": json.dumps(targets_struct),
                "ua": ua.tolist(),
            }
        )

    catalog_df = pl.DataFrame(records).with_columns(
        pl.col("ua").cast(pl.List(pl.Float32))
    )

    catalog_df.write_parquet(output_catalog_path)

    n_with_targets = catalog_df.filter(pl.col("n_targets") > 0).height
    size_mb = output_catalog_path.stat().st_size / (1024 * 1024)

    print(f"\n-> Catalogue théorique sauvegardé dans : {output_catalog_path}")
    print(f"-> Principes actifs totaux : {catalog_df.height}")
    print(
        f"-> Principes actifs avec cibles cliniquement actives : {n_with_targets}/{catalog_df.height} ({n_with_targets/catalog_df.height*100:.1f}%)"
    )
    print(
        f"-> Cibles in vitro éliminées (affinité > {affinity_cutoff_nm/1000:.0f} uM) : {n_off_targets_removed}"
    )
    print(f"-> Poids du fichier : {size_mb:.2f} Mo")

    return catalog_df