#!/usr/bin/env python
"""
Screening symétrique des paires candidates via DrugCohort.
Intègre la résolution de cohorte incidente (New-User Design) :
- Si t0_A < t0_B : patient conservé uniquement dans A (naïf).
- Si t0_B < t0_A : patient conservé uniquement dans B (naïf).
- Si t0_A == t0_B : patient exclu de la paire (coprescription concomitante).
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

# Import du module racine
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from the_map.cohort_extractor import DrugCohort

COHORTS_DIR = Path("data/cohorts")
MANIFEST_PATH = COHORTS_DIR / "manifest.parquet"
OUTPUT_PAIRS_PATH = Path("data/candidate_pairs.parquet")

K_CENTROIDS = 3
MIN_PATIENTS_THRESHOLD = 50
MAX_COSINE_DISTANCE = 0.35
TOP_N_MATCHES_PER_DRUG = 10
MUTUAL_NEIGHBORS_ONLY = False


def load_cohorts() -> list[DrugCohort]:
    """Reconstitue les objets DrugCohort avec leurs métadonnées et prototypes."""
    manifest = pl.read_parquet(MANIFEST_PATH)
    viable = manifest.filter(pl.col("n_final_target") >= MIN_PATIENTS_THRESHOLD)

    cohorts = []
    for row in viable.iter_rows(named=True):
        cid = row["rxnorm_concept_id"]
        c_dir = COHORTS_DIR / f"cohort_{cid}"
        rep_file = c_dir / "motor_reps.parquet"

        if rep_file.exists():
            name = row.get("concept_name") or f"Concept_{cid}"
            cohort = DrugCohort.from_disk(cohort_dir=c_dir, rxnorm_name=name)
            cohort.compute_prototypes(k=K_CENTROIDS, cohort_dir=c_dir)
            cohorts.append(cohort)

    return cohorts


def resolve_incident_exposure(
    df_a: pl.DataFrame, 
    df_b: pl.DataFrame
) -> tuple[int, int, int, int]:
    """
    Résout l'antériorité temporelle pour les patients présents dans les deux bras.
    Retourne : (n_clean_a, n_clean_b, n_cross_dropped, n_concomitant_dropped)
    """
    if df_a is None or df_b is None or df_a.is_empty() or df_b.is_empty():
        return 0, 0, 0, 0

    pts_a = df_a.select(["person_id", "t0"])
    pts_b = df_b.select(["person_id", "t0"])

    # Détection des patients communs
    overlap = pts_a.join(pts_b, on="person_id", how="inner", suffix="_b").rename({"t0": "t0_a"})

    if overlap.is_empty():
        return len(pts_a), len(pts_b), 0, 0

    # 1. Coprescription le jour même (t0_a == t0_b) -> Exclus des deux
    concomitant = overlap.filter(pl.col("t0_a") == pl.col("t0_b"))
    n_concomitant = len(concomitant)

    # 2. Séquentiels : A avant B -> Gardé dans A, supprimé de B
    a_first = overlap.filter(pl.col("t0_a") < pl.col("t0_b"))
    
    # 3. Séquentiels : B avant A -> Gardé dans B, supprimé de A
    b_first = overlap.filter(pl.col("t0_b") < pl.col("t0_a"))

    n_cross = len(a_first) + len(b_first)

    # Effectifs incidents effectifs
    n_clean_a = len(pts_a) - n_concomitant - len(b_first)
    n_clean_b = len(pts_b) - n_concomitant - len(a_first)

    return n_clean_a, n_clean_b, n_cross, n_concomitant


def build_symmetric_candidate_pairs(cohorts: list[DrugCohort]) -> pl.DataFrame:
    """Calcule les paires candidates et applique la résolution incidente."""
    n = len(cohorts)
    protos_tensor = np.array([c.prototypes for c in cohorts])

    # Distance min-linkage cosinus globale
    flat = protos_tensor.reshape(-1, 768)
    sims = np.dot(flat, flat.T).reshape(n, K_CENTROIDS, n, K_CENTROIDS)
    max_sims = np.max(sims, axis=(1, 3))
    
    dist_matrix = np.clip(1.0 - max_sims, 0.0, 2.0)
    np.fill_diagonal(dist_matrix, np.inf)

    # Sélection Top-N
    sorted_idx = np.argsort(dist_matrix, axis=1)
    top_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        valid = [
            j for j in sorted_idx[i, :TOP_N_MATCHES_PER_DRUG]
            if dist_matrix[i, j] <= MAX_COSINE_DISTANCE
        ]
        top_mask[i, valid] = True

    edge_mask = (top_mask & top_mask.T) if MUTUAL_NEIGHBORS_ONLY else (top_mask | top_mask.T)
    i_idx, j_idx = np.where(np.triu(edge_mask, k=1))

    print(f"Filtrage incident sur {len(i_idx):,} paires topologiques candidates...")

    records = []
    dropped_underpowered = 0

    for i, j in zip(i_idx, j_idx):
        c1, c2 = cohorts[i], cohorts[j]

        # Résolution des patients chevauchants
        n_clean_a, n_clean_b, n_cross, n_concomitant = resolve_incident_exposure(
            c1.target_patients, 
            c2.target_patients
        )

        # Vérification du seuil d'éligibilité post-nettoyage
        if n_clean_a < MIN_PATIENTS_THRESHOLD or n_clean_b < MIN_PATIENTS_THRESHOLD:
            dropped_underpowered += 1
            continue

        records.append({
            "drug_a_id": c1.rxnorm_concept_id,
            "drug_a_name": c1.rxnorm_name,
            "drug_b_id": c2.rxnorm_concept_id,
            "drug_b_name": c2.rxnorm_name,
            "min_cosine_dist": float(dist_matrix[i, j]),
            "n_raw_a": c1.n_target,
            "n_raw_b": c2.n_target,
            "n_clean_a": n_clean_a,
            "n_clean_b": n_clean_b,
            "n_cross_dropped": n_cross,
            "n_concomitant_dropped": n_concomitant,
        })

    df_pairs = pl.DataFrame(records).sort("min_cosine_dist")

    if dropped_underpowered > 0:
        print(f"{dropped_underpowered:,} paires écartées car leur effectif incident net est tombé sous {MIN_PATIENTS_THRESHOLD}.")

    return df_pairs


def main():
    cohorts = load_cohorts()
    print(f"{len(cohorts)} cohortes DrugCohort prêtes.")

    df_pairs = build_symmetric_candidate_pairs(cohorts)
    df_pairs.write_parquet(OUTPUT_PAIRS_PATH)
    
    print(f"\nPaires candidates incidentes enregistrées : {df_pairs.height:,}")
    print(f"Fichier : {OUTPUT_PAIRS_PATH}")
    print("\nAperçu des 5 premières paires :")
    print(
        df_pairs.select([
            "drug_a_name", "drug_b_name", "min_cosine_dist", 
            "n_clean_a", "n_clean_b", "n_cross_dropped", "n_concomitant_dropped"
        ]).head(5)
    )


if __name__ == "__main__":
    main()