# scripts/build_mapping.py
import sys
import time
from pathlib import Path
import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
VOCAB_DIR = Path("/remote/shared/collab/omop-vocabularies/v20250227")
OUT_PATH = ROOT_DIR / "data" / "ingredient_to_prescriptions.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"{'='*75}")
print("GÉNÉRATION DU MAPPING OFFICIEL RXNORM (CONCEPT_ANCESTOR.csv)")
print(f"Source vocabulaire : {VOCAB_DIR}")
print(f"Destination        : {OUT_PATH}")
print(f"{'='*75}")

t0 = time.time()
con = duckdb.connect()
con.execute("PRAGMA threads=16;")
con.execute("PRAGMA max_memory='32GB';")
con.execute("PRAGMA preserve_insertion_order=false;")

query = f"""
COPY (
    WITH rx_ingredients AS (
        -- 1. Isolation préalable des seuls ingrédients actifs RxNorm
        SELECT concept_id::BIGINT AS ingredient_id
        FROM read_parquet('{VOCAB_DIR}/CONCEPT.parquet')
        WHERE vocabulary_id = 'RxNorm' 
          AND concept_class_id = 'Ingredient'
    ),
    ca_filtered AS (
        -- 2. Filtrage au vol de CONCEPT_ANCESTOR.csv sur ces ingrédients
        SELECT 
            ca.ancestor_concept_id::BIGINT AS ancestor_concept_id,
            ca.descendant_concept_id::BIGINT AS descendant_concept_id
        FROM read_csv('{VOCAB_DIR}/CONCEPT_ANCESTOR.csv', auto_detect=true) ca
        SEMI JOIN rx_ingredients rx 
               ON ca.ancestor_concept_id::BIGINT = rx.ingredient_id
    ),
    ingredient_counts AS (
        -- 3. Comptage du nombre d'ingrédients pour chaque forme descendante
        SELECT 
            descendant_concept_id AS drug_concept_id,
            COUNT(DISTINCT ancestor_concept_id) AS n_ingredients
        FROM ca_filtered
        GROUP BY descendant_concept_id
    )
    -- 4. Mapping final avec flag monothérapie
    SELECT 
        ca.ancestor_concept_id::BIGINT AS ingredient_id,
        ca.descendant_concept_id::BIGINT AS drug_concept_id,
        (ic.n_ingredients = 1)::BOOLEAN AS is_monotherapy
    FROM ca_filtered ca
    JOIN ingredient_counts ic 
      ON ca.descendant_concept_id = ic.drug_concept_id
) TO '{OUT_PATH}' (FORMAT PARQUET);
"""

con.execute(query)
elapsed = time.time() - t0

n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
n_ing = con.execute(f"SELECT COUNT(DISTINCT ingredient_id) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
metformin_rx = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}') WHERE ingredient_id = 1503297"
).fetchone()[0]

print(f"\nMapping généré en {elapsed:.1f} secondes.")
print(f"   -> Prescriptions cliniques indexées : {n_rows:,}")
print(f"   -> Ingrédients uniques couverts     : {n_ing:,}")
print(f"   -> Formes cliniques pour Metformine  : {metformin_rx} codes RxNorm")
print(f"{'='*75}")