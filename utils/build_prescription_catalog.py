import duckdb
from pathlib import Path
import polars as pl

OMOP_DIR = Path("../../../remote/private/starr_omop_deid/ro/STARR_OMOP_tables/som-rit-phi-starr-prod.starr_omop_cdm54_confidential_lite_2026_07_22")
CATALOG_PATH = Path("data/embedded_drug_catalog.parquet")
OUTPUT_PATH = Path("data/ingredient_to_prescriptions.parquet")

con = duckdb.connect()

# Charger les ingrédients cibles
catalog_df = pl.read_parquet(CATALOG_PATH)
target_ingredients = catalog_df.select(
    pl.col("rxnorm_concept_id").cast(pl.Int64).alias("ingredient_id")
)
con.register("target_ingredients", target_ingredients)

concept_ancestor_glob = str(OMOP_DIR / "concept_ancestor" / "*.csv.zst")
concept_glob = str(OMOP_DIR / "concept" / "*.csv.zst")

query = f"""
WITH target_descendants AS (
    SELECT 
        ti.ingredient_id,
        ca.descendant_concept_id
    FROM read_csv('{concept_ancestor_glob}', auto_detect=true) ca
    JOIN target_ingredients ti 
      ON ca.ancestor_concept_id = ti.ingredient_id
),
ingredient_counts AS (
    SELECT 
        ca.descendant_concept_id,
        COUNT(DISTINCT ca.ancestor_concept_id) AS nb_ingredients
    FROM read_csv('{concept_ancestor_glob}', auto_detect=true) ca
    JOIN read_csv('{concept_glob}', auto_detect=true) c 
      ON ca.ancestor_concept_id = c.concept_id
    WHERE ca.descendant_concept_id IN (SELECT descendant_concept_id FROM target_descendants)
      AND c.concept_class_id = 'Ingredient'
      AND c.standard_concept = 'S'
    GROUP BY ca.descendant_concept_id
)
SELECT 
    td.ingredient_id,
    td.descendant_concept_id AS drug_concept_id,
    -- Flag : TRUE uniquement pour les monothérapies pures
    (ic.nb_ingredients = 1) AS is_monotherapy
FROM target_descendants td
JOIN ingredient_counts ic 
  ON td.descendant_concept_id = ic.descendant_concept_id;
"""

print("Résolution du graphe ontologique complet (monothérapies et combinaisons)...")
mapping_df = con.sql(query).pl()

# L'ingrédient lui-même est toujours une monothérapie
self_mapping = target_ingredients.select(
    pl.col("ingredient_id"),
    pl.col("ingredient_id").alias("drug_concept_id"),
    pl.lit(True).alias("is_monotherapy")
)

final_mapping_df = (
    pl.concat([mapping_df, self_mapping])
    .unique(subset=["ingredient_id", "drug_concept_id"])
    .sort(["ingredient_id", "drug_concept_id"])
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_mapping_df.write_parquet(OUTPUT_PATH)

print(f"Mapping exporté avec succès dans {OUTPUT_PATH}")
print(f"Nombre total de paires : {final_mapping_df.height:,}")
print(f"Dont monothérapies pures : {final_mapping_df.filter(pl.col('is_monotherapy')).height:,}")
print(f"Dont composantes de polythérapies : {final_mapping_df.filter(~pl.col('is_monotherapy')).height:,}")