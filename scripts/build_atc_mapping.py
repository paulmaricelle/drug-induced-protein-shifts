# scripts/build_atc_mapping.py
from pathlib import Path
import sys
import time
import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VOCAB_DIR = Path("/remote/shared/collab/omop-vocabularies/v20250227")
OUT_PATH = ROOT_DIR / "data" / "ingredient_to_atc4.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_atc4_mapping() -> None:
    print("=" * 80)
    print("EXTRACTION DU MAPPING OFFICIEL RXNORM -> ATC4 (PRIMARY LINKS SEULS)")
    print(f"Vocabulaires OMOP : {VOCAB_DIR}")
    print(f"Sortie Parquet    : {OUT_PATH}")
    print("=" * 80)

    start_time = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=16;")
    con.execute("PRAGMA max_memory='32GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    query = f"""
    COPY (
        WITH rx_ingredients AS (
            SELECT 
                concept_id::BIGINT AS ingredient_id,
                concept_name AS ingredient_name
            FROM read_parquet('{VOCAB_DIR}/CONCEPT.parquet')
            WHERE vocabulary_id = 'RxNorm'
              AND concept_class_id = 'Ingredient'
              AND standard_concept = 'S'
        ),
        atc_concepts AS (
            SELECT 
                concept_id::BIGINT AS atc_concept_id,
                concept_code AS atc_code,
                concept_name AS atc_name,
                concept_class_id
            FROM read_parquet('{VOCAB_DIR}/CONCEPT.parquet')
            WHERE vocabulary_id = 'ATC'
        ),
        -- Extraction bidirectionnelle des relations primaires exclusives
        primary_relations AS (
            SELECT 
                CASE 
                    WHEN c1.vocabulary_id = 'RxNorm' THEN cr.concept_id_1::BIGINT 
                    ELSE cr.concept_id_2::BIGINT 
                END AS ingredient_id,
                CASE 
                    WHEN c1.vocabulary_id = 'ATC' THEN cr.concept_id_1::BIGINT 
                    ELSE cr.concept_id_2::BIGINT 
                END AS atc_concept_id,
                cr.relationship_id,
                CASE 
                    WHEN cr.relationship_id ILIKE '%pr lat%' THEN 1  -- Substance pure ATC 5th
                    WHEN cr.relationship_id ILIKE '%pr up%'  THEN 2  -- Classe thérapeutique parente
                    ELSE 3 
                END AS rank_rel
            FROM read_parquet('{VOCAB_DIR}/CONCEPT_RELATIONSHIP.parquet') cr
            JOIN read_parquet('{VOCAB_DIR}/CONCEPT.parquet') c1 ON cr.concept_id_1 = c1.concept_id
            JOIN read_parquet('{VOCAB_DIR}/CONCEPT.parquet') c2 ON cr.concept_id_2 = c2.concept_id
            WHERE (
                cr.relationship_id IN ('RxNorm - ATC pr lat', 'ATC - RxNorm pr lat',
                                       'RxNorm - ATC pr up',  'ATC - RxNorm pr up')
            )
            AND (cr.invalid_reason IS NULL OR cr.invalid_reason = '')
        ),
        matched AS (
            SELECT 
                pr.ingredient_id,
                rx.ingredient_name,
                SUBSTRING(atc.atc_code, 1, 5) AS atc4_code,
                atc.atc_code,
                atc.atc_name,
                pr.rank_rel,
                -- Priorité aux mono-substances pures
                CASE 
                    WHEN LOWER(TRIM(atc.atc_name)) = LOWER(TRIM(rx.ingredient_name)) THEN 1
                    WHEN atc.atc_name NOT ILIKE '%comb%' 
                     AND atc.atc_name NOT ILIKE '% and %' 
                     AND atc.atc_name NOT ILIKE '%with%' THEN 2
                    ELSE 3
                END AS rank_mono
            FROM primary_relations pr
            JOIN rx_ingredients rx ON pr.ingredient_id = rx.ingredient_id
            JOIN atc_concepts atc ON pr.atc_concept_id = atc.atc_concept_id
        ),
        ranked AS (
            SELECT 
                m.ingredient_id,
                m.ingredient_name,
                m.atc4_code,
                a4.atc_name AS atc4_name,
                a4.atc_concept_id AS atc4_concept_id,
                ROW_NUMBER() OVER (
                    PARTITION BY m.ingredient_id 
                    ORDER BY m.rank_rel ASC, m.rank_mono ASC, m.atc_code ASC
                ) AS rn
            FROM matched m
            JOIN atc_concepts a4 ON m.atc4_code = a4.atc_code 
                                AND (a4.concept_class_id = 'ATC 4th' OR LENGTH(a4.atc_code) = 5)
        )
        SELECT 
            ingredient_id,
            ingredient_name,
            atc4_code,
            atc4_name,
            atc4_concept_id
        FROM ranked
        WHERE rn = 1
        ORDER BY ingredient_id
    ) TO '{OUT_PATH}' (FORMAT PARQUET);
    """

    con.execute(query)
    elapsed = time.time() - start_time

    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
    n_atc = con.execute(f"SELECT COUNT(DISTINCT atc4_code) FROM read_parquet('{OUT_PATH}')").fetchone()[0]

    print(f"\nMapping ATC4 officiel généré en {elapsed:.1f} secondes :")
    print(f"  * Ingrédients cartographiés sans ambiguïté : {n_rows:,}")
    print(f"  * Classes ATC4 distinctes couvertes       : {n_atc:,}")
    print("=" * 80)


if __name__ == "__main__":
    build_atc4_mapping()