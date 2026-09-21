from pathlib import Path
import polars as pl

ATHENA_DIR = Path("data/vocab_download")
OUTPUT_PATH = Path("data/clean_drug_catalog.parquet")

# 1. LISTE STRICTE DES EXCLUSIONS PHARMACOLOGIQUES
BANNED_ATC_PREFIXES = (
    "V",       # Diagnostic (V04), contrastes (V08), solvants (V07), radiopharma (V09/10), allergènes (V01)
    "A11",     # Vitamines
    "A12",     # Compléments minéraux
    "B05",     # Solutions pour perfusion, substituts du plasma, électrolytes IV
    "J06",     # Sérums et immunoglobulines (ex: clesrovimab)
    "J07",     # Vaccins
    "A06AC",   # Laxatifs de lest
    "A06AD",   # Laxatifs osmotiques
    "A02AA",   # Antiacides magnésium
    "A02AB",   # Antiacides aluminium
    "A02AD",   # Associations antiacides
    "S01",     # Ophtalmologie locale (collyres, ex: aceclidine)
    "S02",     # Otologie locale
    "S03",    # Préparations ophtalmiques et otologiques
    #AFFINAGE DERMATOLOGIQUE (On bannit les topiques stricts)
    "D01A",    # Antifongiques à usage topique (D01B reste autorisé : antifongiques systémiques comme la terbinafine)
    "D02",     # Émollients et protecteurs cutanés
    "D03",     # Préparations pour le traitement des plaies et ulcères
    "D04",     # Antiprurigineux topiques (antihistaminiques, anesthésiques locaux)
    "D05A",    # Antipsoriasiques topiques (D05B reste autorisé : rétinoïdes oraux comme l'acitrétine)
    "D06",     # Antibiotiques et chimiothérapie à usage dermatologique (crèmes)
    "D07",     # Corticoïdes à usage dermatologique (dermocorticoïdes locaux)
    "D08",     # Antiseptiques et désinfectants
    "D09",     # Pansements médicamenteux
    "D10A",    # Anti-acnéiques topiques (peroxyde de benzoyle, etc. ; D10B reste autorisé : isotrétinoïne orale)
    "D11A",
)

BANNED_LEXICAL_SUBSTRINGS = (
    "extract", "pollen", "allergen", "water", "saline", 
    "infusion", "rinse", "vaccine", "immunoglobulin", 
    "inert", "vehicle"
)


def build_clean_catalog(exclude_combinations: bool = True):
    """
    Construit le catalogue théorique purifié des médicaments pour 'The Map'.
    
    1 ligne = 1 molécule active unique (concept_id RxNorm Ingredient standard).

    Gestion des associations à doses fixes (paramètre exclude_combinations) :
    -------------------------------------------------------------------------
    Dans la nomenclature officielle ATC de l'OMS :
      - Un code ATC 5th se terminant par 01 à 49 correspond à une MONOTHÉRAPIE pure.
      - Un code ATC 5th se terminant par 50 ou plus (>= 50) correspond à une 
        ASSOCIATION À DOSES FIXES (ex: J01DI55 sulopenem + probenecid, 
        C09BA02 enalapril + hydrochlorothiazide).

    Conséquence sur la régression f(X) -> ΔY :
      - X provient de l'embedding textuel de la molécule isolée (ex: sulopenem).
      - Si le code ATC est une association (>= 50), le phénotype d'administration
        clinique dans les EHR correspond à deux principes actifs pris simultanément.
      - Le shift protéomique ΔY mesuré reflète alors l'effet combiné des deux molécules,
        introduisant une pollution systématique de l'étiquette (label pollution)
        pour un modèle censé prédire l'effet d'une molécule seule.

    Options :
      - exclude_combinations=True :
          1. Supprime les codes d'association (>= 50) au niveau ATC 5th.
          2. Si une molécule existe en monothérapie ET en association (ex: enalapril),
             seul son code pur (< 50) est conservé.
          3. Si une molécule n'existe QUE sous forme de cocktail (ex: sulopenem),
             elle est entièrement retirée du catalogue d'entraînement.
      - exclude_combinations=False :
          Conserve les associations. Nécessitera plus tard d'adapter l'entrée X
          (ex: pooling ou somme des embeddings des principes actifs du cocktail)
          ou de filtrer strictement sur les monothérapies au moment des requêtes EHR.
    """
    print(f"1. Lecture de CONCEPT.csv (exclude_combinations={exclude_combinations})...")
    concepts = pl.read_csv(
        ATHENA_DIR / "CONCEPT.csv",
        separator="\t",
        quote_char=None,
        ignore_errors=True,
        columns=["concept_id", "concept_name", "concept_code", "vocabulary_id", "concept_class_id", "standard_concept"]
    )

    # Ingrédients RxNorm standard valides
    rx_ingredients = concepts.filter(
        (pl.col("vocabulary_id") == "RxNorm") &
        (pl.col("concept_class_id") == "Ingredient") &
        (pl.col("standard_concept") == "S")
    ).select([
        pl.col("concept_id").alias("rxnorm_concept_id"),
        pl.col("concept_name").str.to_lowercase().str.strip_chars().alias("rxnorm_name")
    ])
    print(f"Total ingrédients RxNorm standard bruts : {len(rx_ingredients)}")

    # 2. Liens ATC niveau 5 (Substance chimique à 7 caractères, ex: C09AA02)
    atc_level_5 = concepts.filter(
        (pl.col("vocabulary_id") == "ATC") &
        (pl.col("concept_class_id") == "ATC 5th") &
        (pl.col("concept_code").str.len_chars() == 7)
    ).select([
        pl.col("concept_id").alias("atc_concept_id"),
        pl.col("concept_code").alias("atc5_code"),
        pl.col("concept_name").alias("atc5_name")
    ])

    # Filtrage des associations fixes (codes se terminant par >= 50)
    if exclude_combinations:
        # Les 2 derniers caractères indiquent le numéro de série dans le sous-groupe ATC
        atc_level_5 = atc_level_5.filter(
            pl.col("atc5_code").str.slice(5, 2).cast(pl.Int32, strict=False) < 50
        )
        print("Filtre activé : seuls les codes ATC 5th monothérapies (< 50) sont conservés.")

    print(f"Concepts ATC 5th retenus : {len(atc_level_5)}")

    print("3. Chargement des liaisons via CONCEPT_ANCESTOR.csv...")
    ancestors = pl.read_csv(
        ATHENA_DIR / "CONCEPT_ANCESTOR.csv",
        separator="\t",
        columns=["descendant_concept_id", "ancestor_concept_id"]
    )

    # Jointure directe : RxNorm -> ATC 5th valide
    rx_to_atc5 = (
        rx_ingredients.select("rxnorm_concept_id")
        .join(ancestors, left_on="rxnorm_concept_id", right_on="descendant_concept_id", how="inner")
        .join(atc_level_5, left_on="ancestor_concept_id", right_on="atc_concept_id", how="inner")
        .select(["rxnorm_concept_id", "atc5_code", "atc5_name"])
        .unique()
    )

    # 4. Élimination des molécules associées aux préfixes ATC bannis
    print("4. Application des exclusions pharmacologiques...")
    banned_conditions = [
        pl.col("atc5_code").str.starts_with(prefix) for prefix in BANNED_ATC_PREFIXES
    ]
    is_banned = banned_conditions[0]
    for cond in banned_conditions[1:]:
        is_banned = is_banned | cond

    banned_rx_ids = rx_to_atc5.filter(is_banned).select("rxnorm_concept_id").unique()
    print(f"Molécules éliminées par les préfixes ATC bannis : {len(banned_rx_ids)}")

    # Conservation des molécules pures valides
    valid_rx_to_atc = rx_to_atc5.join(banned_rx_ids, on="rxnorm_concept_id", how="anti")
    clean_rx = rx_ingredients.join(banned_rx_ids, on="rxnorm_concept_id", how="anti")

    # Filtre lexical résiduel
    for pattern in BANNED_LEXICAL_SUBSTRINGS:
        clean_rx = clean_rx.filter(~pl.col("rxnorm_name").str.contains(pattern))

    # 5. Agrégation : 1 ligne = 1 molécule
    atc_aggregated = (
        valid_rx_to_atc.group_by("rxnorm_concept_id")
        .agg([
            pl.col("atc5_code").unique().alias("atc_codes"),
            pl.col("atc5_name").unique().alias("atc_names"),
            pl.col("atc5_code").first().str.slice(0, 4).alias("atc3_primary"),
            pl.col("atc5_code").first().str.slice(0, 1).alias("atc1_code")
        ])
    )

    catalog = clean_rx.join(atc_aggregated, on="rxnorm_concept_id", how="inner")

    print(f"\n-> Catalogue propre final : {len(catalog)} molécules retenues.")
    print(catalog.head(10))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(OUTPUT_PATH)
    print(f"Sauvegardé sous : {OUTPUT_PATH}")
    return catalog


if __name__ == "__main__":
    build_clean_catalog(exclude_combinations=True)