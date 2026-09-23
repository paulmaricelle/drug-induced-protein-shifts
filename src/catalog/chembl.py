# src/catalog/chembl.py
from __future__ import annotations

from pathlib import Path
import re
import polars as pl

# Mots-clés des sels, hydrates, et isomères à neutraliser pour retrouver la base OMOP
SALT_MODIFIERS = {
    "hydrochloride",
    "dihydrochloride",
    "monohydrochloride",
    "hcl",
    "besylate",
    "besilate",
    "benzenesulfonate",
    "maleate",
    "fumarate",
    "succinate",
    "tartrate",
    "mesylate",
    "tosylate",
    "acetate",
    "gluconate",
    "citrate",
    "bromide",
    "chloride",
    "sulfate",
    "phosphate",
    "nitrate",
    "potassium",
    "monopotassium",
    "dipotassium",
    "sodium",
    "monosodium",
    "disodium",
    "calcium",
    "magnesium",
    "zinc",
    "anhydrous",
    "trihydrate",
    "dihydrate",
    "monohydrate",
    "pentahydrate",
    "sesquihydrate",
    "hydrate",
    "salt",
    "free",
    "base",
}


def normalize_drug_name(name: str | None) -> str:
  """Normalise un nom pharmaceutique ChEMBL vers sa base neutre RxNorm

  (ex: 'metformin hydrochloride' -> 'metformin', 'atorvastatin calcium' ->
  'atorvastatin').
  """
  if not name:
    return ""
  text = str(name).lower()
  # 1. Retrait des parenthèses et de leur contenu (ex: '(as besilate)', '(anhydrous)')
  text = re.sub(r"\(.*?\)", "", text)
  # 2. Remplacement de la ponctuation par des espaces
  text = re.sub(r"[,;:\-_/]", " ", text)
  # 3. Filtrage token par token
  tokens = text.split()
  clean_tokens = [t for t in tokens if t not in SALT_MODIFIERS]

  return " ".join(clean_tokens) if clean_tokens else text.strip()


def compute_biological_direction(
    action_type: str | None,
    mechanism: str | None,
) -> int:
  """Détermine sigma_ap (-1 ou +1) avec priorité stricte aux termes inhibiteurs/antagonistes."""
  text = f"{action_type or ''} {mechanism or ''}".lower()

  negative_patterns = [
      "antagonist",
      "inhibitor",
      "blocker",
      "negative",
      "suppressor",
      "inverse agonist",
      "degrader",
      "downregulator",
      "inactivator",
  ]
  if any(term in text for term in negative_patterns):
    return -1

  positive_patterns = [
      "agonist",
      "activator",
      "positive",
      "inducer",
      "potentiator",
      "stimulator",
      "opener",
      "enhancer",
  ]
  if any(term in text for term in positive_patterns):
    return 1

  return -1


def load_and_map_chembl_targets(
    vocab_dir: Path,
    chembl_dir: Path,
) -> pl.DataFrame:
  """Joint RxNorm avec ChEMBL, recalcule sigma_ap et associe les constantes d'affinité."""
  # 1. Ingrédients STANDARDS RxNorm issus des vocabulaires OMOP (standard_concept = 'S')
  concepts = (
      pl.read_parquet(
          vocab_dir / "CONCEPT.parquet",
          columns=[
              "concept_id",
              "concept_name",
              "concept_class_id",
              "vocabulary_id",
              "standard_concept",
          ],
      )
      .filter(
          (pl.col("vocabulary_id") == "RxNorm")
          & (pl.col("concept_class_id") == "Ingredient")
          & (
              pl.col("standard_concept") == "S"
          )  # Exclut Atorvastatin Calcium (40010636) au profit de Atorvastatin (1545958)
      )
      .with_columns(
          pl.col("concept_name").str.to_lowercase().alias("match_name")
      )
  )

  # 2. Mécanismes ChEMBL normalisés vers leur base neutre
  mech = (
      pl.read_parquet(chembl_dir / "chembl_mechanisms.parquet")
      .with_columns(
          pl.col("drug_name")
          .map_elements(normalize_drug_name, return_dtype=pl.String)
          .alias("match_name")
      )
      .filter(pl.col("match_name") != "")
  )

  # 3. Recalcul de sigma_ap
  mech = mech.with_columns(
      pl.struct(["action_type", "mechanism_of_action"])
      .map_elements(
          lambda s: compute_biological_direction(
              s["action_type"], s["mechanism_of_action"]
          ),
          return_dtype=pl.Int32,
      )
      .alias("direction")
  )

  # 4. Jointure RxNorm standard <-> ChEMBL normalisé
  matched = concepts.join(mech, on="match_name", how="inner")

  # 5. Priorisation des affinités (Kd > Ki > IC50)
  aff = pl.read_parquet(chembl_dir / "chembl_affinities.parquet")
  aff_pivoted = (
      aff.with_columns(
          pl.col("standard_type")
          .str.to_uppercase()
          .replace({"KD": 1, "KI": 2, "IC50": 3}, default=4)
          .alias("priority")
      )
      .sort(by=["drug_chembl_id", "uniprot_id", "priority", "min_affinity_nm"])
      .group_by(["drug_chembl_id", "uniprot_id"])
      .first()
  )

  # 6. Jointure finale et déduplication par affinité
  full_targets = (
      matched.join(
          aff_pivoted.select([
              "drug_chembl_id",
              "uniprot_id",
              "standard_type",
              "min_affinity_nm",
          ]),
          on=["drug_chembl_id", "uniprot_id"],
          how="left",
      )
      .sort(
          by=["concept_id", "uniprot_id", "min_affinity_nm"], nulls_last=True
      )
      .unique(subset=["concept_id", "uniprot_id"], keep="first")
  )

  return full_targets