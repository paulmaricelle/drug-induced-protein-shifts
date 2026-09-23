# src/config.py
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProtocolConfig:
  """Paramètres d'émulation d'essai cible (Section 4.1)."""

  obs_pre_days: int = 365  # Antériorité minimale dans STARR avant t0
  obs_post_days: int = 182  # Suivi minimal obligatoire (6 mois)
  followup_12m_days: int = 365  # Horizon complet à 12 mois
  washout_days: int = 365  # Wash-out molécule & classe ATC4
  min_cohort_size: int = (
      1  # Sauvegarder toute monothérapie ayant au moins 1 patient
  )
  min_de_facto_size: int = (
      50  # Seuil garde-fou combinatoire pour les bi-thérapies de facto
  )


@dataclass(frozen=True)
class PathConfig:
  is_sample: bool = False  # Par défaut : Cache complet (data/cache_full)
  root_dir: Path = ROOT_DIR

  omop_dir: Path = Path(
      "/remote/private/starr_omop_deid/ro/STARR_OMOP_tables/"
      "som-rit-phi-starr-prod.starr_omop_cdm54_confidential_lite_2026_07_22"
  )

  mapping_path: Path = ROOT_DIR / "data" / "ingredient_to_prescriptions.parquet"
  catalog_path: Path = ROOT_DIR / "data" / "catalog" / "drug_catalog.jsonl"
  output_cohorts_dir: Path = ROOT_DIR / "data" / "cohorts"

  @property
  def cache_dir(self) -> Path:
    sub = "cache_benchmark" if self.is_sample else "cache_full"
    return self.root_dir / "data" / sub

  @property
  def drug_exposure_parquet(self) -> Path:
    return self.cache_dir / "sampled_drug_exposure.parquet"

  @property
  def observation_period_parquet(self) -> Path:
    return self.cache_dir / "sampled_observation_period.parquet"