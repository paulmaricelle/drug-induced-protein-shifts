from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProtocolConfig:
    """Paramètres temporels et cliniques de l'entonnoir d'attrition."""
    obs_pre_days: int = 365        # Antériorité minimale dans STARR avant t0
    obs_post_days: int = 60        # Suivi minimal après t0
    washout_days: int = 365        # Fenêtre de wash-out ATC4 strict avant t0
    stability_pre_days: int = 30   # Antériorité minimale pour considérer une co-médication stable
    stability_post_days: int = 60  # Fenêtre post-index sans nouvelle co-médication incidente
    max_chronic_gap_days: int = 180# Seuil de persistance (gap max entre deux délivrances)
    min_treatment_coverage_days: int = 60 # Couverture min si prescription unique


@dataclass(frozen=True)
class PathConfig:
    # True pour une partie du cache (20% si inchangé) pour tester les différents filtres de patients
    is_sample: bool = True  

    mapping_path: Path = Path("data/ingredient_to_prescriptions.parquet")
    catalog_path: Path = Path("data/embedded_drug_catalog.parquet")
    output_cohorts_dir: Path = Path("data/cohorts")

    @property
    def cache_dir(self) -> Path:
        return Path("data/cache_benchmark") if self.is_sample else Path("data/cache_full")

    @property
    def drug_exposure_parquet(self) -> Path:
        return self.cache_dir / "sampled_drug_exposure.parquet"

    @property
    def observation_period_parquet(self) -> Path:
        return self.cache_dir / "sampled_observation_period.parquet"