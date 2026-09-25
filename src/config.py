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

  # t0 = première date d'exposition à la cible satisfaisant TOUS les critères
  # (fenêtres d'observation, wash-out 365 j, comptage des nouvelles initiations)

  # Comptage des AUTRES initiations à t0 (la cible est toujours comptée).
  # Règles fixées à l'avance, fondées uniquement sur l'information disponible à t0.
  # 'systemic' : catalogue + ingrédients dont l'ATC4 n'est pas exclu ; 'all' : tout RxNorm
  counter_scope: str = "systemic"
  # Groupes ATC sans effet protéomique attendu (hors ingrédients du catalogue, toujours comptés)
  excluded_atc_prefixes: tuple[str, ...] = (
      "V",  # allergènes, diagnostics, contrastes, radiopharmaceutiques, solvants
      "B05",  # solutés et électrolytes IV
      "A12",  # suppléments minéraux
      "D",  # dermatologie (dont antiseptiques D08)
      "S",  # organes des sens (collyres, gouttes auriculaires)
      "A01",  # stomatologie
      "C05",  # hémorroïdes, vasoprotecteurs locaux
      "G01",  # anti-infectieux gynécologiques locaux
      "R02",  # préparations pour la gorge
  )
  # Ordonnance ponctuelle ignorée : durée prescrite 1..N jours sans renouvellement (0 = désactivé)
  short_order_max_days: int = 14
  # Voies non systémiques ignorées
  ignore_local_routes: bool = True
  local_route_concept_ids: tuple[int, ...] = (
      4263689,  # Topical
      40549429,  # Ocular
      4184451,  # Ophthalmic
      4023156,  # Otic
      4057765,  # Vaginal
      4302785,  # Intravitreal
      4157760,  # Intraocular
      4163765,  # Dental
      40490866,  # Periodontal
      4163770,  # Subconjunctival
      4303673,  # Retrobulbar
      4303409,  # Intracameral
      37174548,  # Suprachoroidal
      4168656,  # Intratympanic
      4006860,  # Intra-articular
      4157758,  # Intralesional
      4156706,  # Intradermal
      37397638,  # Infiltration
      4156708,  # Periarticular
      4163768,  # Intrabursal
      4302352,  # Intrasynovial
      4186838,  # Intravesical
      4233974,  # Urethral
      46270168,  # Sublesional
  )
  # Agents strictement procéduraux ignorés
  ignore_procedural: bool = True
  procedural_atc_prefixes: tuple[str, ...] = ("N01A", "M03A")
  procedural_ingredient_names: tuple[str, ...] = ("neostigmine", "sugammadex")

  # Extension « persistance » (analyse secondaire, colonne t0_rule = 'persistence') :
  # pour les patients sans date éligible selon les règles ci-dessus (t0_rule = 't0_info'),
  # une co-initiation non réexposée dans [d + a, d + b] j est aussi ignorée. Utilise une
  # information postérieure à t0 : à distinguer des ancrages causaux principaux.
  persistence_extension: bool = True
  persistence_window_days: tuple[int, int] = (30, 182)


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
  atc4_path: Path = ROOT_DIR / "data" / "ingredient_to_atc4.parquet"

  @property
  def output_cohorts_dir(self) -> Path:
    # Cohortes échantillon séparées pour ne jamais polluer la production
    sub = "cohorts_benchmark" if self.is_sample else "cohorts"
    return self.root_dir / "data" / sub

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