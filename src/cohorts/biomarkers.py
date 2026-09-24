# src/cohorts/biomarkers.py
"""Spécification unique des 6 biomarqueurs cliniques (Section 4.4).

Utilisée par scripts/cache_biomarkers.py (concepts à extraire) et
scripts/extract_biomarkers.py (harmonisation des unités, plages plausibles).
Les concepts, unités et facteurs ont été vérifiés sur la distribution agrégée
des mesures STARR (concept x unité).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Unités UCUM (concept_id OMOP)
MG_DL = 8840
MG_DL_CALC = 9028
MG_L = 8751
ML_MIN_M2 = 710208  # série eGFR tronquée (">60" enregistré comme 60)


@dataclass(frozen=True)
class LabConcept:
  concept_id: int
  priority: int = 1  # plus petit = préféré lorsqu'un même jour a plusieurs concepts
  factor: float = 1.0  # valeur harmonisée = factor * valeur + offset
  offset: float = 0.0
  # Facteurs spécifiques à une unité (unit_concept_id -> facteur)
  unit_factors: dict[int, float] = field(default_factory=dict)
  excluded_units: tuple[int, ...] = ()
  # Unités non renseignées exclues lorsque l'échelle est ambiguë
  require_unit: bool = False


@dataclass(frozen=True)
class Biomarker:
  name: str
  unit: str
  valid_range: tuple[float, float]  # dans l'unité harmonisée
  concepts: tuple[LabConcept, ...]


BIOMARKERS: tuple[Biomarker, ...] = (
    Biomarker("ldl", "mg/dL", (10.0, 400.0), (
        LabConcept(3028288),  # LDL calculé
        LabConcept(3009966),  # LDL dosage direct
        LabConcept(3028437),  # LDL sans méthode précisée
    )),
    Biomarker("hba1c", "%", (3.0, 20.0), (
        # Les lignes en mg/dL sont la glycémie moyenne estimée, pas l'HbA1c
        LabConcept(3004410, excluded_units=(MG_DL,)),
        LabConcept(3005673, excluded_units=(MG_DL,)),  # HPLC
        LabConcept(3003309, excluded_units=(MG_DL,)),  # électrophorèse
        LabConcept(3007263, excluded_units=(MG_DL,)),  # calcul
        # IFCC en mmol/mol -> NGSP % (équation maîtresse NGSP-IFCC)
        LabConcept(40762352, priority=2, factor=0.09148, offset=2.152),
    )),
    Biomarker("egfr", "mL/min/1.73m2", (3.0, 180.0), (
        # Formules sans coefficient racial uniquement ; priorité à CKD-EPI 2021
        LabConcept(1619025, priority=1),  # CKD-EPI 2021
        LabConcept(40764999, priority=2),  # CKD-EPI 2009
        LabConcept(46236952, priority=3, excluded_units=(ML_MIN_M2,)),  # MDRD
        LabConcept(3049187, priority=3),  # MDRD non-blacks
        LabConcept(3030354, priority=3),  # MDRD
        LabConcept(40771922, priority=4),  # formule non précisée
    )),
    Biomarker("alt", "U/L", (2.0, 2000.0), (
        LabConcept(3006923),  # U/L et UI/L équivalents
        LabConcept(3005755),  # avec phosphate de pyridoxal
    )),
    Biomarker("crp", "mg/L", (0.05, 500.0), (
        # CRP standard rapportée en mg/dL (majorité) ou mg/L : conversion en mg/L
        LabConcept(3020460, unit_factors={MG_DL: 10.0, MG_L: 1.0}, require_unit=True),
        LabConcept(3010156, unit_factors={MG_DL: 10.0, MG_L: 1.0}),  # CRP ultrasensible
    )),
    Biomarker("sbp", "mmHg", (60.0, 260.0), (
        LabConcept(3004249),
    )),
)


def all_concept_ids() -> list[int]:
  return sorted({c.concept_id for b in BIOMARKERS for c in b.concepts})


def concept_table_records() -> list[dict]:
  """Une ligne par (concept, unité spécifique) pour la jointure SQL d'harmonisation.

  unit_concept_id NULL = règle par défaut du concept ; excluded = ligne rejetée.
  """
  rows = []
  for b in BIOMARKERS:
    lo, hi = b.valid_range
    for c in b.concepts:
      base = dict(concept_id=c.concept_id, biomarker=b.name, priority=c.priority,
                  lo=lo, hi=hi, offset=c.offset)
      rows.append({**base, "unit_concept_id": None, "factor": c.factor,
                   "excluded": c.require_unit})
      for u, f in c.unit_factors.items():
        rows.append({**base, "unit_concept_id": u, "factor": f * c.factor, "excluded": False})
      for u in c.excluded_units:
        rows.append({**base, "unit_concept_id": u, "factor": c.factor, "excluded": True})
  return rows
