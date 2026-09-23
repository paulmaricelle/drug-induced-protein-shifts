# src/pairs/pair.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import polars as pl


@dataclass
class CandidatePair:
    """Représentation canonique d'une paire de comparateurs actifs (a, a')

    au sein d'une strate de comparaison clinique (Section 4.2).
    """

    drug_id_a: int
    drug_id_b: int
    stratum_concept_id: int | None = None
    stratum_name: str | None = None

    # Drapeaux d'attribution par méthode (Section 4.2)
    by_indication: bool = False
    by_css: bool = False
    by_kmeans: bool = False
    by_atc4: bool = False

    # Métriques quantitatives associées
    prevalence_a: float | None = None
    prevalence_b: float | None = None
    css_score: float | None = None
    kmeans_min_dist: float | None = None

    # Métadonnées descriptives
    drug_name_a: str = ""
    drug_name_b: str = ""
    n_patients_a: int = 0
    n_patients_b: int = 0

    @property
    def pair_key(self) -> tuple[int, int, int]:
        """Clé canonique ordonnée invariante par permutation."""
        u = min(self.drug_id_a, self.drug_id_b)
        v = max(self.drug_id_a, self.drug_id_b)
        s = (
            self.stratum_concept_id
            if self.stratum_concept_id is not None
            else 0
        )
        return (u, v, s)

    @property
    def n_methods(self) -> int:
        """Nombre de méthodes indépendantes ayant proposé la paire."""
        return sum([self.by_indication, self.by_css, self.by_kmeans])

    @property
    def is_consensus(self) -> bool:
        """Priorisation du protocole : proposée par au moins deux méthodes (Section 4.2)."""
        return self.n_methods >= 2



class PairRegistry:
    """Gestionnaire et catalogue persistant des paires candidates comparatives."""

    def __init__(self, pairs: list[CandidatePair] | None = None):
        self._pairs: dict[tuple[int, int, int], CandidatePair] = {}
        if pairs:
            for p in pairs:
                self.add_or_update(p)
        self.SCHEMA = {
        "drug_id_a": pl.Int64,
        "drug_id_b": pl.Int64,
        "stratum_concept_id": pl.Int64,
        "stratum_name": pl.Utf8,
        "by_indication": pl.Boolean,
        "by_css": pl.Boolean,
        "by_kmeans": pl.Boolean,
        "by_atc4": pl.Boolean,
        "prevalence_a": pl.Float64,
        "prevalence_b": pl.Float64,
        "css_score": pl.Float64,
        "kmeans_min_dist": pl.Float64,
        "drug_name_a": pl.Utf8,
        "drug_name_b": pl.Utf8,
        "n_patients_a": pl.Int64,
        "n_patients_b": pl.Int64,
    }

    def add_or_update(self, pair: CandidatePair) -> None:
        key = pair.pair_key
        if key in self._pairs:
            existing = self._pairs[key]
            existing.by_indication = (
                existing.by_indication or pair.by_indication
            )
            existing.by_css = existing.by_css or pair.by_css
            existing.by_kmeans = existing.by_kmeans or pair.by_kmeans
            existing.by_atc4 = existing.by_atc4 or pair.by_atc4

            if pair.prevalence_a is not None:
                existing.prevalence_a = pair.prevalence_a
                existing.prevalence_b = pair.prevalence_b
            if pair.css_score is not None:
                existing.css_score = pair.css_score
            if pair.kmeans_min_dist is not None:
                existing.kmeans_min_dist = pair.kmeans_min_dist
            if pair.stratum_name and not existing.stratum_name:
                existing.stratum_name = pair.stratum_name
        else:
            self._pairs[key] = pair

    def to_dataframe(self) -> pl.DataFrame:
        records = [asdict(p) for p in self._pairs.values()]
        if not records:
            return pl.DataFrame(schema=self.SCHEMA)
        return pl.DataFrame(records, schema=self.SCHEMA)

    def save_parquet(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.write_parquet(path)
        print(f"-> PairRegistry sauvegardé : {path} ({len(self)} paires)")

    @classmethod
    def load_parquet(cls, filepath: str | Path) -> PairRegistry:
        path = Path(filepath)
        if not path.exists():
            return cls()
        df = pl.read_parquet(path)
        pairs = [CandidatePair(**row) for row in df.iter_rows(named=True)]
        return cls(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs.values())