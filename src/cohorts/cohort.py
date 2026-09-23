# src/cohorts/cohort.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from src.catalog.catalog import DrugItem

# Les 6 biomarqueurs cliniques cibles du protocole (Section 4.4)
CLINICAL_LAB_BIOMARKERS = ["ldl_c", "hba1c", "egfr", "alt", "crp", "sbp"]


class DrugCohort:
    """Gestionnaire clinique et phénotypique multi-sources d'une cohorte de patients

    pour une molécule ou une association médicamenteuse.

    Gère deux sous-cohortes complémentaires :
      1. Stanford EHR (Longitudinal) :
         - Suivi temporel à t0, t0 + 6 mois et t0 + 12 mois.
         - MOTOR baseline z0 à t0 (sans les prescriptions de t0).
         - Mesures réelles des 6 biomarqueurs de laboratoire (baseline, 6m, 12m).
         - Changements du profil RABIT (Delta RABIT 6m et 12m) après ablation
           du médicament d'intérêt dans l'historique EHR.
      2. UK Biobank (Protéomique Olink mesurée) :
         - Participants ayant débuté le médicament entre 3 et 24 mois avant la prise de sang.
         - MOTOR baseline z0 à t0.
         - Niveaux réels mesurés sur le panel Olink Explore (~3 000 protéines).

    Permet le calcul de prototypes cliniques k-Means et de la distance inter-cohortes
    min-linkage cosinus.
    """

    def __init__(
        self,
        drug_id: int,
        name: str | None = None,
        kind: Literal[
            "monotherapy", "fixed_combination", "de_facto_combination"
        ] = "monotherapy",
        ingredient_concept_ids: list[int] | None = None,
        stanford_index: pl.DataFrame | None = None,
        ukb_index: pl.DataFrame | None = None,
    ):
        self.drug_id = int(drug_id)
        self.name = name or f"Drug_{self.drug_id}"
        self.kind = kind
        self.ingredient_concept_ids = ingredient_concept_ids or (
            [self.drug_id] if self.drug_id > 0 else []
        )

        # ---------------------------------------------------------------------
        # 1. Sous-cohorte Stanford EHR (Longitudinale)
        # DataFrame requis : [person_id, t0, t_6m, t_12m, ...]
        # ---------------------------------------------------------------------
        self.stanford_index = stanford_index
        self._stanford_motor_z0: np.ndarray | None = None  # (N_stanford, 768)
        self.stanford_labs: pl.DataFrame | None = None  # Labs réels à t0, 6m, 12m
        self._stanford_delta_rabit_6m: np.ndarray | None = (
            None  # (N_stanford, P)
        )
        self._stanford_delta_rabit_12m: np.ndarray | None = (
            None  # (N_stanford, P)
        )
        self._rabit_protein_names: list[str] | None = None

        # ---------------------------------------------------------------------
        # 2. Sous-cohorte UK Biobank (Olink mesuré)
        # DataFrame requis : [eid, t0, blood_draw_date, months_to_draw] (3 à 24m)
        # ---------------------------------------------------------------------
        self.ukb_index = ukb_index
        self._ukb_motor_z0: np.ndarray | None = None  # (N_ukb, 768)
        self._ukb_olink_measured: np.ndarray | None = None  # (N_ukb, P_olink)
        self._olink_protein_names: list[str] | None = None

        # ---------------------------------------------------------------------
        # 3. Prototypes cliniques
        # ---------------------------------------------------------------------
        self._prototypes: np.ndarray | None = None  # (k, 768)

    # -------------------------------------------------------------------------
    # Constructeur d'usine
    # -------------------------------------------------------------------------
    @classmethod
    def from_drug_item(
        cls,
        drug_item: DrugItem,
        stanford_index: pl.DataFrame | None = None,
        ukb_index: pl.DataFrame | None = None,
    ) -> DrugCohort:
        """Instancie la cohorte directement depuis un DrugItem du DrugCatalog."""
        return cls(
            drug_id=drug_item.drug_id,
            name=drug_item.name,
            kind=drug_item.kind,
            ingredient_concept_ids=list(drug_item.ingredient_concept_ids),
            stanford_index=stanford_index,
            ukb_index=ukb_index,
        )

    # -------------------------------------------------------------------------
    # Propriétés descriptives
    # -------------------------------------------------------------------------
    @property
    def n_stanford(self) -> int:
        return len(self.stanford_index) if self.stanford_index is not None else 0

    @property
    def n_ukb(self) -> int:
        return len(self.ukb_index) if self.ukb_index is not None else 0

    @property
    def n_total(self) -> int:
        return self.n_stanford + self.n_ukb

    @property
    def is_combination(self) -> bool:
        return len(self.ingredient_concept_ids) > 1

    @property
    def stanford_motor_z0(self) -> np.ndarray | None:
        return self._stanford_motor_z0

    @property
    def stanford_delta_rabit_6m(self) -> np.ndarray | None:
        return self._stanford_delta_rabit_6m

    @property
    def stanford_delta_rabit_12m(self) -> np.ndarray | None:
        return self._stanford_delta_rabit_12m

    @property
    def ukb_motor_z0(self) -> np.ndarray | None:
        return self._ukb_motor_z0

    @property
    def ukb_olink_measured(self) -> np.ndarray | None:
        return self._ukb_olink_measured

    @property
    def prototypes(self) -> np.ndarray | None:
        return self._prototypes

    # -------------------------------------------------------------------------
    # Sauvegarde et Chargement Disque
    # -------------------------------------------------------------------------
    def save(self, output_base_dir: str | Path) -> Path:
        """Sauvegarde l'ensemble des métadonnées, index et tenseurs alignés de la cohorte."""
        base_dir = Path(output_base_dir)
        cohort_dir = base_dir / f"cohort_{self.drug_id}"
        cohort_dir.mkdir(parents=True, exist_ok=True)

        # 1. Métadonnées JSON
        metadata = {
            "drug_id": self.drug_id,
            "name": self.name,
            "kind": self.kind,
            "ingredient_concept_ids": self.ingredient_concept_ids,
            "n_stanford": self.n_stanford,
            "n_ukb": self.n_ukb,
            "has_stanford_z0": self._stanford_motor_z0 is not None,
            "has_stanford_labs": self.stanford_labs is not None,
            "has_stanford_rabit": self._stanford_delta_rabit_6m is not None,
            "has_ukb_z0": self._ukb_motor_z0 is not None,
            "has_ukb_olink": self._ukb_olink_measured is not None,
            "k_prototypes": (
                len(self._prototypes) if self._prototypes is not None else None
            ),
        }
        with open(cohort_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. Index tables
        if self.stanford_index is not None:
            self.stanford_index.write_parquet(
                cohort_dir / "stanford_index.parquet"
            )
        if self.ukb_index is not None:
            self.ukb_index.write_parquet(cohort_dir / "ukb_index.parquet")

        # 3. Données cliniques de laboratoire (Stanford)
        if self.stanford_labs is not None:
            self.stanford_labs.write_parquet(
                cohort_dir / "stanford_labs.parquet"
            )

        # 4. Tenseurs MOTOR et outcomes (Parquet / NPY)
        if self._stanford_motor_z0 is not None:
            np.save(
                cohort_dir / "stanford_motor_z0.npy", self._stanford_motor_z0
            )

        if self._stanford_delta_rabit_6m is not None:
            np.save(
                cohort_dir / "stanford_delta_rabit_6m.npy",
                self._stanford_delta_rabit_6m,
            )
        if self._stanford_delta_rabit_12m is not None:
            np.save(
                cohort_dir / "stanford_delta_rabit_12m.npy",
                self._stanford_delta_rabit_12m,
            )
        if self._rabit_protein_names:
            with open(cohort_dir / "rabit_proteins.json", "w") as f:
                json.dump(self._rabit_protein_names, f)

        if self._ukb_motor_z0 is not None:
            np.save(cohort_dir / "ukb_motor_z0.npy", self._ukb_motor_z0)

        if self._ukb_olink_measured is not None:
            np.save(
                cohort_dir / "ukb_olink_measured.npy", self._ukb_olink_measured
            )
        if self._olink_protein_names:
            with open(cohort_dir / "olink_proteins.json", "w") as f:
                json.dump(self._olink_protein_names, f)

        # 5. Prototypes
        if self._prototypes is not None:
            k = len(self._prototypes)
            np.save(cohort_dir / f"prototypes_k{k}.npy", self._prototypes)

        return cohort_dir

    @classmethod
    def from_disk(cls, cohort_dir: str | Path) -> DrugCohort:
        """Reconstitue intégralement une DrugCohort depuis son dossier persistant."""
        path = Path(cohort_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {path}")

        meta_file = path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json introuvable dans {path}")

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        # Instanciation de base
        cohort = cls(
            drug_id=meta["drug_id"],
            name=meta.get("name"),
            kind=meta.get("kind", "monotherapy"),
            ingredient_concept_ids=meta.get("ingredient_concept_ids", []),
        )

        # Rechargement des index
        st_index_file = path / "stanford_index.parquet"
        if st_index_file.exists():
            cohort.stanford_index = pl.read_parquet(st_index_file)

        ukb_index_file = path / "ukb_index.parquet"
        if ukb_index_file.exists():
            cohort.ukb_index = pl.read_parquet(ukb_index_file)

        # Rechargement des tables d'outcomes cliniques
        labs_file = path / "stanford_labs.parquet"
        if labs_file.exists():
            cohort.stanford_labs = pl.read_parquet(labs_file)

        # Rechargement des tenseurs
        if (path / "stanford_motor_z0.npy").exists():
            cohort._stanford_motor_z0 = np.load(
                path / "stanford_motor_z0.npy"
            ).astype(np.float32)

        if (path / "stanford_delta_rabit_6m.npy").exists():
            cohort._stanford_delta_rabit_6m = np.load(
                path / "stanford_delta_rabit_6m.npy"
            ).astype(np.float32)

        if (path / "stanford_delta_rabit_12m.npy").exists():
            cohort._stanford_delta_rabit_12m = np.load(
                path / "stanford_delta_rabit_12m.npy"
            ).astype(np.float32)

        if (path / "rabit_proteins.json").exists():
            with open(path / "rabit_proteins.json") as f:
                cohort._rabit_protein_names = json.load(f)

        if (path / "ukb_motor_z0.npy").exists():
            cohort._ukb_motor_z0 = np.load(path / "ukb_motor_z0.npy").astype(
                np.float32
            )

        if (path / "ukb_olink_measured.npy").exists():
            cohort._ukb_olink_measured = np.load(
                path / "ukb_olink_measured.npy"
            ).astype(np.float32)

        if (path / "olink_proteins.json").exists():
            with open(path / "olink_proteins.json") as f:
                cohort._olink_protein_names = json.load(f)

        # Prototypes
        proto_files = list(path.glob("prototypes_k*.npy"))
        if proto_files:
            cohort._prototypes = np.load(proto_files[0]).astype(np.float32)

        return cohort

    # -------------------------------------------------------------------------
    # Alignement des données de la sous-cohorte Stanford
    # -------------------------------------------------------------------------
    def load_stanford_motor_z0(
        self, parquet_path: str | Path, id_cols: list[str] | None = None
    ) -> np.ndarray:
        """Charge et aligne strictement la matrice MOTOR z0 (N, 768) sur stanford_index

        en utilisant la clé composite (person_id, t0).
        """
        if self.stanford_index is None:
            raise ValueError(
                "stanford_index doit être initialisé avant d'aligner MOTOR z0."
            )

        keys = id_cols or ["person_id", "t0"]
        df_reps = pl.read_parquet(parquet_path)
        feature_cols = [
            c
            for c in df_reps.columns
            if c.startswith("data_") or c.startswith("emb_")
        ]
        if not feature_cols:
            feature_cols = [f"data_{i}" for i in range(768)]

        aligned = self.stanford_index.select(keys).join(
            df_reps.select(keys + feature_cols),
            on=keys,
            how="inner",
        )

        if len(aligned) != len(self.stanford_index):
            print(
                f"[Avertissement] Cohorte {self.drug_id} (Stanford) : "
                f"{len(aligned)}/{len(self.stanford_index)} patients disposent d'un embedding MOTOR z0."
            )

        self._stanford_motor_z0 = (
            aligned.select(feature_cols).to_numpy().astype(np.float32)
        )
        return self._stanford_motor_z0

    def load_stanford_rabit_deltas(
        self,
        parquet_6m: str | Path,
        parquet_12m: str | Path | None = None,
        id_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Aligne les deltas de profils synthétiques Delta RABIT à 6 mois et 12 mois

        calculés après ablation de la molécule de l'historique EHR.
        """
        if self.stanford_index is None:
            raise ValueError(
                "stanford_index doit être initialisé avant d'aligner les deltas RABIT."
            )

        keys = id_cols or ["person_id", "t0"]

        # 1. Delta 6 mois
        df_6m = pl.read_parquet(parquet_6m)
        protein_cols = [c for c in df_6m.columns if c not in keys]
        self._rabit_protein_names = protein_cols

        aligned_6m = self.stanford_index.select(keys).join(
            df_6m.select(keys + protein_cols), on=keys, how="inner"
        )
        self._stanford_delta_rabit_6m = (
            aligned_6m.select(protein_cols).to_numpy().astype(np.float32)
        )

        # 2. Delta 12 mois (si disponible)
        if parquet_12m is not None and Path(parquet_12m).exists():
            df_12m = pl.read_parquet(parquet_12m)
            aligned_12m = self.stanford_index.select(keys).join(
                df_12m.select(keys + protein_cols), on=keys, how="inner"
            )
            self._stanford_delta_rabit_12m = (
                aligned_12m.select(protein_cols).to_numpy().astype(np.float32)
            )

        return self._stanford_delta_rabit_6m, self._stanford_delta_rabit_12m

    # -------------------------------------------------------------------------
    # Alignement des données de la sous-cohorte UK Biobank
    # -------------------------------------------------------------------------
    def load_ukb_olink_data(
        self,
        parquet_olink: str | Path,
        id_col: str = "eid",
    ) -> np.ndarray:
        """Charge et aligne les taux réels de protéines mesurées au prélèvement Olink

        pour les participants ayant débuté la molécule 3 à 24 mois plus tôt.
        """
        if self.ukb_index is None:
            raise ValueError(
                "ukb_index doit être initialisé avant d'aligner les mesures Olink."
            )

        df_olink = pl.read_parquet(parquet_olink)
        protein_cols = [c for c in df_olink.columns if c != id_col]
        self._olink_protein_names = protein_cols

        aligned = self.ukb_index.select([id_col]).join(
            df_olink.select([id_col] + protein_cols), on=id_col, how="inner"
        )

        self._ukb_olink_measured = (
            aligned.select(protein_cols).to_numpy().astype(np.float32)
        )
        return self._ukb_olink_measured

    # -------------------------------------------------------------------------
    # Prototypes Cliniques Multi-Centroïdes (k-Means adaptatif sur z0)
    # -------------------------------------------------------------------------
    def compute_prototypes(
        self,
        k: int = 3,
        source: Literal["stanford", "ukb"] = "stanford",
        force_recompute: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        """Calcule ou charge k prototypes cliniques L2-normalisés dans l'espace MOTOR (768-d).

        Gère automatiquement le cas où l'effectif N <= k en prenant chaque patient
        comme son propre prototype.
        """
        reps = (
            self._stanford_motor_z0
            if source == "stanford"
            else self._ukb_motor_z0
        )
        if reps is None:
            raise ValueError(
                f"Représentations MOTOR z0 non chargées pour la source '{source}'."
            )

        n_samples = reps.shape[0]
        if n_samples == 0:
            raise ValueError(
                f"Cohorte {self.drug_id} ({self.name}) vide : impossible de calculer des prototypes."
            )

        # K-Means adaptatif
        if n_samples <= k:
            raw_prototypes = reps.copy()
        else:
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init="auto",
            )
            kmeans.fit(reps)
            raw_prototypes = kmeans.cluster_centers_.astype(np.float32)

        # Normalisation L2 stricte pour optimiser le calcul cosinus
        norms = np.linalg.norm(raw_prototypes, axis=1, keepdims=True)
        self._prototypes = raw_prototypes / np.maximum(norms, 1e-8)

        return self._prototypes

    # -------------------------------------------------------------------------
    # Distance Inter-Cohortes (Min-Linkage Cosinus)
    # -------------------------------------------------------------------------
    def min_cosine_distance(self, other: DrugCohort) -> float:
        """Calcule la métrique de chevauchement clinique :

            d_min(A, B) = min_{i,j} (1 - cos(mu_{A,i}, mu_{B,j})).

        Retourne un scalaire dans [0.0, 2.0]. Une valeur proche de 0 indique qu'au
        moins un sous-groupe clinique est partagé entre les deux molécules (indication commune).
        """
        if self._prototypes is None:
            raise ValueError(
                f"Prototypes non calculés pour {self.drug_id} ({self.name})."
            )
        if other._prototypes is None:
            raise ValueError(
                f"Prototypes non calculés pour {other.drug_id} ({other.name})."
            )

        sim_matrix = np.dot(self._prototypes, other._prototypes.T)
        max_sim = float(np.max(sim_matrix))

        return float(np.clip(1.0 - max_sim, 0.0, 2.0))

    # -------------------------------------------------------------------------
    # Export FEMR Prediction Times (Inférence MOTOR)
    # -------------------------------------------------------------------------
    def to_femr_prediction_times(
        self,
        source: Literal["stanford", "ukb"] = "stanford",
        timepoint: Literal["baseline_t0", "followup_6m", "followup_12m"] = "baseline_t0",
    ) -> pl.DataFrame:
        """Génère les points d'inférence temporels requis par le moteur FEMR / MOTOR :

        (patient_id, prediction_time) avec granularité à la minute.

        - 'baseline_t0' : date index t0 (sur l'historique sans les prescriptions de t0).
        - 'followup_6m' : date t0 + 6 mois (après ablation de la molécule d'intérêt).
        - 'followup_12m': date t0 + 12 mois (après ablation de la molécule d'intérêt).
        """
        if source == "stanford":
            if self.stanford_index is None:
                raise ValueError("stanford_index non initialisé.")
            df = self.stanford_index
            id_col = "person_id"

            if timepoint == "baseline_t0":
                time_col = "t0"
            elif timepoint == "followup_6m":
                time_col = "t_6m"
            elif timepoint == "followup_12m":
                time_col = "t_12m"
            else:
                raise ValueError(f"Timepoint inconnu : {timepoint}")

        elif source == "ukb":
            if self.ukb_index is None:
                raise ValueError("ukb_index non initialisé.")
            df = self.ukb_index
            id_col = "eid"
            time_col = "t0"
        else:
            raise ValueError(f"Source inconnue : {source}")

        return (
            df.select(
                [
                    pl.col(id_col).cast(pl.Int64).alias("patient_id"),
                    pl.col(time_col)
                    .dt.strftime("%Y-%m-%d %H:%M:00")
                    .alias("prediction_time"),
                ]
            )
            .unique(subset=["patient_id", "prediction_time"])
            .sort(["patient_id", "prediction_time"])
        )

    def __repr__(self) -> str:
        proto_str = (
            f", k_proto={len(self._prototypes)}"
            if self._prototypes is not None
            else ""
        )
        return (
            f"<DrugCohort id={self.drug_id} name='{self.name}' kind='{self.kind}' "
            f"(N_stanford={self.n_stanford:,}, N_ukb={self.n_ukb:,}{proto_str})>"
        )