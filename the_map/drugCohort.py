from pathlib import Path
from typing import Optional, Union
import numpy as np
import polars as pl
from sklearn.cluster import KMeans


class DrugCohort:
    """Gestionnaire d'une cohorte de patients pour une molécule cible.
    
    Gère la persistance locale des patients retenus, le chargement et l'alignement
    des représentations d'états cachés MOTOR (768-dim), le calcul adaptatif de 
    prototypes cliniques (k-Means multi-centroïdes) et la métrique de distance min-linkage.
    """

    def __init__(
        self,
        rxnorm_concept_id: int,
        rxnorm_name: Optional[str] = None,
        target_patients: Optional[pl.DataFrame] = None,
    ):
        self.rxnorm_concept_id = int(rxnorm_concept_id)
        self.rxnorm_name = rxnorm_name
        self.target_patients = target_patients
        self._motor_reps: Optional[np.ndarray] = None
        self._prototypes: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # Propriétés
    # -------------------------------------------------------------------------
    @property
    def n_target(self) -> int:
        return len(self.target_patients) if self.target_patients is not None else 0

    @property
    def motor_reps(self) -> Optional[np.ndarray]:
        return self._motor_reps

    @property
    def prototypes(self) -> Optional[np.ndarray]:
        return self._prototypes

    # -------------------------------------------------------------------------
    # I/O Disque : Sauvegarde et Chargement de base
    # -------------------------------------------------------------------------
    def save(self, output_base_dir: Union[str, Path]) -> Path:
        """Sauvegarde la table target_patients dans data/cohorts/cohort_<concept_id>/."""
        base_dir = Path(output_base_dir)
        cohort_dir = base_dir / f"cohort_{self.rxnorm_concept_id}"
        cohort_dir.mkdir(parents=True, exist_ok=True)

        if self.target_patients is not None:
            self.target_patients.write_parquet(cohort_dir / "target_patients.parquet")

        return cohort_dir

    @classmethod
    def from_disk(
        cls, 
        cohort_dir: Union[str, Path], 
        rxnorm_name: Optional[str] = None
    ) -> "DrugCohort":
        """Reconstitue un objet DrugCohort à partir d'un dossier cohorte sur disque."""
        path = Path(cohort_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {path}")

        try:
            cid = int(path.name.replace("cohort_", ""))
        except ValueError:
            raise ValueError(f"Nom de dossier non conforme (attendu 'cohort_<id>') : {path.name}")

        target_file = path / "target_patients.parquet"
        target_df = pl.read_parquet(target_file) if target_file.exists() else None

        return cls(
            rxnorm_concept_id=cid,
            rxnorm_name=rxnorm_name,
            target_patients=target_df,
        )

    # -------------------------------------------------------------------------
    # Intégration des représentations latentes MOTOR
    # -------------------------------------------------------------------------
    def load_motor_representations(
        self, 
        cohort_dir: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """Charge la matrice MOTOR (N, 768) en alignant strictement chaque ligne
        sur target_patients via les clés jointes (person_id, t0).
        """
        if cohort_dir is None:
            cohort_path = Path(f"data/cohorts/cohort_{self.rxnorm_concept_id}")
        else:
            cohort_path = Path(cohort_dir)

        rep_path = cohort_path / "motor_reps.parquet"
        if not rep_path.exists():
            raise FileNotFoundError(f"Représentations MOTOR introuvables : {rep_path}")

        # Recharger target_patients si absent de la RAM
        if self.target_patients is None:
            target_file = cohort_path / "target_patients.parquet"
            if not target_file.exists():
                raise FileNotFoundError(f"target_patients.parquet introuvable : {target_file}")
            self.target_patients = pl.read_parquet(target_file)

        df_reps = pl.read_parquet(rep_path)
        feature_cols = [f"data_{i}" for i in range(768)]

        # Jointure interne stricte pour garantir l'alignement index-to-index
        aligned = self.target_patients.select(["person_id", "t0"]).join(
            df_reps.select(["person_id", "t0"] + feature_cols),
            on=["person_id", "t0"],
            how="inner",
        )

        if len(aligned) != len(self.target_patients):
            print(
                f"[Avertissement] Cohorte {self.rxnorm_concept_id} : "
                f"{len(aligned)}/{len(self.target_patients)} patients disposent d'un embedding MOTOR."
            )

        self._motor_reps = aligned.select(feature_cols).to_numpy().astype(np.float32)
        return self._motor_reps

    # -------------------------------------------------------------------------
    # Multi-centroïdes adaptatifs (k-Means)
    # -------------------------------------------------------------------------
    def compute_prototypes(
        self,
        k: int = 3,
        cohort_dir: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        """Calcule ou charge k prototypes cliniques L2-normalisés.
        
        Gère automatiquement le cas N <= k (chaque patient devient son propre prototype).
        Persiste le résultat sous format 'prototypes_k{k}.npy' dans le dossier cohorte.
        """
        if cohort_dir is None:
            cohort_path = Path(f"data/cohorts/cohort_{self.rxnorm_concept_id}")
        else:
            cohort_path = Path(cohort_dir)

        proto_file = cohort_path / f"prototypes_k{k}.npy"

        # 1. Vérification du cache disque
        if proto_file.exists() and not force_recompute:
            self._prototypes = np.load(proto_file).astype(np.float32)
            return self._prototypes

        # 2. Vérification / Chargement des représentations MOTOR
        if self._motor_reps is None:
            self.load_motor_representations(cohort_path)

        n_samples = self._motor_reps.shape[0]
        if n_samples == 0:
            raise ValueError(f"Cohorte {self.rxnorm_concept_id} vide : impossible de calculer des prototypes.")

        # 3. K-Means adaptatif selon l'effectif réel
        if n_samples <= k:
            raw_prototypes = self._motor_reps.copy()
        else:
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init="auto",
            )
            kmeans.fit(self._motor_reps)
            raw_prototypes = kmeans.cluster_centers_.astype(np.float32)

        # 4. Normalisation L2 stricte (optimise les produits scalaires cosinus ultérieurs)
        norms = np.linalg.norm(raw_prototypes, axis=1, keepdims=True)
        self._prototypes = raw_prototypes / np.maximum(norms, 1e-8)

        # 5. Persistance disque
        if cohort_path.exists():
            np.save(proto_file, self._prototypes)

        return self._prototypes

    # -------------------------------------------------------------------------
    # Distance clinique inter-cohortes (Min-Linkage Cosinus)
    # -------------------------------------------------------------------------
    def min_cosine_distance(self, other: "DrugCohort") -> float:
        """Calcule d_min(A, B) = min_{i,j} (1 - cos(mu_{A,i}, mu_{B,j})).
        
        Retourne un scalaire dans [0.0, 2.0] : 0.0 indique qu'au moins une sous-population
        clinique est alignée entre les deux molécules.
        """
        if self._prototypes is None:
            raise ValueError(f"Prototypes non calculés pour la cohorte {self.rxnorm_concept_id}.")
        if other._prototypes is None:
            raise ValueError(f"Prototypes non calculés pour la cohorte {other.rxnorm_concept_id}.")

        # Produit matriciel rapide (k_A, k_B)
        sim_matrix = np.dot(self._prototypes, other._prototypes.T)
        max_sim = float(np.max(sim_matrix))

        return float(np.clip(1.0 - max_sim, 0.0, 2.0))

    # -------------------------------------------------------------------------
    # Export pour le stage 1 de Rabit / FEMR
    # -------------------------------------------------------------------------
    def to_femr_prediction_times(self) -> pl.DataFrame:
        """Génère le format requis par FEMR : colonnes (patient_id, prediction_time)
        avec granularité à la minute (secondes et microsecondes tronquées à 00).
        """
        if self.target_patients is None:
            raise ValueError("target_patients non chargé.")

        return (
            self.target_patients
            .select([
                pl.col("person_id").alias("patient_id"),
                pl.col("t0").dt.truncate("1m").dt.strftime("%Y-%m-%d %H:%M:%S").alias("prediction_time"),
            ])
            .unique(subset=["patient_id"])
        )

    def __repr__(self) -> str:
        name_str = f" '{self.rxnorm_name}'" if self.rxnorm_name else ""
        n_pts = self.n_target
        proto_str = f", k={len(self._prototypes)}" if self._prototypes is not None else ""
        return f"<DrugCohort cid={self.rxnorm_concept_id}{name_str} (N={n_pts:,}{proto_str})>"