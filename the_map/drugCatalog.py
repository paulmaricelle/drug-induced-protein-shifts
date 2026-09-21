from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Union
import polars as pl
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


@dataclass
class DrugItem:
    concept_id: int
    name: str
    atc_codes: List[str] = field(default_factory=list)
    atc_code: Optional[str] = None
    atc3: Optional[str] = None
    atc4: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    prescription_codes: Set[int] = field(default_factory=set)
    n_cohort_patients: int = 0
    is_viable: bool = False

    def __post_init__(self):
        # Normalisation si atc_codes est une string ou None
        if isinstance(self.atc_codes, str):
            self.atc_codes = [self.atc_codes]
        elif self.atc_codes is None:
            self.atc_codes = []

        # Synchronisation singulier / pluriel
        if not self.atc_code and self.atc_codes:
            self.atc_code = self.atc_codes[0]
        elif self.atc_code and not self.atc_codes:
            self.atc_codes = [self.atc_code]

        # Dérivation automatique des niveaux ATC
        if self.atc_code:
            code = self.atc_code.strip()
            if not self.atc3 and len(code) >= 4:
                self.atc3 = code[:4]
            if not self.atc4 and len(code) >= 5:
                self.atc4 = code[:5]
    
class DrugCatalog:
    """
    Interface centralisée pour 'The Map'.
    Gère l'alignement entre les métadonnées cliniques, les tenseurs d'embeddings X,
    les filtres ATC et les correspondances de formulations OMOP CDM.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        prescription_map: Optional[Dict[int, Set[int]]] = None,
    ):
        # Tri canonique sur les identifiants pour garantir un ordre d'indexation stable
        self._df = df.sort("rxnorm_concept_id")

        # Index de recherche rapide O(1)
        self._id_to_idx: Dict[int, int] = {
            int(cid): idx for idx, cid in enumerate(self._df["rxnorm_concept_id"].to_list())
        }
        self._name_to_idx: Dict[str, int] = {
            str(name).lower().strip(): idx
            for idx, name in enumerate(self._df["rxnorm_name"].to_list())
        }

        # Matrice dense de représentation X (M, D)
        if "embedding" in self._df.columns:
            emb_list = self._df["embedding"].to_list()
            self._tensor_x = torch.tensor(emb_list, dtype=torch.float32)
        else:
            self._tensor_x = torch.empty((len(self._df), 0), dtype=torch.float32)

        # Mappings ontologiques OMOP
        # Ingrédient -> {drug_concept_id_1, drug_concept_id_2, ...}
        self._id_to_prescriptions: Dict[int, Set[int]] = prescription_map or {
            cid: set() for cid in self._id_to_idx.keys()
        }
        
        # Table inversée pour lookup O(1) sur événements : drug_concept_id -> Ingrédient
        self._prescription_to_id: Dict[int, int] = {}
        for ing_id, p_codes in self._id_to_prescriptions.items():
            for p_code in p_codes:
                self._prescription_to_id[p_code] = ing_id

    # --- Constructeurs d'usine ---

    @classmethod
    def from_parquet(
        cls,
        catalog_path: Union[str, Path] = "data/embedded_drug_catalog.parquet",
        mapping_path: Optional[Union[str, Path]] = "data/ingredient_to_prescriptions.parquet",
    ) -> DrugCatalog:
        """Instancie le catalogue et charge optionnellement le mapping de prescriptions."""
        c_path = Path(catalog_path)
        if not c_path.exists():
            raise FileNotFoundError(f"Catalogue introuvable à : {c_path}")

        df = pl.read_parquet(c_path)
        instance = cls(df)

        if mapping_path is not None:
            m_path = Path(mapping_path)
            if m_path.exists():
                instance.load_prescription_mapping(m_path)
            else:
                print(f"Note: Aucun mapping OMOP trouvé à {m_path}. Utilisez .load_prescription_mapping().")

        return instance

    # --- Intégration du mapping ontologique OMOP ---

    def load_prescription_mapping(
        self, mapping_path: Union[str, Path] = "data/ingredient_to_prescriptions.parquet"
    ) -> None:
        """
        Charge la table précalculée ingrédient <-> codes cliniques OMOP.
        Seules les monothérapies pures (is_monotherapy == True) alimentent 
        l'index de recherche inversé O(1) et prescription_codes.
        """
        path = Path(mapping_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de mapping introuvable : {path}")

        df_map = pl.read_parquet(path)

        id_to_prescriptions: Dict[int, Set[int]] = {cid: set() for cid in self._id_to_idx.keys()}
        prescription_to_id: Dict[int, int] = {}

        # Si le flag is_monotherapy est présent, on filtre strictement pour le catalogue
        if "is_monotherapy" in df_map.columns:
            mono_df = df_map.filter(pl.col("is_monotherapy"))
        else:
            mono_df = df_map

        for row in mono_df.iter_rows(named=True):
            ing_id = int(row["ingredient_id"])
            drug_id = int(row["drug_concept_id"])

            if ing_id in id_to_prescriptions:
                id_to_prescriptions[ing_id].add(drug_id)
                prescription_to_id[drug_id] = ing_id

        self._id_to_prescriptions = id_to_prescriptions
        self._prescription_to_id = prescription_to_id
        
        n_mapped = sum(1 for s in self._id_to_prescriptions.values() if len(s) > 0)
        print(f"Mapping OMOP lié : {len(prescription_to_id):,} codes de monothérapies résolus pour {n_mapped:,} molécules.")

    def get_prescription_codes(self, identifier: Union[int, str]) -> Set[int]:
        """Retourne l'ensemble des drug_concept_id OMOP pour une molécule donnée."""
        cid = self._resolve_to_id(identifier)
        return self._id_to_prescriptions.get(cid, set())

    def match_prescription(self, drug_concept_id: int) -> Optional[int]:
        """
        Lookup O(1) : retourne le concept_id de l'ingrédient si drug_concept_id 
        est une monothérapie connue, sinon None.
        """
        return self._prescription_to_id.get(int(drug_concept_id))

    def get_atc4_family_ids(self, identifier: Union[int, str]) -> Set[int]:
        """
        Retourne l'ensemble des concept_id (ingrédients) du catalogue appartenant
        à la même classe ATC4 que la molécule cible (pour le washout de classe).
        """
        item = self.get(identifier)
        atc4_prefixes = {str(c)[:5] for c in item.atc_codes if c and len(str(c)) >= 5}
        
        if not atc4_prefixes and item.atc3:
            atc4_prefixes = {str(item.atc3)[:4]}

        if not atc4_prefixes:
            return {item.concept_id}

        family_ids = set()
        for candidate in self:
            if any(any(str(c).startswith(pfx) for pfx in atc4_prefixes) for c in candidate.atc_codes if c):
                family_ids.add(candidate.concept_id)

        return family_ids

    def load_cohort_manifest(
        self,
        manifest_path: str | Path = "data/cohorts/manifest.parquet",
        min_patients: int = 10,
    ) -> None:
        """Lie les effectifs réels et active les molécules viables selon le seuil choisi."""
        manifest_df = pl.read_parquet(manifest_path)
        
        # Dictionnaire {concept_id: n_patients}
        counts = dict(
            zip(
                manifest_df["rxnorm_concept_id"].to_list(),
                manifest_df["n_final_target"].to_list(),
            )
        )

        for cid, item in self._items.items():
            n = counts.get(cid, 0)
            item.n_cohort_patients = n
            # Seuil dynamique appliqué ici au runtime
            item.is_viable = (n >= min_patients)

    @property
    def viable_concept_ids(self) -> list[int]:
        """Retourne les molécules sélectionnées pour le seuil actuel."""
        return [cid for cid, item in self._items.items() if item.is_viable]

    def get_viable_tensor_x(self) -> np.ndarray:
        """
        Retourne le tenseur X d'embeddings (N_viable, d) aligné 
        strictement sur l'ordre de `viable_concept_ids`.
        """
        return np.stack([self._items[cid].embedding for cid in self.viable_concept_ids])

    # --- Propriétés et accesseurs matriciels ---

    @property
    def tensor_x(self) -> torch.Tensor:
        """Retourne la matrice dense PyTorch des représentations X (M, D)."""
        return self._tensor_x

    @property
    def concept_ids(self) -> List[int]:
        """Liste ordonnée des concept_ids alignée avec tensor_x."""
        return self._df["rxnorm_concept_id"].to_list()

    @property
    def names(self) -> List[str]:
        """Liste ordonnée des noms de molécules alignée avec tensor_x."""
        return self._df["rxnorm_name"].to_list()

    # --- Requêtes unitaires et filtrage ---

    def _resolve_to_id(self, identifier: Union[int, str]) -> int:
        if isinstance(identifier, int):
            if identifier not in self._id_to_idx:
                raise KeyError(f"Identifiant {identifier} non répertorié dans le catalogue.")
            return identifier
        elif isinstance(identifier, str):
            clean_name = identifier.lower().strip()
            idx = self._name_to_idx.get(clean_name)
            if idx is None:
                raise KeyError(f"Molécule '{identifier}' non répertoriée dans le catalogue.")
            return self._df["rxnorm_concept_id"][idx]
        raise TypeError("L'identifiant doit être un int ou une chaîne de caractères.")

    def get(self, identifier: Union[int, str]) -> DrugItem:
        """Récupère une molécule sous forme de DrugItem immuable."""
        cid = self._resolve_to_id(identifier)
        idx = self._id_to_idx[cid]
        row = self._df.row(idx, named=True)

        return DrugItem(
            concept_id=cid,
            name=row["rxnorm_name"],
            atc3=row.get("atc3_primary", row.get("atc3", "")),
            atc_codes=row.get("atc_codes", []),
            embedding=self._tensor_x[idx],
            prescription_codes=self._id_to_prescriptions.get(cid, set()),
        )

    def filter_by_atc(self, atc_prefix: str) -> DrugCatalog:
        """
        Retourne un sous-catalogue filtré selon un code ou préfixe ATC (ex. 'C10' pour les statines).
        Maintient l'alignement des tenseurs et des mappings.
        """
        prefix = atc_prefix.upper().strip()
        filtered_df = self._df.filter(
            pl.col("atc_codes").list.eval(pl.element().str.starts_with(prefix)).list.any()
        )
        
        # On ne transmet que le sous-ensemble de mapping correspondant
        sub_cids = set(filtered_df["rxnorm_concept_id"].to_list())
        sub_prescription_map = {
            cid: self._id_to_prescriptions.get(cid, set()) for cid in sub_cids
        }
        return DrugCatalog(filtered_df, prescription_map=sub_prescription_map)

    def to_polars(self) -> pl.DataFrame:
        """Retourne la table Polars sous-jacente."""
        return self._df.clone()

    # --- Méthodes magiques Python ---

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, identifier: Union[int, str]) -> DrugItem:
        return self.get(identifier)

    def __contains__(self, identifier: Union[int, str]) -> bool:
        try:
            self._resolve_to_id(identifier)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[DrugItem]:
        for cid in self.concept_ids:
            yield self.get(cid)

    def __repr__(self) -> str:
        mapped_count = sum(1 for s in self._id_to_prescriptions.values() if len(s) > 0)
        return (
            f"DrugCatalog(nb_molecules={len(self):,}, "
            f"dim_embedding={self._tensor_x.shape[1] if self._tensor_x.numel() > 0 else 0}, "
            f"mapped_omop={mapped_count:,}/{len(self):,})"
        )