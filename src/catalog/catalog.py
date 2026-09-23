# src/catalog/catalog.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal
import numpy as np
import polars as pl


@dataclass
class TargetEngagement:
    """Engagement pharmacologique sur une protéine cible p (Section 3.3)."""

    uniprot_id: str
    gene_symbol: str = ""
    kd_nm: float | None = None
    cmax_unbound_nm: float | None = None
    direction: int = -1  # -1 : inhibition/antagonisme, +1 : activation/agonisme
    potency_weight: float = 1.0


@dataclass
class DrugItem:
    """Représentation canonique d'un principe actif ou d'une association médicamenteuse."""

    drug_id: int  # OMOP ingredient_concept_id (ou ID synthétique négatif pour combo)
    name: str
    kind: str  # 'monotherapy', 'fixed_combination', 'de_facto_combination'

    # Ingrédients constitutifs
    ingredient_concept_ids: list[int] = field(default_factory=list)
    ingredient_names: list[str] = field(default_factory=list)

    # Classification ATC niveau 4 (ex: 'C10AA' pour Statines)
    atc4: str | None = None
    atc4_name: str | None = None

    # Codes OMOP descendants (Clinical Drug / Branded Drug dans drug_exposure)
    descendant_concept_ids: set[int] = field(default_factory=set)

    # Cibles biologiques et vecteur u_a (1590-d)
    targets: list[TargetEngagement] = field(default_factory=list)
    target_vector: np.ndarray | None = None

    # Embeddings textuels denses OMOP (1024-d)
    text_embedding: np.ndarray | None = None

    @property
    def is_combination(self) -> bool:
        return len(self.ingredient_concept_ids) > 1

    def get_vector(
        self, modality: Literal["bio", "text", "concat"] = "bio"
    ) -> np.ndarray | None:
        """Retourne la représentation vectorielle selon la modalité choisie."""
        if modality == "bio":
            return self.target_vector
        if modality == "text":
            return self.text_embedding
        if modality == "concat":
            if self.target_vector is None or self.text_embedding is None:
                return None
            return np.concatenate([self.target_vector, self.text_embedding])
        raise ValueError(f"Modalité inconnue : {modality}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["descendant_concept_ids"] = sorted(list(self.descendant_concept_ids))
        if self.target_vector is not None:
            data["target_vector"] = self.target_vector.tolist()
        if self.text_embedding is not None:
            data["text_embedding"] = self.text_embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrugItem:
        targets = [TargetEngagement(**t) for t in data.pop("targets", [])]
        descendants = set(data.pop("descendant_concept_ids", []))
        t_vec = data.pop("target_vector", None)
        txt_vec = data.pop("text_embedding", None)

        return cls(
            **data,
            targets=targets,
            descendant_concept_ids=descendants,
            target_vector=(
                np.array(t_vec, dtype=np.float32) if t_vec is not None else None
            ),
            text_embedding=(
                np.array(txt_vec, dtype=np.float32)
                if txt_vec is not None
                else None
            ),
        )


class DrugCatalog:
    """Catalogue unifié des médicaments : résout les prescriptions OMOP en O(1),

    indexe les familles ATC4 et fournit les vecteurs u_a et textuels.
    """

    def __init__(self, items: list[DrugItem] | None = None):
        self._items: dict[int, DrugItem] = {}
        self._pair_to_combo_id: dict[tuple[int, int], int] = {}
        self._descendant_to_drug_id: dict[int, int] = {}
        self._atc4_to_drug_ids: dict[str, set[int]] = {}

        if items:
            for item in items:
                self.add_item(item)

    def add_item(self, item: DrugItem) -> None:
        """Enregistre un item et met à jour les index inverses."""
        self._items[item.drug_id] = item

        # Index ATC4 pour la recherche rapide des comparateurs
        if item.atc4:
            code = item.atc4[:5].upper()
            self._atc4_to_drug_ids.setdefault(code, set()).add(item.drug_id)

        # Index des bi-thérapies par paire ordonnée d'ingrédients
        if item.is_combination and len(item.ingredient_concept_ids) == 2:
            pair = tuple(sorted(item.ingredient_concept_ids))
            self._pair_to_combo_id[pair] = item.drug_id

        # Index inverse : descendant_concept_id (code prescrit) -> drug_id
        for desc_id in item.descendant_concept_ids:
            self._descendant_to_drug_id[desc_id] = item.drug_id
        self._descendant_to_drug_id[item.drug_id] = item.drug_id

    def get(self, drug_id: int) -> DrugItem | None:
        return self._items.get(drug_id)

    def get_by_prescribed_concept_id(
        self, prescribed_concept_id: int
    ) -> DrugItem | None:
        """Résolution en temps constant O(1) depuis un code de prescription issu de drug_exposure."""
        drug_id = self._descendant_to_drug_id.get(prescribed_concept_id)
        if drug_id is not None:
            return self._items.get(drug_id)
        return None

    def get_atc4_family_ids(self, drug_id: int) -> list[int]:
        """Retourne tous les identifiants de molécules partageant la même classe ATC4 (exclut drug_id)."""
        item = self.get(drug_id)
        if item is None or not item.atc4:
            return []

        code = str(item.atc4)[:5].upper()
        family_members = self._atc4_to_drug_ids.get(code, set())
        return sorted([did for did in family_members if did != drug_id])

    def get_or_create_de_facto_combination(
        self, ing_id_a: int, ing_id_b: int
    ) -> DrugItem:
        """Récupère ou instancie à la volée une bi-thérapie prescrite conjointement (Section 3.3)."""
        pair = tuple(sorted([ing_id_a, ing_id_b]))
        if pair in self._pair_to_combo_id:
            return self._items[self._pair_to_combo_id[pair]]

        item_a = self.get(ing_id_a)
        item_b = self.get(ing_id_b)
        if not item_a or not item_b:
            raise ValueError(f"Ingrédients introuvables : {ing_id_a}, {ing_id_b}")

        import hashlib

        hash_digest = hashlib.md5(f"{pair[0]}_{pair[1]}".encode()).hexdigest()
        synth_id = int(hash_digest[:12], 16) % (10**12) + 9_000_000_000_000
        name = f"{item_a.name} + {item_b.name}"

        # u_{A+B} = u_A + u_B (Section 3.3)
        u_combo = None
        if item_a.target_vector is not None and item_b.target_vector is not None:
            u_combo = item_a.target_vector + item_b.target_vector

        # Embedding textuel combiné
        text_combo = None
        if item_a.text_embedding is not None and item_b.text_embedding is not None:
            text_combo = (item_a.text_embedding + item_b.text_embedding) / 2.0
            norm = np.linalg.norm(text_combo)
            if norm > 0:
                text_combo = text_combo / norm

        combo_item = DrugItem(
            drug_id=synth_id,
            name=name,
            kind="de_facto_combination",
            ingredient_concept_ids=list(pair),
            ingredient_names=[item_a.name, item_b.name],
            descendant_concept_ids=set(),
            targets=item_a.targets + item_b.targets,
            target_vector=u_combo,
            text_embedding=text_combo,
        )
        self.add_item(combo_item)
        return combo_item

    def register_de_facto_cohorts(self, cohorts_dir: str | Path) -> int:
        """Détecte et enregistre dans le catalogue les bi-thérapies de facto

        sauvegardées sur le disque (cohort_9*), en reconstituant leur vecteur u_a.
        """
        cohorts_path = Path(cohorts_dir)
        n_loaded = 0
        for meta_file in cohorts_path.glob("cohort_9*/metadata.json"):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)

                combo_id = int(meta["drug_id"])
                if combo_id in self._items:
                    continue

                ing_ids = meta.get("ingredient_concept_ids", [])
                if len(ing_ids) != 2:
                    continue

                item_a = self.get(ing_ids[0])
                item_b = self.get(ing_ids[1])

                u_combo = None
                if (
                    item_a
                    and item_b
                    and item_a.target_vector is not None
                    and item_b.target_vector is not None
                ):
                    u_combo = item_a.target_vector + item_b.target_vector

                text_combo = None
                if (
                    item_a
                    and item_b
                    and item_a.text_embedding is not None
                    and item_b.text_embedding is not None
                ):
                    text_combo = (
                        item_a.text_embedding + item_b.text_embedding
                    ) / 2.0
                    norm = np.linalg.norm(text_combo)
                    if norm > 0:
                        text_combo = text_combo / norm

                combo_item = DrugItem(
                    drug_id=combo_id,
                    name=meta.get(
                        "name",
                        f"{item_a.name if item_a else ing_ids[0]} + {item_b.name if item_b else ing_ids[1]} (De Facto)",
                    ),
                    kind="de_facto_combination",
                    ingredient_concept_ids=ing_ids,
                    ingredient_names=[
                        item_a.name if item_a else str(ing_ids[0]),
                        item_b.name if item_b else str(ing_ids[1]),
                    ],
                    atc4=None,
                    descendant_concept_ids=set(),
                    targets=(item_a.targets if item_a else [])
                    + (item_b.targets if item_b else []),
                    target_vector=u_combo,
                    text_embedding=text_combo,
                )
                self.add_item(combo_item)
                n_loaded += 1
            except Exception:
                continue

        if n_loaded > 0:
            print(
                f"-> {n_loaded:,} bi-thérapies de facto rattachées au"
                " DrugCatalog."
            )
        return n_loaded

    @classmethod
    def from_pipeline_artifacts(
        cls,
        final_catalog_parquet: Path,
        omop_vocab_dir: Path,
        mapping_parquet: Path | None = None,
        atc_parquet: Path | None = None,
    ) -> DrugCatalog:
        """Construit le catalogue complet en reliant drug_catalog_final.parquet,

        ingredient_to_prescriptions.parquet et ingredient_to_atc4.parquet.
        """
        print(f"Chargement du catalogue enrichi depuis {final_catalog_parquet.name}...")
        df_cat = pl.read_parquet(final_catalog_parquet)
        catalog = cls()

        root_dir = final_catalog_parquet.parents[2]
        atc_file = atc_parquet or (root_dir / "data" / "ingredient_to_atc4.parquet")

        # 1. Chargement de l'index ATC4
        atc_dict: dict[int, tuple[str, str]] = {}
        if atc_file.exists():
            print(f"Chargement des correspondances ATC4 depuis {atc_file.name}...")
            df_atc = pl.read_parquet(atc_file)
            for row in df_atc.iter_rows(named=True):
                ing_id = int(row["ingredient_id"])
                if ing_id not in atc_dict:
                    atc_dict[ing_id] = (str(row["atc4_code"]), str(row["atc4_name"]))
            print(f"  -> {len(atc_dict):,} ingrédients associés à une classe ATC4.")
        else:
            print(f"[Avertissement] Fichier ATC4 introuvable ({atc_file}). Wash-out comparateurs désactivé.")

        # 2. Instanciation des monothérapies
        for row in df_cat.iter_rows(named=True):
            cid = int(row["ingredient_concept_id"])
            name = str(row["ingredient_name"])

            if "targets_json" in row and row["targets_json"]:
                raw_targets = json.loads(row["targets_json"])
                targets = [TargetEngagement(**t) for t in raw_targets]
            else:
                targets = [
                    TargetEngagement(uniprot_id=uid)
                    for uid in row.get("target_uids", [])
                ]

            t_vec = (
                np.array(row["ua"], dtype=np.float32)
                if row.get("ua") is not None
                else None
            )
            txt_vec = (
                np.array(row["text_embedding"], dtype=np.float32)
                if row.get("text_embedding") is not None
                else None
            )

            atc_code, atc_name = atc_dict.get(cid, (None, None))

            item = DrugItem(
                drug_id=cid,
                name=name,
                kind="monotherapy",
                ingredient_concept_ids=[cid],
                ingredient_names=[name],
                atc4=atc_code,
                atc4_name=atc_name,
                descendant_concept_ids=set(),
                targets=targets,
                target_vector=t_vec,
                text_embedding=txt_vec,
            )
            catalog.add_item(item)

        print(f"  -> {len(catalog)} principes actifs enregistrés.")

        # 3. Rattachement direct des prescriptions via ingredient_to_prescriptions.parquet
        map_file = (
            mapping_parquet
            or (root_dir / "data" / "ingredient_to_prescriptions.parquet")
        )

        print(f"Liaison des formes cliniques via {map_file.name}...")
        map_df = pl.read_parquet(map_file)

        mono_mappings = map_df.filter(pl.col("is_monotherapy"))
        n_mono_mapped = 0
        for row in mono_mappings.iter_rows(named=True):
            ing_id = int(row["ingredient_id"])
            drug_id = int(row["drug_concept_id"])
            if ing_id in catalog._items:
                catalog._items[ing_id].descendant_concept_ids.add(drug_id)
                catalog._descendant_to_drug_id[drug_id] = ing_id
                n_mono_mapped += 1

        # 4. Combinaisons fixes (comprimés multi-ingrédients)
        combo_mappings = map_df.filter(~pl.col("is_monotherapy"))
        combos_grouped = combo_mappings.group_by("drug_concept_id").agg(
            pl.col("ingredient_id").unique().alias("ingredients")
        )

        fixed_combos: dict[tuple[int, int], list[int]] = {}
        for row in combos_grouped.iter_rows(named=True):
            ings = row["ingredients"]
            if len(ings) == 2:
                pair = tuple(sorted([int(ings[0]), int(ings[1])]))
                if pair[0] in catalog._items and pair[1] in catalog._items:
                    fixed_combos.setdefault(pair, []).append(
                        int(row["drug_concept_id"])
                    )

        for pair, desc_ids in fixed_combos.items():
            item_a = catalog.get(pair[0])
            item_b = catalog.get(pair[1])
            if not item_a or not item_b:
                continue

            synth_id = -int(abs(hash(pair)) % (10**9))
            combo_name = f"{item_a.name} / {item_b.name} (Fixed Dose)"

            u_combo = (
                (item_a.target_vector + item_b.target_vector)
                if (
                    item_a.target_vector is not None
                    and item_b.target_vector is not None
                )
                else None
            )

            text_combo = None
            if (
                item_a.text_embedding is not None
                and item_b.text_embedding is not None
            ):
                text_combo = (item_a.text_embedding + item_b.text_embedding) / 2.0
                norm = np.linalg.norm(text_combo)
                if norm > 0:
                    text_combo = text_combo / norm

            fixed_item = DrugItem(
                drug_id=synth_id,
                name=combo_name,
                kind="fixed_combination",
                ingredient_concept_ids=list(pair),
                ingredient_names=[item_a.name, item_b.name],
                atc4=None,
                descendant_concept_ids=set(desc_ids),
                targets=item_a.targets + item_b.targets,
                target_vector=u_combo,
                text_embedding=text_combo,
            )
            catalog.add_item(fixed_item)

        print(f"  -> {n_mono_mapped:,} prescriptions reliées à des monothérapies.")
        print(f"  -> {len(fixed_combos)} bi-thérapies fixes instanciées.")
        print(f"  -> Total codes résolubles en O(1) : {len(catalog._descendant_to_drug_id):,}")

        return catalog

    def save(self, filepath: str | Path) -> None:
        """Sauvegarde sérialisée en JSONL."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in self._items.values():
                f.write(json.dumps(item.to_dict()) + "\n")
        print(f"-> DrugCatalog sauvegardé : {filepath} ({len(self)} items)")

    @classmethod
    def load(cls, filepath: str | Path) -> DrugCatalog:
        """Recharge le catalogue sérialisé en reconstruisant les index."""
        catalog = cls()
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = DrugItem.from_dict(json.loads(line))
                    catalog.add_item(item)
        return catalog

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())