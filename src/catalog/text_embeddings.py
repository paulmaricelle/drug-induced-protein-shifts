# src/catalog/text_embeddings.py
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from tqdm import tqdm


def pa_array_from_set(s: set[int]) -> pa.Array:
    """Convertit un set Python d'identifiants en Array PyArrow int64 vectorisé."""
    return pa.array(list(s), type=pa.int64())


def extract_and_merge_text_embeddings(
    catalog_parquet_path: Path,
    feather_dir: Path,
    output_catalog_path: Path,
    output_pt_path: Path,
) -> pl.DataFrame:
    """Scanne les partitions Feather d'embeddings OMOP, extrait les vecteurs textuels (1024-d)

    et fusionne le tout avec le catalogue théorique.
    """
    print(f"1. Lecture du catalogue théorique : {catalog_parquet_path}...")
    catalog = pl.read_parquet(catalog_parquet_path)
    total_molecules = catalog.height
    print(f"   -> {total_molecules} principes actifs chargés.")

    target_ids = set(catalog["ingredient_concept_id"].to_list())
    extracted_embeddings: dict[int, np.ndarray] = {}

    # 2. Scan des partitions Feather
    feather_files = sorted(
        [f for f in feather_dir.glob("*.feather") if not f.name.startswith(".")]
    )
    if not feather_files:
        raise FileNotFoundError(
            f"Aucun fichier .feather trouvé dans {feather_dir}"
        )

    print(f"2. Scan de {len(feather_files)} partitions Feather dans {feather_dir}...")

    for file_path in tqdm(feather_files, desc="Scan des partitions"):
        if len(extracted_embeddings) == len(target_ids):
            print("\nTous les embeddings ont été trouvés ! Arrêt anticipé du scan.")
            break

        try:
            table = feather.read_table(
                file_path, columns=["concept_id", "dense_embedding"]
            )
            remaining_targets = target_ids - set(extracted_embeddings.keys())
            if not remaining_targets:
                break

            mask = pc.is_in(
                table["concept_id"],
                value_set=pa_array_from_set(remaining_targets),
            )
            filtered_table = table.filter(mask)

            if filtered_table.num_rows > 0:
                cids = filtered_table["concept_id"].to_pylist()
                embs = filtered_table["dense_embedding"].to_pylist()

                for cid, emb in zip(cids, embs):
                    extracted_embeddings[int(cid)] = np.array(
                        emb, dtype=np.float32
                    )

        except Exception as e:
            print(
                f"\n[Erreur] Problème lors de la lecture de {file_path.name} : {e}"
            )

    # 3. Bilan d'audit de la couverture
    found_ids = set(extracted_embeddings.keys())
    missing_ids = target_ids - found_ids

    print("\n" + "=" * 80)
    print("BILAN DE L'EXTRACTION DES EMBEDDINGS TEXTUELS :")
    print(f" - Molécules cibles : {len(target_ids)}")
    print(
        f" - Embeddings trouvés : {len(found_ids)} ({len(found_ids)/len(target_ids):.1%})"
    )
    print(f" - Molécules sans embedding textuel : {len(missing_ids)}")
    print("=" * 80)

    if missing_ids:
        missing_df = (
            catalog.filter(
                pl.col("ingredient_concept_id").is_in(list(missing_ids))
            )
            .select(["ingredient_concept_id", "ingredient_name", "n_targets"])
            .sort("ingredient_name")
        )
        print("\nExemple de molécules orphelines de texte :")
        print(missing_df.head(10))

    # 4. Sauvegarde PyTorch (.pt)
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch_embeddings = {
        cid: torch.from_numpy(arr) for cid, arr in extracted_embeddings.items()
    }
    torch.save(torch_embeddings, output_pt_path)
    print(f"\n-> Dictionnaire PyTorch sauvegardé : {output_pt_path}")

    # 5. Fusion et sauvegarde Parquet final
    emb_records = [
        {"ingredient_concept_id": cid, "text_embedding": arr.tolist()}
        for cid, arr in extracted_embeddings.items()
    ]
    emb_df = pl.DataFrame(emb_records).with_columns(
        pl.col("text_embedding").cast(pl.List(pl.Float32))
    )

    # Jointure avec le catalogue
    final_catalog = catalog.join(
        emb_df, on="ingredient_concept_id", how="left"
    )

    output_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    final_catalog.write_parquet(output_catalog_path)
    size_mb = output_catalog_path.stat().st_size / (1024 * 1024)

    dim_text = len(next(iter(extracted_embeddings.values()))) if extracted_embeddings else 0
    dim_ua = len(catalog["ua"][0]) if "ua" in catalog.columns else 0

    print(f"-> Catalogue enrichi final sauvegardé : {output_catalog_path}")
    print(f"   Shape : {final_catalog.shape} ({size_mb:.2f} Mo)")
    print(f"   Dimension biologique u_a : {dim_ua}")
    print(f"   Dimension textuelle     : {dim_text}")

    return final_catalog