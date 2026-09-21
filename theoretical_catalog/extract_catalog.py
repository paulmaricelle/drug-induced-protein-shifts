from pathlib import Path
import polars as pl
import pyarrow.feather as feather
import pyarrow.compute as pc
import torch
import numpy as np
from tqdm import tqdm

# Chemins sur le serveur
PARQUET_INPUT_PATH = Path("data/clean_drug_catalog.parquet")
FEATHER_DIR = Path("/remote/shared/collab/omop-text-embeddings/unified/v1/RxNorm/")

OUTPUT_PT_PATH = Path("data/omop_dense_embeddings.pt")
OUTPUT_PARQUET_PATH = Path("data/embedded_drug_catalog.parquet")


def extract_embeddings_for_clean_catalog():
    print(f"1. Lecture du catalogue épuré : {PARQUET_INPUT_PATH}...")
    catalog = pl.read_parquet(PARQUET_INPUT_PATH)
    
    total_molecules = catalog.height
    print(f"Catalogue chargé : {total_molecules} molécules uniques.")

    target_ids = set(catalog["rxnorm_concept_id"].to_list())

    # Dictionnaire mémoire : concept_id -> np.ndarray ou torch.Tensor
    extracted_embeddings = {}

    # ---------------------------------------------------------
    # Scan des partitions Feather
    # ---------------------------------------------------------
    feather_files = sorted([
        f for f in FEATHER_DIR.glob("*.feather")
        if not f.name.startswith(".")
    ])

    if not feather_files:
        raise FileNotFoundError(
            f"Aucun fichier .feather trouvé dans {FEATHER_DIR}"
        )

    print(f"2. Scan de {len(feather_files)} partitions Feather dans {FEATHER_DIR}...")
    
    for file_path in tqdm(feather_files, desc="Scan des partitions"):
        # Si tous les IDs cibles sont déjà trouvés, on peut arrêter le scan en avance
        if len(extracted_embeddings) == len(target_ids):
            print("\nTous les embeddings ont été trouvés ! Arrêt anticipé du scan.")
            break

        try:
            # Lecture PyArrow directe (très rapide)
            table = feather.read_table(
                file_path, 
                columns=["concept_id", "dense_embedding"]
            )
            
            # Filtrage vectorisé sur les IDs cibles non encore trouvés
            remaining_targets = target_ids - set(extracted_embeddings.keys())
            mask = pc.is_in(table["concept_id"], value_set=pa_array_from_set(remaining_targets))
            filtered_table = table.filter(mask)

            if filtered_table.num_rows > 0:
                cids = filtered_table["concept_id"].to_pylist()
                embs = filtered_table["dense_embedding"].to_pylist()

                for cid, emb in zip(cids, embs):
                    extracted_embeddings[int(cid)] = np.array(emb, dtype=np.float32)

        except Exception as e:
            print(f"\n[Erreur] Problème lors de la lecture de {file_path.name} : {e}")

    # ---------------------------------------------------------
    # Bilan et audit des molécules manquantes
    # ---------------------------------------------------------
    found_ids = set(extracted_embeddings.keys())
    missing_ids = target_ids - found_ids

    print("\n" + "=" * 80)
    print(f"BILAN DE L'EXTRACTION :")
    print(f" - Molécules cibles : {len(target_ids)}")
    print(f" - Embeddings trouvés : {len(found_ids)} ({len(found_ids)/len(target_ids):.1%})")
    print(f" - Molécules orphelines (sans embedding) : {len(missing_ids)}")
    print("=" * 80)

    if missing_ids:
        missing_df = (
            catalog.filter(pl.col("rxnorm_concept_id").is_in(list(missing_ids)))
            .select(["rxnorm_concept_id", "rxnorm_name", "atc3_primary", "atc1_code"])
            .sort("rxnorm_name")
        )
        print("\nExemple de molécules sans embedding (seront exclues du catalogue final) :")
        print(missing_df.head(15))
    else:
        print("\nCouverture parfaite : 100 % des molécules disposent d'un embedding !")

    # ---------------------------------------------------------
    # Sauvegarde 1 : Dictionnaire de tenseurs PyTorch (.pt)
    # ---------------------------------------------------------
    OUTPUT_PT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch_embeddings = {
        cid: torch.from_numpy(arr) for cid, arr in extracted_embeddings.items()
    }
    torch.save(torch_embeddings, OUTPUT_PT_PATH)
    print(f"\n[1/2] Dictionnaire PyTorch sauvegardé : {OUTPUT_PT_PATH}")

    # ---------------------------------------------------------
    # Sauvegarde 2 : Catalogue final enrichi (.parquet)
    # ---------------------------------------------------------
    # On convertit le dictionnaire en DataFrame Polars pour faire une jointure interne propre
    emb_records = [
        {"rxnorm_concept_id": cid, "embedding": arr.tolist()} 
        for cid, arr in extracted_embeddings.items()
    ]
    emb_df = pl.DataFrame(emb_records)

    final_embedded_catalog = catalog.join(emb_df, on="rxnorm_concept_id", how="inner")
    
    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_embedded_catalog.write_parquet(OUTPUT_PARQUET_PATH)
    print(f"[2/2] Catalogue enrichi sauvegardé : {OUTPUT_PARQUET_PATH}")
    print(f"Nombre de lignes prêtes pour la phase EHR : {final_embedded_catalog.height}")


def pa_array_from_set(s):
    """Helper pour convertir un set Python en Array PyArrow pour pc.is_in."""
    import pyarrow as pa
    return pa.array(list(s), type=pa.int64())


if __name__ == "__main__":
    extract_embeddings_for_clean_catalog()