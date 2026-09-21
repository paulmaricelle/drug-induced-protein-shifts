from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

from data.test_triplets_50 import BENCHMARK_TRIPLETS

FEATHER_DIR = Path("/remote/shared/collab/omop-text-embeddings/unified/v1/RxNorm/")
OUTPUT_BENCHMARK_PT = Path("data/benchmark_embeddings_50.pt")


def extract_directly_from_feather():
    # 1. Molécules uniques recherchées (en minuscules)
    needed_names = set()
    for t in BENCHMARK_TRIPLETS:
        needed_names.add(t["A"].lower().strip())
        needed_names.add(t["B"].lower().strip())
        needed_names.add(t["C"].lower().strip())

    print(f"Recherche de {len(needed_names)} molécules directement dans les fichiers Feather...")

    feather_files = list(FEATHER_DIR.glob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"Aucun fichier feather trouvé dans {FEATHER_DIR}")

    extracted_embeddings = {}  # concept_id -> tensor
    name_to_cid = {}           # name_lower -> concept_id

    # 2. Scan direct des Feather (on ne charge que 3 colonnes pour aller vite)
    for file_path in tqdm(feather_files, desc="Scan des partitions"):
        df = pd.read_feather(
            file_path,
            columns=["concept_id", "concept_name", "dense_embedding"]
        )

        # Filtre sur les noms exacts en minuscules
        df["concept_name_lower"] = df["concept_name"].str.lower().str.strip()
        matched = df[df["concept_name_lower"].isin(needed_names)]

        for _, row in matched.iterrows():
            name = row["concept_name_lower"]
            cid = int(row["concept_id"])

            if name not in name_to_cid:
                name_to_cid[name] = cid
                emb = np.array(row["dense_embedding"], dtype=np.float32)
                extracted_embeddings[cid] = torch.from_numpy(emb)

        # Arrêt précoce si toutes les molécules ont été trouvées
        if len(name_to_cid) == len(needed_names):
            print("\nToutes les molécules ont été trouvées !")
            break

    # 3. Bilan
    found = set(name_to_cid.keys())
    missing = needed_names - found

    print(f"\nExtraction terminée : {len(found)} / {len(needed_names)} molécules trouvées.")
    if missing:
        print(f"Molécules introuvables : {sorted(list(missing))}")

    # 4. Sauvegarde
    torch.save({
        "embeddings": extracted_embeddings,
        "name_to_cid": name_to_cid,
        "triplets": BENCHMARK_TRIPLETS
    }, OUTPUT_BENCHMARK_PT)

    print(f"Artefact sauvegardé dans : {OUTPUT_BENCHMARK_PT}")


if __name__ == "__main__":
    extract_directly_from_feather()