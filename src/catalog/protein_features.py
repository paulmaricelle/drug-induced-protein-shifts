# src/catalog/protein_features.py
import gzip
from pathlib import Path
import re
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.decomposition import TruncatedSVD

def extract_targets_from_reference_proteome(
    chembl_targets_df: pl.DataFrame,
    ref_proteome_gz: Path,
    output_fasta_path: Path,
) -> tuple[int, int]:
    """Extrait les séquences des cibles directement depuis le protéome humain de référence (14 Mo)."""
    output_fasta_path.parent.mkdir(parents=True, exist_ok=True)

    # Récupérer l'ensemble des cibles UniProt
    target_ids = set(
        chembl_targets_df.filter(pl.col("uniprot_id").is_not_null())
        .select("uniprot_id")
        .unique()["uniprot_id"]
        .to_list()
    )
    # Nettoyer les suffixes d'isoformes (P12345-1 -> P12345)
    clean_target_ids = {uid.split("-")[0].strip() for uid in target_ids if uid}

    print(f"Recherche locale de {len(clean_target_ids)} cibles dans {ref_proteome_gz.name}...")

    extracted_fastas = {}
    current_header = None
    current_acc = None
    current_seq = []

    with gzip.open(ref_proteome_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                # Sauvegarder la séquence précédente si elle correspondait
                if current_acc and current_acc in clean_target_ids:
                    extracted_fastas[current_acc] = f"{current_header}\n{''.join(current_seq)}\n"

                current_header = line.strip()
                current_seq = []
                # Extraire l'accession UniProt (ex: >sp|P08908|...)
                match = re.search(r"\|([A-Z0-9]+)\|", current_header)
                current_acc = match.group(1) if match else current_header.split()[0].replace(">", "")
            else:
                current_seq.append(line.strip())

        # Dernier enregistrement
        if current_acc and current_acc in clean_target_ids:
            extracted_fastas[current_acc] = f"{current_header}\n{''.join(current_seq)}\n"

    # Écriture du fichier final
    with open(output_fasta_path, "w", encoding="utf-8") as f:
        for fasta in extracted_fastas.values():
            f.write(fasta)

    found = len(extracted_fastas)
    total = len(clean_target_ids)
    size_kb = output_fasta_path.stat().st_size / 1024

    print(f"-> Succès : {found}/{total} séquences extraites localement.")
    print(f"-> Fichier généré : {output_fasta_path} ({size_kb:.1f} Ko)")
    
    missing = clean_target_ids - set(extracted_fastas.keys())
    if missing:
        print(f"-> {len(missing)} identifiants non présents dans le protéome de référence (ex: {list(missing)[:5]})")

    return found, total
def read_fasta_file(fasta_path: Path) -> dict[str, str]:
    """Lit un fichier FASTA et retourne un dictionnaire {uniprot_id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                # Parse le header Swiss-Prot (>sp|P12345|... ou >P12345)
                match = re.search(r"\|([A-Z0-9]+)\|", line)
                current_id = (
                    match.group(1) if match else line.split()[0].replace(">", "")
                )
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            sequences[current_id] = "".join(current_seq)

    return sequences


def compute_esm2_embeddings(
    fasta_path: Path,
    output_parquet: Path,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    max_length: int = 1024,
) -> pl.DataFrame:
    """Encode les séquences protéiques avec ESM-2 650M en demi-précision (1280-d)."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement de {model_name} sur {device}...")

    # Choix automatique du type flottant (bfloat16 si supporté, sinon float16)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    seq_dict = read_fasta_file(fasta_path)
    uids = list(seq_dict.keys())
    print(
        f"Génération des embeddings pour {len(uids)} cibles (batch_size={batch_size}, max_len={max_length})..."
    )

    all_embeddings = []

    with torch.inference_mode():
        for i in range(0, len(uids), batch_size):
            batch_uids = uids[i : i + batch_size]
            # Tronquage à max_length si une protéine dépasse
            batch_seqs = [seq_dict[uid][:max_length] for uid in batch_uids]

            tokens = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**tokens)

            # Mean-pooling en excluant les tokens de padding et les tokens spéciaux
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            token_embeddings = (
                outputs.last_hidden_state
            )  # (B, Seq_len, 1280) in fp16/bf16

            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            mean_embeddings = sum_embeddings / torch.clamp(
                attention_mask.sum(dim=1), min=1e-9
            )

            # Conversion en float32 pour le stockage
            all_embeddings.append(mean_embeddings.to(torch.float32).cpu().numpy())

            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(uids):
                done = min(i + batch_size, len(uids))
                print(f"  -> {done}/{len(uids)} séquences encodées ({done/len(uids)*100:.1f}%)")

    emb_matrix = np.vstack(all_embeddings)  # shape: (798, 1280)

    # Sauvegarde Parquet : list de float32 par ligne
    df = pl.DataFrame(
        {
            "uniprot_id": uids,
            "esm2_embedding": [
                emb_matrix[idx].tolist() for idx in range(len(uids))
            ],
        }
    )

    df.write_parquet(output_parquet)
    size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"\n-> Sauvegardé : {output_parquet} ({size_mb:.2f} Mo, shape: {df.shape})")
    return df

def compute_reactome_embeddings(
    uniprot_ids: list[str],
    reactome_path: Path,
    output_parquet: Path,
    n_components: int = 128,
) -> pl.DataFrame:
    """Construit la matrice binaire Protéines x Voies Reactome et la projette à 128-d via SVD."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"Traitement des voies Reactome depuis {reactome_path.name}...")

    # Le fichier est un TSV sans en-tête :
    # Col 0: UniProt ID, Col 1: Reactome ID, Col 2: URL, Col 3: Nom de la voie, Col 5: Espèce
    reactome_df = pl.read_csv(
        reactome_path,
        separator="\t",
        has_header=False,
        new_columns=["uniprot_id", "pathway_id", "url", "pathway_name", "evidence", "species"],
    ).filter(
        (pl.col("species") == "Homo sapiens") & (pl.col("uniprot_id").is_in(uniprot_ids))
    )

    # Récupérer les index uniques
    target_uids = sorted(list(set(uniprot_ids)))
    pathways = sorted(reactome_df["pathway_id"].unique().to_list())

    uid_to_idx = {uid: i for i, uid in enumerate(target_uids)}
    path_to_idx = {p: j for j, p in enumerate(pathways)}

    print(f"  -> {len(target_uids)} cibles croisées avec {len(pathways)} voies Reactome humaines distinctes.")

    # Construction de la matrice binaire creuse
    mat = np.zeros((len(target_uids), len(pathways)), dtype=np.float32)
    for row in reactome_df.iter_rows(named=True):
        i = uid_to_idx[row["uniprot_id"]]
        j = path_to_idx[row["pathway_id"]]
        mat[i, j] = 1.0

    # Projection SVD à 128 dimensions
    n_comp = min(n_components, mat.shape[1] - 1, mat.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(mat)  # shape: (n_targets, n_comp)

    # Compléter par des zéros si n_comp < n_components
    if embeddings.shape[1] < n_components:
        pad = np.zeros((len(target_uids), n_components - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.hstack([embeddings, pad])

    df = pl.DataFrame({
        "uniprot_id": target_uids,
        "reactome_embedding": [embeddings[i].tolist() for i in range(len(target_uids))],
    })

    df.write_parquet(output_parquet)
    print(f"-> Embeddings Reactome sauvegardés : {output_parquet} (shape: {df.shape})")
    return df

def compute_string_embeddings(
    uniprot_ids: list[str],
    links_path: Path,
    aliases_path: Path,
    output_parquet: Path,
    n_components: int = 128,
    min_score: int = 400,
) -> pl.DataFrame:
    """Construit l'interactome humain global STRING v12, applique une SVD 128-d

    et extrait les profils de nos cibles.
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    target_uids = sorted(list(set(uniprot_ids)))
    target_set = set(target_uids)

    print(
        f"1. Résolution des identifiants STRING <-> UniProt depuis {aliases_path.name}..."
    )
    # Le fichier aliases contient : #string_protein_id, alias, source
    aliases_df = (
        pl.read_csv(
            aliases_path,
            separator="\t",
            has_header=True,
            new_columns=["string_id", "alias", "source"],
        )
        .filter(pl.col("alias").is_in(target_set))
        .select(["string_id", "alias"])
        .unique(subset=["alias"])
    )

    uniprot_to_string = dict(
        zip(aliases_df["alias"].to_list(), aliases_df["string_id"].to_list())
    )
    print(
        f"  -> {len(uniprot_to_string)}/{len(target_uids)} cibles résolues dans STRING."
    )

    print(
        f"2. Chargement du réseau d'interactions depuis {links_path.name} (seuil={min_score})..."
    )
    # Le fichier links est séparé par des espaces : protein1, protein2, combined_score
    links_df = pl.read_csv(
        links_path,
        separator=" ",
        has_header=True,
    ).filter(pl.col("combined_score") >= min_score)

    # Indexer tous les nœuds du graphe global
    all_nodes = sorted(
        list(
            set(links_df["protein1"].unique().to_list())
            | set(links_df["protein2"].unique().to_list())
        )
    )
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    print(
        f"  -> Graphe STRING construit : {n_nodes} protéines, {len(links_df)} interactions."
    )

    # Construction de la matrice d'adjacence creuse pondérée
    row_idx = [node_to_idx[p] for p in links_df["protein1"].to_list()]
    col_idx = [node_to_idx[p] for p in links_df["protein2"].to_list()]
    weights = (links_df["combined_score"] / 1000.0).to_numpy().astype(np.float32)

    adj = csr_matrix((weights, (row_idx, col_idx)), shape=(n_nodes, n_nodes))

    print(
        f"3. Réduction spectrale SVD ({n_components} composantes) sur le graphe global..."
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    node_embeddings = svd.fit_transform(adj)  # shape: (n_nodes, 128)

    # 4. Mapper les embeddings vers nos cibles UniProt
    target_embeddings = []
    for uid in target_uids:
        string_id = uniprot_to_string.get(uid)
        if string_id and string_id in node_to_idx:
            idx = node_to_idx[string_id]
            target_embeddings.append(node_embeddings[idx].tolist())
        else:
            # Imputation neutre : vecteur nul si non présent dans STRING
            target_embeddings.append([0.0] * n_components)

    df = pl.DataFrame(
        {
            "uniprot_id": target_uids,
            "string_embedding": target_embeddings,
        }
    )

    df.write_parquet(output_parquet)
    size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(
        f"-> Embeddings STRING sauvegardés : {output_parquet} ({size_mb:.2f} Mo, shape: {df.shape})"
    )
    return df

def compute_gtex_embeddings(
    uniprot_ids: list[str],
    gtex_gct_gz: Path,
    protein_vocab_feather: Path,
    output_parquet: Path,
) -> pl.DataFrame:
    """Extrait les profils d'expression tissulaire GTEx v8 (54 tissus, log(1+TPM))

    pour les cibles UniProt.
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    target_uids = sorted(list(set(uniprot_ids)))

    print(
        f"1. Résolution UniProt -> Symboles de gènes via {protein_vocab_feather.name}..."
    )
    # Mapping UniProt (concept_code) -> Symbole de gène (concept_name)
    vocab_df = (
        pl.read_ipc(protein_vocab_feather)
        .filter(pl.col("concept_code").is_in(target_uids))
        .select(["concept_code", "concept_name"])
        .unique(subset=["concept_code"])
    )

    uniprot_to_gene = dict(
        zip(
            vocab_df["concept_code"].to_list(),
            vocab_df["concept_name"].str.to_uppercase().to_list(),
        )
    )
    gene_to_uniprot = {g: u for u, g in uniprot_to_gene.items()}
    print(
        f"  -> {len(uniprot_to_gene)}/{len(target_uids)} cibles associées à un symbole de gène."
    )

    print(
        f"2. Lecture sélective des profils tissulaires depuis {gtex_gct_gz.name}..."
    )
    # Le format GCT a 2 lignes de métadonnées au début, puis l'en-tête
    # Colonnes : Name (ENSG...), Description (Gene Symbol), puis 54 colonnes de tissus
    gene_profiles = {}

    with gzip.open(gtex_gct_gz, "rt", encoding="utf-8") as f:
        # Sauter les deux lignes d'en-tête GCT (#1.2 et dimensions)
        f.readline()
        f.readline()
        header = f.readline().strip().split("\t")
        tissue_names = header[2:]  # 54 tissus
        n_tissues = len(tissue_names)

        for line in f:
            parts = line.strip().split("\t")
            gene_symbol = parts[1].strip().upper()
            if gene_symbol in gene_to_uniprot:
                # log(1 + TPM) pour stabiliser la variance
                tpm_vals = [np.log1p(float(val)) for val in parts[2:]]
                gene_profiles[gene_symbol] = tpm_vals

    print(
        f"  -> {len(gene_profiles)}/{len(target_uids)} cibles trouvées dans les 54 tissus GTEx."
    )

    # Construction des vecteurs finaux pour chaque UniProt ID
    embeddings = []
    for uid in target_uids:
        gene = uniprot_to_gene.get(uid)
        if gene and gene in gene_profiles:
            embeddings.append(gene_profiles[gene])
        else:
            # Imputation neutre : vecteur nul si non exprimé / non répertorié
            embeddings.append([0.0] * n_tissues)

    df = pl.DataFrame(
        {
            "uniprot_id": target_uids,
            "gtex_embedding": embeddings,
        }
    )

    df.write_parquet(output_parquet)
    size_kb = output_parquet.stat().st_size / 1024
    print(
        f"-> Embeddings GTEx sauvegardés : {output_parquet} ({size_kb:.1f} Ko, shape: {df.shape})"
    )
    return df

def assemble_target_features_vp(
    features_dir: Path,
    output_parquet: Path,
) -> pl.DataFrame:
    """
    Joint les 4 modalités (ESM-2, Reactome, STRING, GTEx), concatène en 1590-d
    et applique la normalisation L2 : v_p = v_p / ||v_p||_2.
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print("Assemblage final du vecteur multi-omique v_p (1 590 dimensions)...")

    # 1. Chargement des 4 tables
    esm = pl.read_parquet(features_dir / "esm2_embeddings.parquet")
    react = pl.read_parquet(features_dir / "reactome_embeddings.parquet")
    string = pl.read_parquet(features_dir / "string_embeddings.parquet")
    gtex = pl.read_parquet(features_dir / "gtex_embeddings.parquet")

    # 2. Jointures strictes sur uniprot_id
    merged = (
        esm.join(react, on="uniprot_id", how="inner")
        .join(string, on="uniprot_id", how="inner")
        .join(gtex, on="uniprot_id", how="inner")
        .sort("uniprot_id")
    )

    uids = merged["uniprot_id"].to_list()
    
    # 3. Concaténation matricielle NumPy
    mat_esm = np.array(merged["esm2_embedding"].to_list(), dtype=np.float32)       # (N, 1280)
    mat_react = np.array(merged["reactome_embedding"].to_list(), dtype=np.float32) # (N, 128)
    mat_string = np.array(merged["string_embedding"].to_list(), dtype=np.float32)  # (N, 128)
    mat_gtex = np.array(merged["gtex_embedding"].to_list(), dtype=np.float32)      # (N, 54)

    vp_raw = np.hstack([mat_esm, mat_react, mat_string, mat_gtex])                 # (N, 1590)

    # 4. Normalisation L2 par ligne (Section 3.3 du protocole)
    norms = np.linalg.norm(vp_raw, axis=1, keepdims=True)
    vp_normed = vp_raw / np.maximum(norms, 1e-9)

    # 5. Sauvegarde
    df_vp = pl.DataFrame({
        "uniprot_id": uids,
        "vp": [vp_normed[i].tolist() for i in range(len(uids))],
    })
    
    df_vp.write_parquet(output_parquet)
    size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"-> Matrice v_p finalisée : {output_parquet} ({size_mb:.2f} Mo, shape: {df_vp.shape})")
    print(f"-> Dimension finale par cible : {len(df_vp['vp'][0])} (Norme L2 moyenne = {np.mean(np.linalg.norm(vp_normed, axis=1)):.4f})")
    
    return df_vp