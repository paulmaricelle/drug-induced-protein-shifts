# scripts/build_catalog.py
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import polars as pl
from src.catalog.chembl import load_and_map_chembl_targets
from src.catalog.protein_features import extract_targets_from_reference_proteome, compute_esm2_embeddings, compute_reactome_embeddings, compute_string_embeddings, compute_gtex_embeddings, assemble_target_features_vp
from src.catalog.drug_features import build_theoretical_drug_catalog
from src.catalog.text_embeddings import extract_and_merge_text_embeddings
from src.catalog.catalog import DrugCatalog

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline du Drug Catalog")
    parser.add_argument(
        "--check-chembl", action="store_true", help="Vérifie le mapping ChEMBL"
    )
    parser.add_argument(
        "--fetch-fastas",
        action="store_true",
        help="Télécharge les séquences FASTA des 811 cibles",
    )
    parser.add_argument(
        "--embed-esm2",
        action="store_true",
        help="Encode les cibles FASTA avec ESM-2 650M",
    )
    parser.add_argument(
    "--embed-reactome",
    action="store_true",
    help="Calcule les représentations SVD (128-d) des voies Reactome",
    )
    parser.add_argument(
    "--embed-string",
    action="store_true",
    help="Calcule les représentations topologiques STRING v12 (128-d)",
    )
    parser.add_argument(
    "--embed-gtex",
    action="store_true",
    help="Extrait les profils tissulaires GTEx v8 (54-d)",
    )

    parser.add_argument(
        "--assemble-vp",
        action="store_true",
        help="Concatène et normalise en L2 les 4 modalités pour produire vp (1590-d)",
    )
    parser.add_argument(
    "--build-catalog",
    action="store_true",
    help="Compile le catalogue théorique final des médicaments (RxNorm -> ua)",
    )

    parser.add_argument(
    "--embed-text",
    action="store_true",
    help="Extrait les embeddings textuels OMOP (Philip) et finalise le catalogue",
    )
    parser.add_argument(
    "--build-final-catalog",
    action="store_true",
    help="Instancie et sérialise l'objet DrugCatalog avec les relations OMOP",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    vocab_dir = Path("/remote/shared/collab/omop-vocabularies/v20250227")
    chembl_dir = ROOT_DIR / "data" / "chembl"
    ref_dir = ROOT_DIR / "data" / "reference"

    if args.check_chembl:
        print("Analyse du mapping RxNorm <-> ChEMBL...")
        df = load_and_map_chembl_targets(vocab_dir, chembl_dir)
        print(f"Paires Médicament-Cible : {len(df)}")

    if args.fetch_fastas:
        print("1. Chargement des cibles ChEMBL...")
        df = load_and_map_chembl_targets(vocab_dir, chembl_dir)
        gz_path = ref_dir / "UP000005640_9606.fasta.gz"
        out_fasta = ref_dir / "targets_811.fasta"
        extract_targets_from_reference_proteome(df, gz_path, out_fasta)

    if args.embed_esm2:
        fasta_path = ref_dir / "targets_811.fasta"
        out_parquet = ROOT_DIR / "data" / "features" / "esm2_embeddings.parquet"
        compute_esm2_embeddings(
            fasta_path=fasta_path,
            output_parquet=out_parquet,
            model_name="facebook/esm2_t33_650M_UR50D",
            batch_size=8,
        )
    if args.embed_reactome:
        esm_df = pl.read_parquet(ROOT_DIR / "data" / "features" / "esm2_embeddings.parquet")
        uids = esm_df["uniprot_id"].to_list()
        reactome_file = ref_dir / "UniProt2Reactome.txt"
        out_parquet = ROOT_DIR / "data" / "features" / "reactome_embeddings.parquet"
        compute_reactome_embeddings(uids, reactome_file, out_parquet, n_components=128)
    if args.embed_string:
        esm_df = pl.read_parquet(
            ROOT_DIR / "data" / "features" / "esm2_embeddings.parquet"
        )
        uids = esm_df["uniprot_id"].to_list()
        links_file = ref_dir / "9606.protein.links.v12.0.txt.gz"
        aliases_file = ref_dir / "9606.protein.aliases.v12.0.txt.gz"
        out_parquet = ROOT_DIR / "data" / "features" / "string_embeddings.parquet"
        compute_string_embeddings(uids, links_file, aliases_file, out_parquet)

    if args.embed_gtex:
        esm_df = pl.read_parquet(
            ROOT_DIR / "data" / "features" / "esm2_embeddings.parquet"
        )
        uids = esm_df["uniprot_id"].to_list()
        gtex_file = (
            ref_dir
            / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
        )
        feather_file = (
            Path("/remote/shared/collab/protein-vocabularies")
            / "uniprot_human_concepts_tab_delimited.feather"
        )
        out_parquet = ROOT_DIR / "data" / "features" / "gtex_embeddings.parquet"
        compute_gtex_embeddings(uids, gtex_file, feather_file, out_parquet)

    if args.assemble_vp:
        feat_dir = ROOT_DIR / "data" / "features"
        out_vp = feat_dir / "vp_targets.parquet"
        assemble_target_features_vp(feat_dir, out_vp)

    if args.build_catalog:
        print("1. Chargement des correspondances ChEMBL...")
        chembl_df = load_and_map_chembl_targets(vocab_dir, chembl_dir)
        vp_path = ROOT_DIR / "data" / "features" / "vp_targets.parquet"
        out_catalog = ROOT_DIR / "data" / "catalog" / "drug_catalog_theoretical.parquet"
        build_theoretical_drug_catalog(chembl_df, vp_path, out_catalog)

    if args.embed_text:
        cat_path = ROOT_DIR / "data" / "catalog" / "drug_catalog_theoretical.parquet"
        feather_dir = Path(
            "/remote/shared/collab/omop-text-embeddings/unified/v1/RxNorm/"
        )
        out_cat = ROOT_DIR / "data" / "catalog" / "drug_catalog_final.parquet"
        out_pt = ROOT_DIR / "data" / "catalog" / "omop_dense_embeddings.pt"
        extract_and_merge_text_embeddings(cat_path, feather_dir, out_cat, out_pt)

    if args.build_final_catalog:
        final_cat_parquet = (
            ROOT_DIR / "data" / "catalog" / "drug_catalog_final.parquet"
        )
        vocab_dir = Path("/remote/shared/collab/omop-vocabularies/v20250227")
        out_jsonl = ROOT_DIR / "data" / "catalog" / "drug_catalog.jsonl"

        catalog = DrugCatalog.from_pipeline_artifacts(
            final_catalog_parquet=final_cat_parquet,
            omop_vocab_dir=vocab_dir,
        )
        catalog.save(out_jsonl)

if __name__ == "__main__":
    main()