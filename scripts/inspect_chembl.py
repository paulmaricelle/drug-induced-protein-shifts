from pathlib import Path
import polars as pl

chembl_dir = Path("data/chembl")
mech_path = chembl_dir / "chembl_mechanisms.parquet"
aff_path = chembl_dir / "chembl_affinities.parquet"

print("=== 1. MÉCANISMES (chembl_mechanisms.parquet) ===")
if mech_path.exists():
    df_mech = pl.read_parquet(mech_path)
    print("Shape :", df_mech.shape)
    print("Colonnes :", df_mech.columns)
    print(df_mech.head(3))
else:
    print(f"Fichier introuvable : {mech_path}")

print("\n=== 2. AFFINITÉS (chembl_affinities.parquet) ===")
if aff_path.exists():
    df_aff = pl.read_parquet(aff_path)
    print("Shape :", df_aff.shape)
    print("Colonnes :", df_aff.columns)
    print(df_aff.head(3))

    if "standard_type" in df_aff.columns:
        print("\nRépartition des types de mesure :")
        print(df_aff["standard_type"].value_counts().head(10))

    if "standard_units" in df_aff.columns:
        print("\nUnités de mesure :")
        print(df_aff["standard_units"].value_counts().head(5))
else:
    print(f"Fichier introuvable : {aff_path}")
