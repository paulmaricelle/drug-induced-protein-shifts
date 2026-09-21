#!/usr/bin/env python
"""
Extraction robuste et résumable des représentations MOTOR pré-t0.
- Checkpointing : détecte les instances déjà extraites et n'extrait que le delta.
- Dispatch immédiat : chaque passe finie est immédiatement enregistrée dans les Parquets.
- Sécurisation disque : redirige TMPDIR pour protéger la partition système /tmp.
"""

import os

# 1. Bridage CPU strict sur serveur partagé
for var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
):
    os.environ[var] = "1"

# 2. Redirection de TMPDIR vers l'espace projet
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_TMP = os.path.join(BASE_DIR, "data", "tmp_femr")
os.makedirs(CUSTOM_TMP, exist_ok=True)
os.environ["TMPDIR"] = CUSTOM_TMP

import argparse
import gc
import subprocess
import sys
from pathlib import Path
import polars as pl

FEMR_ENV_PYTHON = "/remote/private/starr_omop_deid/rabit/env/femr_v1_comp/bin/python"
RABIT_PIPELINE_SCRIPT = "/remote/private/starr_omop_deid/rabit/rabit_pipeline.py"


def parse_args():
    p = argparse.ArgumentParser(description="Extraction MOTOR pré-t0 (Résumable)")
    p.add_argument("--gpu", type=int, default=2, help="Index du GPU à utiliser (GPU 2 recommandé)")
    p.add_argument("--batch_size", type=int, default=2048, help="Taille des batchs d'inférence")
    p.add_argument("--test", action="store_true", help="Mode test limité à 2 cohortes")
    p.add_argument("--data_source", default="shc_2026", help="Source FEMR (shc ou shc_2026)")
    return p.parse_args()


def get_already_processed_keys(cohorts_dir: Path, cids: list[int]) -> set[tuple[int, str]]:
    """Scanne les fichiers motor_reps.parquet existants pour identifier ce qui est déjà fait."""
    processed = set()
    for cid in cids:
        rep_file = cohorts_dir / f"cohort_{cid}" / "motor_reps.parquet"
        if rep_file.exists():
            try:
                df = pl.read_parquet(rep_file, columns=["person_id", "t0"])
                for row in df.iter_rows():
                    processed.add((row[0], str(row[1])))
            except Exception as e:
                print(f"[Attention] Impossible de lire {rep_file}, ré-extraction prévue : {e}")
    return processed


def prepare_remaining_passes(cohorts_dir: Path, manifest_path: Path, temp_dir: Path, is_test: bool):
    manifest = pl.read_parquet(manifest_path)
    
    if is_test:
        cids = manifest.filter(pl.col("n_final_target") >= 10)["rxnorm_concept_id"].head(2).to_list()
        print(f"[TEST] Mode test sur cohortes : {cids}")
    else:
        cids = manifest.filter(pl.col("n_final_target") >= 1)["rxnorm_concept_id"].to_list()

    # 1. Identifier ce qui est déjà sur disque
    already_done = get_already_processed_keys(cohorts_dir, cids)
    if already_done:
        print(f"Reprise détectée : {len(already_done):,} instances déjà extraites et sécurisées.")

    # 2. Collecter les instances manquantes
    all_records = []
    for cid in cids:
        c_file = cohorts_dir / f"cohort_{cid}" / "target_patients.parquet"
        if not c_file.exists():
            continue
        
        df = pl.read_parquet(c_file).select([
            pl.col("person_id").alias("patient_id"),
            pl.col("t0"),
            pl.lit(cid).alias("concept_id")
        ])
        if is_test:
            df = df.head(25)
        all_records.append(df)

    if not all_records:
        sys.exit("Aucune cohorte trouvée.")

    full_df = pl.concat(all_records)
    
    # 3. Filtrer les instances déjà complètes
    if already_done:
        # Création d'une clé de filtrage rapide
        full_df = full_df.with_columns(
            pl.concat_str([pl.col("patient_id"), pl.lit("_"), pl.col("t0").cast(pl.Utf8)]).alias("key")
        )
        done_keys = {f"{pid}_{t0}" for pid, t0 in already_done}
        full_df = full_df.filter(~pl.col("key").is_in(done_keys)).drop("key")

    if full_df.height == 0:
        print("Toutes les représentations de toutes les cohortes sont déjà extraites !")
        return [], None

    # 4. Décalage temporel anti-leakage : t0 - 1 minute
    full_df = full_df.with_columns(
        (pl.col("t0").cast(pl.Datetime) - pl.duration(minutes=1))
        .dt.truncate("1m")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("prediction_time")
    )

    # 5. Déduplication par passes (patient_id unique au sein d'une passe)
    full_df = full_df.with_columns(
        pl.int_range(0, pl.len()).over("patient_id").alias("pass_id")
    )
    
    n_passes = full_df["pass_id"].max() + 1
    print(f"Reste à extraire : {full_df.height:,} instances réparties en {n_passes} passe(s).")

    pass_files = []
    for p in range(n_passes):
        p_df = full_df.filter(pl.col("pass_id") == p)
        pass_csv = temp_dir / f"pred_times_pass_{p}.csv"
        p_df.select(["patient_id", "prediction_time"]).write_csv(pass_csv)
        pass_files.append((p, pass_csv))

    return pass_files, full_df


def run_single_pass_reps(pred_csv: Path, out_dir: Path, gpu: int, batch_size: int, data_source: str) -> Path:
    tag = pred_csv.stem
    rep_csv = out_dir / f"{tag}_motor_rep.csv"

    femr_bin_dir = str(Path(FEMR_ENV_PYTHON).parent)
    env = os.environ.copy()
    env["PATH"] = f"{femr_bin_dir}:{env.get('PATH', '')}"

    cmd = [
        FEMR_ENV_PYTHON,
        RABIT_PIPELINE_SCRIPT,
        "--data_source", data_source,
        "--pred_times", str(pred_csv),
        "--out_dir", str(out_dir),
        "--stage", "reps",
        "--gpu", str(gpu),
        "--batch_size", str(batch_size),
    ]
    print(f"[{tag}] Inférence MOTOR : {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return rep_csv


def dispatch_single_pass(rep_csv: Path, pass_id: int, mapping_df: pl.DataFrame, cohorts_dir: Path):
    """Dispatche immédiatement une passe terminée dans les fichiers Parquet et libère l'espace."""
    feature_cols = [f"data_{i}" for i in range(768)]
    print(f"[Passe {pass_id}] Ventilation des représentations vers les cohortes...")

    df_rep = pl.read_csv(
        rep_csv,
        columns=["patient_ids"] + feature_cols,
        schema_overrides={"patient_ids": pl.Int64}
    ).rename({"patient_ids": "patient_id"})

    mapping_p = mapping_df.filter(pl.col("pass_id") == pass_id)

    merged = mapping_p.join(
        df_rep,
        on="patient_id",
        how="inner"
    ).rename({"patient_id": "person_id"})

    grouped = merged.partition_by("concept_id", as_dict=True)
    for (cid,), df_c in grouped.items():
        out_file = cohorts_dir / f"cohort_{cid}" / "motor_reps.parquet"
        data_to_write = df_c.drop(["concept_id", "pass_id", "prediction_time"])
        
        if out_file.exists():
            existing = pl.read_parquet(out_file)
            data_to_write = pl.concat([existing, data_to_write]).unique(subset=["person_id", "t0"])

        data_to_write.write_parquet(out_file, compression="zstd")

    del df_rep, merged, grouped
    gc.collect()

    # Nettoyage immédiat du gros CSV pour libérer le disque
    if rep_csv.exists():
        rep_csv.unlink()
    print(f"[Passe {pass_id}] Sauvegarde terminée et CSV intermédiaire supprimé.")


def main():
    args = parse_args()
    
    cohorts_dir = Path("data/cohorts")
    manifest_path = cohorts_dir / "manifest.parquet"
    temp_dir = Path("data/temp_motor_test" if args.test else "data/temp_motor")
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = temp_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_files, mapping_df = prepare_remaining_passes(cohorts_dir, manifest_path, temp_dir, args.test)
    if not pass_files:
        return

    for pass_id, p_csv in pass_files:
        print(f"\n--- Démarrage de la Passe {pass_id + 1}/{len(pass_files)} ---")
        rep_csv = run_single_pass_reps(p_csv, out_dir, args.gpu, args.batch_size, args.data_source)
        dispatch_single_pass(rep_csv, pass_id, mapping_df, cohorts_dir)
        if p_csv.exists():
            p_csv.unlink()

    print("\nExtraction complète achevée avec succès.")


if __name__ == "__main__":
    main()