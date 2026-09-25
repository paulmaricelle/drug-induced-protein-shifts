#!/usr/bin/env python3
"""Extraction résumable des représentations MOTOR baseline z0 à t0 (Section 4.1).

Pipeline :
  1. Collecte des couples (person_id, t0) de toutes les cohortes sélectionnées
     (manifeste : status == SAVED & n_final_stanford >= min_n), déduplication :
     un couple commun à plusieurs cohortes n'est inféré qu'une fois.
  2. Reprise : les couples déjà présents dans le magasin central
     (``<out_root>/store_<anchor>/chunk_*.parquet``) ou déjà en échec définitif
     (``failures_*.parquet`` : patient absent de l'extrait FEMR, pas d'historique)
     sont ignorés.
  3. Découpage par PATIENTS (tous les t0 d'un patient dans le même lot) :
     ``femr_compute_representations`` accepte plusieurs instants de prédiction
     par patient et ne parcourt la timeline qu'une fois (contrairement à
     ``rabit_pipeline.py`` qui impose un patient par fichier).
  4. Chaque lot est exécuté dans l'environnement partagé ``femr_v1_comp`` par le
     worker de ``src/cohorts/motor.py`` (contrôle + inférence GPU), puis écrit
     immédiatement comme fragment du magasin (person_id, t0, z[768] float32).
  5. ``--dispatch`` : redistribution vers chaque cohorte sous forme de
     ``stanford_motor_z0.npy`` (N, 768) aligné ligne à ligne sur
     ``stanford_index.parquet`` (NaN si aucune représentation), + un JSON de
     provenance ``stanford_motor_z0.json``. C'est le fichier relu par
     ``DrugCohort.from_disk``.

Ancrage t0 -> prediction_time : voir ``src/cohorts/motor.py`` (défaut
``day_start`` = t0 00:00 : historique jusqu'à la fin de la veille, jour t0 exclu).

Exemples :
  # Test sur 2 petites cohortes, 20 couples chacune, GPU 0
  python scripts/extract_motor_representations.py --drugs 123 456 \\
      --sample-per-cohort 20 --out-root data/motor_test --gpu 0 --dispatch
  # Production (ne pas lancer sans décision du responsable)
  python scripts/extract_motor_representations.py --min-n 1 --gpu 0 --dispatch
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl

from src.cohorts.motor import (
    ANCHOR_OFFSETS_MIN,
    DEFAULT_ANCHOR,
    FEMR_DATA_SOURCES,
    FEMR_ENV_PYTHON,
    MOTOR_MODEL_DIR,
    REP_DIM,
    STATUS_NO_REP,
    STATUS_OK,
)
from src.config import PathConfig

WORKER_SCRIPT = ROOT_DIR / "src" / "cohorts" / "motor.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extraction MOTOR z0 à t0 (résumable, dédupliquée)")
    p.add_argument("--cohorts-dir", type=Path, default=PathConfig(is_sample=False).output_cohorts_dir)
    p.add_argument("--out-root", type=Path, default=ROOT_DIR / "data" / "motor",
                   help="Racine du magasin central et des fichiers de travail (sous data/, gitignoré)")
    p.add_argument("--drugs", type=int, nargs="*", default=None, help="Restreindre à ces drug_id")
    p.add_argument("--min-n", type=int, default=1, help="n_final_stanford minimal")
    p.add_argument("--sample-per-cohort", type=int, default=None,
                   help="TEST : nb de couples tirés au hasard par cohorte")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchor", choices=list(ANCHOR_OFFSETS_MIN), default=DEFAULT_ANCHOR)
    p.add_argument("--data-source", default="shc_2026", choices=list(FEMR_DATA_SOURCES))
    p.add_argument("--gpu", type=int, required=False, default=None, help="Index GPU (obligatoire pour l'inférence)")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Taille de lot FEMR en événements (puissance de 2 ; 1024 = défaut rabit_pipeline)")
    p.add_argument("--mem-fraction", type=float, default=0.25,
                   help="Fraction de mémoire GPU préallouée par JAX (GPU partagés)")
    p.add_argument("--max-gpu-used-mb", type=int, default=4000,
                   help="Refuse de lancer si le GPU utilise déjà plus que cette mémoire (sauf --force)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--chunk-patients", type=int, default=20000, help="Patients par lot FEMR")
    p.add_argument("--max-chunks", type=int, default=None, help="Arrêt après N lots (tests / tranches)")
    p.add_argument("--shard", default=None,
                   help="Répartition multi-GPU 'i/n' : ce processus ne traite que les patients"
                        " person_id %% n == i (un processus par GPU, magasin partagé)")
    p.add_argument("--qc-max", type=int, default=2000, help="Instances par lot pour le contrôle détaillé du jour t0")
    p.add_argument("--dispatch", action="store_true", help="Redistribuer le magasin vers les cohortes")
    p.add_argument("--dispatch-only", action="store_true", help="Pas d'inférence, redistribution seule")
    p.add_argument("--dispatch-dir", type=Path, default=None,
                   help="Dossier des cohortes cibles pour la redistribution (défaut : --cohorts-dir ; "
                        "en mode test : <out-root>/cohorts)")
    p.add_argument("--keep-work", action="store_true", help="Conserver les dossiers de travail des lots")
    return p.parse_args()


# =============================================================================
# 1. Collecte et déduplication des instances
# =============================================================================
def collect_instances(args: argparse.Namespace) -> pl.DataFrame:
    """Retourne (drug_id, row_idx, person_id, t0) pour les cohortes sélectionnées."""
    manifest = pl.read_parquet(args.cohorts_dir / "manifest.parquet")
    sel = manifest.filter((pl.col("status") == "SAVED") & (pl.col("n_final_stanford") >= args.min_n))
    if args.drugs:
        sel = sel.filter(pl.col("drug_id").is_in(args.drugs))
    frames = []
    for drug_id in sorted(sel["drug_id"].to_list()):
        idx_file = args.cohorts_dir / f"cohort_{drug_id}" / "stanford_index.parquet"
        if not idx_file.exists():
            print(f"[Attention] Index absent pour la cohorte {drug_id}")
            continue
        df = (
            pl.read_parquet(idx_file, columns=["person_id", "t0"])
            .with_row_index("row_idx")
            .select(
                pl.lit(int(drug_id), dtype=pl.Int64).alias("drug_id"),
                pl.col("row_idx").cast(pl.Int64),
                pl.col("person_id").cast(pl.Int64),
                pl.col("t0").cast(pl.Date),
            )
        )
        if args.sample_per_cohort:
            df = df.sample(n=min(args.sample_per_cohort, df.height), seed=args.seed)
        frames.append(df)
    if not frames:
        sys.exit("Aucune cohorte sélectionnée.")
    return pl.concat(frames)


def add_prediction_time(df: pl.DataFrame, anchor: str) -> pl.DataFrame:
    offset = ANCHOR_OFFSETS_MIN[anchor]
    return df.with_columns(
        (pl.col("t0").cast(pl.Datetime("us")) + pl.duration(minutes=offset))
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("prediction_time")
    )


def store_dir(args: argparse.Namespace) -> Path:
    return args.out_root / f"store_{args.anchor}"


def done_keys(sdir: Path) -> pl.DataFrame:
    """Couples déjà extraits ou en échec définitif."""
    files = sorted(sdir.glob("chunk_*.parquet")) + sorted(sdir.glob("failures_*.parquet"))
    if not files:
        return pl.DataFrame(schema={"person_id": pl.Int64, "t0": pl.Date})
    return pl.concat(
        [pl.read_parquet(f, columns=["person_id", "t0"]) for f in files]
    ).unique()


# =============================================================================
# 2. GPU
# =============================================================================
def check_gpu(gpu: int, max_used_mb: int, force: bool) -> None:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    used, total, util = (int(x.strip()) for x in out.split(","))
    print(f"GPU {gpu} : {used:,}/{total:,} MiB utilisés, utilisation {util} %")
    if used > max_used_mb and not force:
        sys.exit(f"GPU {gpu} déjà chargé (> {max_used_mb} MiB). Choisir un autre GPU ou --force.")


# =============================================================================
# 3. Inférence par lots
# =============================================================================
def run_chunk(args: argparse.Namespace, chunk: pl.DataFrame, chunk_id: str) -> dict:
    sdir = store_dir(args)
    work = args.out_root / "work" / chunk_id
    work.mkdir(parents=True, exist_ok=True)
    inst_csv = work / "instances.csv"
    chunk.select(
        "person_id", pl.col("t0").dt.strftime("%Y-%m-%d"), "prediction_time"
    ).write_csv(inst_csv)

    cmd = [
        FEMR_ENV_PYTHON, str(WORKER_SCRIPT), "worker",
        "--instances", str(inst_csv),
        "--out_dir", str(work),
        "--data_path", FEMR_DATA_SOURCES[args.data_source],
        "--model_path", MOTOR_MODEL_DIR,
        "--gpu", str(args.gpu),
        "--batch_size", str(args.batch_size),
        "--tmpdir", str(args.out_root / "tmp_femr"),
        "--qc_max", str(args.qc_max),
        "--mem_fraction", str(args.mem_fraction),
    ]
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    t_start = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - t_start

    qc = json.loads((work / "qc.json").read_text())
    status = pl.read_csv(work / "status.csv", schema_overrides={"person_id": pl.Int64, "t0": pl.Utf8})
    status = status.with_columns(pl.col("t0").str.to_date())

    shard = None
    if (work / "reps.npy").exists():
        reps = np.load(work / "reps.npy")
        keys = pl.read_csv(work / "keys.csv", schema_overrides={"patient_id": pl.Int64})
        keyed = keys.with_row_index("rep_row").join(
            chunk.select("person_id", "t0", "prediction_time"),
            left_on=["patient_id", "prediction_time"],
            right_on=["person_id", "prediction_time"],
            how="inner",
        )
        z = reps[keyed["rep_row"].to_numpy()]
        shard = pl.DataFrame({
            "person_id": keyed["patient_id"],
            "t0": keyed["t0"],
        }).with_columns(pl.Series("z", z, dtype=pl.Array(pl.Float32, REP_DIM)))
        tmp = sdir / f".chunk_{chunk_id}.parquet.tmp"
        shard.write_parquet(tmp, compression="zstd")
        os.replace(tmp, sdir / f"chunk_{chunk_id}.parquet")

    # Échecs définitifs (absent de FEMR, pas d'historique, label sans représentation)
    got = shard.select("person_id", "t0") if shard is not None else \
        pl.DataFrame(schema={"person_id": pl.Int64, "t0": pl.Date})
    fails = status.filter(pl.col("status") != STATUS_OK)
    no_rep = status.filter(pl.col("status") == STATUS_OK).join(got, on=["person_id", "t0"], how="anti") \
        .with_columns(pl.lit(STATUS_NO_REP).alias("status"))
    fails = pl.concat([fails, no_rep])
    if fails.height:
        fails.with_columns(pl.lit(chunk_id).alias("chunk")).write_parquet(sdir / f"failures_{chunk_id}.parquet")

    qc["n_no_rep"] = no_rep.height
    qc["wall_s"] = round(elapsed, 1)
    (sdir / f"qc_{chunk_id}.json").write_text(json.dumps(qc, indent=2))
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return qc


def summarize_qc(qcs: list[dict]) -> None:
    if not qcs:
        return
    tot = {k: sum(q.get(k, 0) for q in qcs) for k in
           ("n_instances", "n_patients", "n_absent_femr", "n_no_history", "n_ok", "n_t0_in_timeline",
            "n_reps", "n_no_rep")}
    inf_s = sum(q["timing_s"].get("femr_compute_representations", 0) for q in qcs)
    wall = sum(q["wall_s"] for q in qcs)
    print("\n" + "=" * 70)
    print("BILAN DE L'EXTRACTION (agrégats)")
    for k, v in tot.items():
        print(f"  {k:<22} {v:>10,}")
    if tot["n_instances"]:
        print(f"  Taux d'appariement FEMR (patients présents) : "
              f"{1 - tot['n_absent_femr'] / tot['n_instances']:.1%} des couples")
    print(f"  Temps femr_compute_representations : {inf_s:,.0f} s ; temps total lots : {wall:,.0f} s")
    if tot["n_reps"] and wall:
        print(f"  Débit : {tot['n_reps'] / wall:,.1f} couples/s (mur), "
              f"{tot['n_reps'] / max(inf_s, 1e-9):,.1f} couples/s (inférence FEMR)")
    for q in qcs:
        if q.get("qc_day_t0"):
            print(f"  Contrôle jour t0 : {json.dumps(q['qc_day_t0'])}")


# =============================================================================
# 4. Redistribution vers les cohortes
# =============================================================================
def dispatch(args: argparse.Namespace, instances: pl.DataFrame) -> None:
    sdir = store_dir(args)
    shards = sorted(sdir.glob("chunk_*.parquet"))
    if not shards:
        print("Magasin vide : rien à redistribuer.")
        return
    target_root = args.dispatch_dir
    sizes = instances.group_by("drug_id").agg(pl.col("row_idx").max() + 1)
    # En mode échantillon, les lignes non tirées restent NaN : taille = cohorte complète
    n_rows = {}
    for drug_id in sizes["drug_id"].to_list():
        idx_file = args.cohorts_dir / f"cohort_{drug_id}" / "stanford_index.parquet"
        n_rows[drug_id] = pl.scan_parquet(idx_file).select(pl.len()).collect().item()

    tmp_files = {}
    for drug_id, n in n_rows.items():
        out_dir = target_root / f"cohort_{drug_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / ".stanford_motor_z0.tmp.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, REP_DIM))
        mm[:] = np.nan
        mm.flush()
        del mm
        tmp_files[drug_id] = tmp

    filled = {d: 0 for d in n_rows}
    mapping = instances.select("drug_id", "row_idx", "person_id", "t0")
    for shard in shards:
        df = pl.read_parquet(shard).join(mapping, on=["person_id", "t0"], how="inner")
        if df.height == 0:
            continue
        for (drug_id,), part in df.partition_by("drug_id", as_dict=True).items():
            mm = np.lib.format.open_memmap(tmp_files[drug_id], mode="r+")
            mm[part["row_idx"].to_numpy()] = part["z"].to_numpy()
            mm.flush()
            del mm
            filled[drug_id] += part.height

    for drug_id, tmp in tmp_files.items():
        out_dir = tmp.parent
        os.replace(tmp, out_dir / "stanford_motor_z0.npy")
        prov = {
            "anchor": args.anchor,
            "prediction_time_offset_min": ANCHOR_OFFSETS_MIN[args.anchor],
            "data_source": args.data_source,
            "model": MOTOR_MODEL_DIR,
            "batch_size": args.batch_size,
            "n_rows": n_rows[drug_id],
            "n_with_z0": filled[drug_id],
            "date": datetime.date.today().isoformat(),
        }
        (out_dir / "stanford_motor_z0.json").write_text(json.dumps(prov, indent=2))
        meta_file = out_dir / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            meta["has_stanford_z0"] = True
            meta_file.write_text(json.dumps(meta, indent=2))
    n_tot, n_fill = sum(n_rows.values()), sum(filled.values())
    print(f"Redistribution : {len(tmp_files)} cohortes -> {target_root} ; "
          f"{n_fill:,}/{n_tot:,} lignes avec z0 ({n_fill / max(n_tot, 1):.1%})")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    if args.dispatch_dir is None:
        args.dispatch_dir = args.out_root / "cohorts" if args.sample_per_cohort else args.cohorts_dir
    sdir = store_dir(args)
    sdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"EXTRACTION MOTOR z0 | ancrage={args.anchor} | source={args.data_source}")
    print("=" * 70)
    instances = collect_instances(args)
    uniq = instances.select("person_id", "t0").unique()
    print(f"Instances : {instances.height:,} lignes de cohortes, "
          f"{uniq.height:,} couples (patient, t0) uniques, {uniq['person_id'].n_unique():,} patients, "
          f"{instances['drug_id'].n_unique():,} cohortes")

    if not args.dispatch_only:
        todo = uniq.join(done_keys(sdir), on=["person_id", "t0"], how="anti")
        print(f"Reste à extraire : {todo.height:,} couples ({uniq.height - todo.height:,} déjà traités)")
        if todo.height:
            if args.gpu is None:
                sys.exit("--gpu est obligatoire pour l'inférence.")
            check_gpu(args.gpu, args.max_gpu_used_mb, args.force)
            shard_tag = ""
            if args.shard:
                shard_i, shard_n = (int(x) for x in args.shard.split("/"))
                if not 0 <= shard_i < shard_n:
                    sys.exit(f"--shard invalide : {args.shard}")
                todo = todo.filter((pl.col("person_id") % shard_n) == shard_i)
                shard_tag = f"_s{shard_i}of{shard_n}"
                print(f"Shard {shard_i}/{shard_n} : {todo.height:,} couples pour ce processus")
            todo = add_prediction_time(todo, args.anchor)
            pids = todo["person_id"].unique().sort()
            n_chunks = (len(pids) + args.chunk_patients - 1) // args.chunk_patients
            run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            qcs = []
            for i in range(n_chunks):
                if args.max_chunks is not None and i >= args.max_chunks:
                    print(f"Arrêt après {args.max_chunks} lot(s) (--max-chunks).")
                    break
                chunk_pids = pids[i * args.chunk_patients:(i + 1) * args.chunk_patients]
                chunk = todo.filter(pl.col("person_id").is_in(chunk_pids.implode()))
                chunk_id = f"{run_id}{shard_tag}_{i:05d}"
                print(f"\n--- Lot {i + 1}/{n_chunks} ({chunk_id}) : {chunk.height:,} couples, "
                      f"{len(chunk_pids):,} patients ---")
                try:
                    qcs.append(run_chunk(args, chunk, chunk_id))
                except subprocess.CalledProcessError as e:
                    # Échec technique : consigné, le lot sera repris au prochain lancement
                    with open(sdir / "errors.log", "a") as f:
                        f.write(f"{datetime.datetime.now().isoformat()} lot {chunk_id} : {e}\n")
                    print(f"[Erreur] Lot {chunk_id} en échec (voir {sdir / 'errors.log'})")
            summarize_qc(qcs)

    if args.dispatch or args.dispatch_only:
        dispatch(args, instances)


if __name__ == "__main__":
    main()
