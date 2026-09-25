# src/cohorts/motor.py
"""Représentations MOTOR baseline z0 à t0 (Section 4.1).

Ce module a deux usages :

1. Côté projet (venv Python 3.12) : constantes et helpers purs (sémantique de
   l'ancrage t0 -> prediction_time FEMR), importés par ``DrugCohort`` et par
   ``scripts/extract_motor_representations.py``.
2. Côté FEMR (environnement partagé ``femr_v1_comp``, Python 3.10) : le
   sous-programme ``worker`` est exécuté comme un script
   (``<femr_python> src/cohorts/motor.py worker ...``). Il contrôle chaque couple
   (patient, t0) dans l'extrait FEMR, lance ``femr_compute_representations``
   (MOTOR motor-t-base, JAX/GPU) avec PLUSIEURS instants de prédiction par
   patient, puis écrit ``reps.npy`` (N, 768) + ``keys.csv`` alignés.

Seule la bibliothèque standard est importée au niveau module : femr, numpy et
pandas sont importés paresseusement dans le worker, pour que ce fichier reste
importable depuis les deux environnements.

Sémantique de l'ancrage (FEMR v1 : la représentation d'un label au temps T est
celle du dernier événement d'âge <= T, cf. ``compute_repr_label_alignment``).
L'ETL STARR -> FEMR (``move_to_day_end``) place les événements datés sans heure
à 23:59 le même jour ; les événements horodatés gardent leur heure.

- ``day_start``  : T = t0 00:00. Historique complet jusqu'à la fin de la veille
  (y compris les événements « 23:59 » de t0 - 1), AUCUN événement du jour t0.
  Repli du protocole (défaut).
- ``minus_1min`` : T = t0 - 1 min = veille 23:59 (ancien script). Identique à
  ``day_start`` sauf pour d'éventuels événements exactement à t0 00:00.
- ``day_end``    : T = t0 23:59. Inclut TOUT le jour t0, y compris les
  médicaments initiés à t0 (fuite de l'exposition) : analyse de sensibilité
  uniquement, non conforme au protocole.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

REP_DIM = 768

RABIT_HOME = "/remote/private/starr_omop_deid/rabit"
FEMR_ENV_PYTHON = f"{RABIT_HOME}/env/femr_v1_comp/bin/python"
FEMR_ENV_BIN = f"{RABIT_HOME}/env/femr_v1_comp/bin"
MOTOR_MODEL_DIR = f"{RABIT_HOME}/models/motor-t-base"
FEMR_DATA_SOURCES = {
    "shc": f"{RABIT_HOME}/stanford_all_patients_5main_2025_04_14/extracts/extract",
    "shc_2026": f"{RABIT_HOME}/stanford_all_patients_5main_2026_07_22/extracts/extract",
}

# Décalage (en minutes) de prediction_time par rapport à t0 00:00
ANCHOR_OFFSETS_MIN = {
    "day_start": 0,
    "minus_1min": -1,
    "day_end": 24 * 60 - 1,
}
DEFAULT_ANCHOR = "day_start"

# Codes de statut par couple (patient, t0) écrits par le worker
STATUS_OK = "ok"
STATUS_ABSENT = "absent_femr"  # person_id absent de l'extrait FEMR
STATUS_NO_HISTORY = "no_history"  # aucun événement clinique <= prediction_time
STATUS_NO_REP = "no_rep"  # label valide mais aucune représentation renvoyée


def anchor_prediction_time(t0: datetime.date, anchor: str = DEFAULT_ANCHOR) -> datetime.datetime:
    """Convertit une date t0 en prediction_time FEMR (résolution minute)."""
    if anchor not in ANCHOR_OFFSETS_MIN:
        raise ValueError(f"Ancrage inconnu : {anchor} (attendu : {list(ANCHOR_OFFSETS_MIN)})")
    base = datetime.datetime.combine(t0, datetime.time.min)
    return base + datetime.timedelta(minutes=ANCHOR_OFFSETS_MIN[anchor])


# =============================================================================
# Worker (environnement femr_v1_comp uniquement)
# =============================================================================
_CLINICAL_TABLES = ("condition_occurrence", "procedure_occurrence", "measurement", "drug_exposure")


def _qc_patient_day(events, t0: datetime.date) -> dict:
    """Agrégats du jour t0 pour un patient (booléens, aucun contenu clinique)."""
    one = datetime.timedelta(days=1)
    out = {
        "drug_t0": False, "drug_t0m1": False, "cond_t0": False,
        "proc_t0": False, "meas_t0": False, "nondrug_t0": False,
        "midnight_t0": False, "n_t0_2359": 0, "n_t0_timed": 0,
    }
    for e in events:
        d = e.start.date()
        if d < t0 - one:
            continue
        if d > t0:
            break
        table = getattr(e, "omop_table", None)
        if d == t0 - one:
            if table == "drug_exposure":
                out["drug_t0m1"] = True
            continue
        # d == t0
        if e.start.time() == datetime.time.min:
            out["midnight_t0"] = True
        if (e.start.hour, e.start.minute) == (23, 59):
            out["n_t0_2359"] += 1
        else:
            out["n_t0_timed"] += 1
        if table == "drug_exposure":
            out["drug_t0"] = True
        elif table == "condition_occurrence":
            out["cond_t0"] = out["nondrug_t0"] = True
        elif table == "procedure_occurrence":
            out["proc_t0"] = out["nondrug_t0"] = True
        elif table == "measurement":
            out["meas_t0"] = out["nondrug_t0"] = True
    return out


def run_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import pickle
    import femr.datasets

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    inst = pd.read_csv(args.instances, dtype={"person_id": "int64", "t0": str, "prediction_time": str})
    inst["t0_date"] = pd.to_datetime(inst["t0"]).dt.date
    inst["pt"] = pd.to_datetime(inst["prediction_time"])

    db = femr.datasets.PatientDatabase(args.data_path)
    status = []
    qc_rows = []
    n_in_timeline = 0
    n_present_checked = 0
    qc_budget = args.qc_max

    for pid, grp in inst.groupby("person_id", sort=False):
        try:
            patient = db[int(pid)]
        except (IndexError, KeyError):
            for idx in grp.index:
                status.append((idx, STATUS_ABSENT))
            continue
        events = patient.events
        clinical = [e for e in events if getattr(e, "omop_table", None) in _CLINICAL_TABLES] \
            if qc_budget > 0 else None
        # Premier / dernier événement clinique (les événements 'person' sont à la naissance)
        first = next((e.start for e in events if getattr(e, "omop_table", None) != "person"), None)
        last = events[-1].start if len(events) else None
        for idx, row in grp.iterrows():
            if first is None or first > row["pt"].to_pydatetime():
                status.append((idx, STATUS_NO_HISTORY))
                continue
            status.append((idx, STATUS_OK))
            n_present_checked += 1
            if first.date() <= row["t0_date"] <= last.date():
                n_in_timeline += 1
            if qc_budget > 0:
                q = _qc_patient_day(clinical, row["t0_date"])
                pt = row["pt"].to_pydatetime()
                q["n_events_hist"] = sum(1 for e in events if e.start <= pt)
                qc_rows.append(q)
                qc_budget -= 1

    st = pd.Series(dict(status)).reindex(inst.index)
    inst["status"] = st.values
    ok = inst[inst["status"] == STATUS_OK]

    qc = {
        "n_instances": int(len(inst)),
        "n_patients": int(inst["person_id"].nunique()),
        "n_absent_femr": int((inst["status"] == STATUS_ABSENT).sum()),
        "n_no_history": int((inst["status"] == STATUS_NO_HISTORY).sum()),
        "n_ok": int(len(ok)),
        "n_t0_in_timeline": int(n_in_timeline),
        "qc_day_t0": {},
        "timing_s": {"qc": round(time.time() - t_start, 1)},
    }
    if qc_rows:
        q = pd.DataFrame(qc_rows)
        qc["qc_day_t0"] = {
            "n": int(len(q)),
            "frac_drug_on_t0": float(q["drug_t0"].mean()),
            "frac_drug_on_t0_minus_1": float(q["drug_t0m1"].mean()),
            "frac_condition_on_t0": float(q["cond_t0"].mean()),
            "frac_procedure_on_t0": float(q["proc_t0"].mean()),
            "frac_measurement_on_t0": float(q["meas_t0"].mean()),
            "frac_any_nondrug_on_t0": float(q["nondrug_t0"].mean()),
            "frac_event_exactly_t0_midnight": float(q["midnight_t0"].mean()),
            "total_t0_events_at_2359": int(q["n_t0_2359"].sum()),
            "total_t0_events_timed": int(q["n_t0_timed"].sum()),
            "mean_events_in_history": float(q["n_events_hist"].mean()),
            "median_events_in_history": float(q["n_events_hist"].median()),
        }
    del db

    labels_csv = os.path.join(args.out_dir, "labels.csv")
    ok[["person_id", "prediction_time"]].rename(columns={"person_id": "patient_id"}) \
        .drop_duplicates().to_csv(labels_csv, index=False)

    rep_pkl = os.path.join(args.out_dir, "reps.pkl")
    if len(ok) > 0:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PATH"] = f"{FEMR_ENV_BIN}:{env.get('PATH', '')}"
        if args.tmpdir:
            os.makedirs(args.tmpdir, exist_ok=True)
            env["TMPDIR"] = args.tmpdir
        if args.mem_fraction:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_fraction)
        cmd = [
            f"{FEMR_ENV_BIN}/femr_compute_representations",
            "--data_path", args.data_path,
            "--model_path", args.model_path,
            "--prediction_times_path", labels_csv,
            "--batch_size", str(args.batch_size),
            rep_pkl,
        ]
        t_inf = time.time()
        with open(os.path.join(args.out_dir, "femr.log"), "w") as logf:
            subprocess.run(cmd, check=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
        qc["timing_s"]["femr_compute_representations"] = round(time.time() - t_inf, 1)

        with open(rep_pkl, "rb") as f:
            reprs = pickle.load(f)
        mat = np.asarray(reprs["representations"], dtype=np.float32)[:, :REP_DIM]
        keys = pd.DataFrame({
            "patient_id": np.asarray(reprs["patient_ids"], dtype=np.int64),
            "prediction_time": [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
                                for t in reprs["prediction_times"]],
        })
        np.save(os.path.join(args.out_dir, "reps.npy"), mat)
        keys.to_csv(os.path.join(args.out_dir, "keys.csv"), index=False)
        os.remove(rep_pkl)
        qc["n_reps"] = int(len(keys))
        qc["rep_dim_raw"] = int(np.asarray(reprs["representations"]).shape[1])
    else:
        qc["n_reps"] = 0

    inst[["person_id", "t0", "status"]].to_csv(os.path.join(args.out_dir, "status.csv"), index=False)
    qc["timing_s"]["total"] = round(time.time() - t_start, 1)
    with open(os.path.join(args.out_dir, "qc.json"), "w") as f:
        json.dump(qc, f, indent=2)
    print(json.dumps({k: v for k, v in qc.items() if k != "qc_day_t0"}), flush=True)


def _parse_worker_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Worker FEMR : contrôle + inférence MOTOR (env femr_v1_comp)")
    p.add_argument("mode", choices=["worker"])
    p.add_argument("--instances", required=True, help="CSV person_id, t0, prediction_time")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_path", required=True, help="Extrait FEMR v1")
    p.add_argument("--model_path", default=MOTOR_MODEL_DIR)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1024, help="Puissance de 2 (événements/lot)")
    p.add_argument("--tmpdir", default=None, help="TMPDIR des lots FEMR temporaires")
    p.add_argument("--qc_max", type=int, default=2000, help="Nb max d'instances pour le contrôle détaillé du jour t0")
    p.add_argument("--mem_fraction", type=float, default=None, help="XLA_PYTHON_CLIENT_MEM_FRACTION")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_worker(_parse_worker_args(sys.argv[1:]))
