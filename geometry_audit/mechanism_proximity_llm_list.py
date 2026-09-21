from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats
import torch
import torch.nn.functional as F

BENCHMARK_PT = Path("data/benchmark_embeddings_50.pt")


def run_benchmark_audit():
    data = torch.load(BENCHMARK_PT, weights_only=False)
    embeddings = data["embeddings"]
    name_to_cid = data["name_to_cid"]
    triplets = data["triplets"]

    results = []
    untestable = []

    for t in triplets:
        mol_a, mol_b, mol_c = t["A"].lower(), t["B"].lower(), t["C"].lower()

        # Vérifier disponibilité
        if not (mol_a in name_to_cid and mol_b in name_to_cid and mol_c in name_to_cid):
            untestable.append((t["id"], "Concept ID manquant"))
            continue

        cid_a, cid_b, cid_c = name_to_cid[mol_a], name_to_cid[mol_b], name_to_cid[mol_c]

        if not (cid_a in embeddings and cid_b in embeddings and cid_c in embeddings):
            untestable.append((t["id"], "Embedding non extrait"))
            continue

        # Calcul distance cosinus : 1 - cos_sim
        va = F.normalize(embeddings[cid_a].unsqueeze(0), p=2, dim=1)
        vb = F.normalize(embeddings[cid_b].unsqueeze(0), p=2, dim=1)
        vc = F.normalize(embeddings[cid_c].unsqueeze(0), p=2, dim=1)

        d_ab = float(1.0 - torch.mm(va, vb.t()).squeeze())
        d_ac = float(1.0 - torch.mm(va, vc.t()).squeeze())
        delta = d_ac - d_ab

        results.append({
            "id": t["id"],
            "A": t["A"],
            "B": t["B"],
            "C": t["C"],
            "mech": t["mech"],
            "ind": t["ind"],
            "d_AB": d_ab,
            "d_AC": d_ac,
            "delta": delta,
            "pass": delta > 0
        })

    df = pl.DataFrame(results)
    n_eval = len(df)
    n_pass = int(df["pass"].sum())

    print("\n" + "=" * 90)
    print(f"RÉSULTATS DE L'AUDIT SUR LES 50 TRIPLETS GOLD STANDARD ({n_eval}/50 évalués)")
    print("=" * 90)

    deltas = df["delta"].to_numpy()
    t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
    w_stat, p_val_w = stats.wilcoxon(deltas, alternative="two-sided")

    print(f"Victoires Mécanisme d(A, B) < d(A, C) : {n_pass} / {n_eval} ({n_pass / n_eval * 100:.1f}%)")
    print(f"Marge Delta moyenne (d_AC - d_AB)    : {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")
    print(f"Marge Delta médiane                  : {np.median(deltas):+.4f}")
    print(f"P-value (t-test apparié)             : {p_val:.2e}")
    print(f"P-value (Wilcoxon)                   : {p_val_w:.2e}")
    print("-" * 90)

    print(f"{'ID':<4} | {'A':<13} | {'B (Méca)':<13} | {'C (Ind)':<13} | {'d(A,B)':<7} | {'d(A,C)':<7} | {'Delta':<7} | {'État'}")
    print("-" * 90)
    for r in df.iter_rows(named=True):
        status = "PASS" if r["pass"] else "FAIL"
        print(f"{r['id']:<4} | {r['A']:<13} | {r['B']:<13} | {r['C']:<13} | {r['d_AB']:<7.3f} | {r['d_AC']:<7.3f} | {r['delta']:<+7.3f} | {status}")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark_audit()