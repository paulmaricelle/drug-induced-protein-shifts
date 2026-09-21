from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement du manifeste
manifest_path = Path("data/cohorts/manifest.parquet")
if not manifest_path.exists():
    raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")

df = pl.read_parquet(manifest_path)

# Normalisation du nom de colonne selon les versions
target_col = "n_final_target" if "n_final_target" in df.columns else "n_target_patients"
name_col = "rxnorm_name" if "rxnorm_name" in df.columns else "name"

# Filtrer sur les cohortes qui ont au moins 1 patient retenu
valid_df = df.filter(pl.col(target_col) > 0).sort(target_col, descending=True)
counts = valid_df[target_col].to_numpy()

# -----------------------------------------------------------------------------
# 2. Statistiques descriptives & Quantiles
# -----------------------------------------------------------------------------
quantiles = [10, 25, 50, 75, 90, 95, 99]
q_values = np.percentile(counts, quantiles)

print("\n" + "=" * 70)
print(f"DISTRIBUTION DE LA TAILLE DES COHORTES (SUR {len(counts):,} MOLÉCULES > 0 PATIENTS)")
print("=" * 70)
print(f"Nombre total de molécules évaluées : {df.height:,}")
print(f"Molécules avec >= 1 patient final  : {len(counts):,}")
print(f"Volume total cumulé de patients    : {counts.sum():,}")
print(f"Moyenne de patients par cohorte    : {counts.mean():.1f}")
print(f"Écart-type                         : {counts.std():.1f}")
print("-" * 70)
print("QUANTILES :")
print(f"  Min   : {counts.min():>8,}")
for q, val in zip(quantiles, q_values):
    print(f"  P{q:<2}   : {int(val):>8,}")
print(f"  Max   : {counts.max():>8,}")

# -----------------------------------------------------------------------------
# 3. Répartition par paliers de puissance statistique
# -----------------------------------------------------------------------------
thresholds = [10, 20, 50, 100, 250, 500, 1000, 5000]
print("-" * 70)
print("COHORTES DISPONIBLES PAR SEUIL :")
for th in thresholds:
    n_drugs = (counts >= th).sum()
    pct = (n_drugs / len(counts)) * 100
    print(f"  >= {th:<5} patients : {n_drugs:>5,} molécules ({pct:>5.1f}%)")

# -----------------------------------------------------------------------------
# 4. Top 15 des molécules les plus documentées
# -----------------------------------------------------------------------------
print("-" * 70)
print("TOP 15 DES PLUS GROSSES COHORTES :")
top_cols = ["rxnorm_concept_id", name_col, target_col]
if "atc3" in valid_df.columns:
    top_cols.insert(2, "atc3")

for i, row in enumerate(valid_df.select(top_cols).head(15).iter_rows(named=True), 1):
    atc_str = f"[{row.get('atc3', 'N/A')}]" if "atc3" in row else ""
    print(f"  {i:>2}. {row[name_col]:<28} {atc_str:<8} (ID: {row['rxnorm_concept_id']}) : {row[target_col]:>7,} patients")
print("=" * 70 + "\n")

# -----------------------------------------------------------------------------
# 5. Visualisation graphique (Histogramme Log & ECDF)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme en échelle logarithmique
ax1.hist(counts, bins=np.logspace(np.log10(1), np.log10(counts.max()), 30), color="#1f77b4", edgecolor="black", alpha=0.8)
ax1.set_xscale("log")
ax1.axvline(10, color="orange", linestyle="--", label="Seuil min (10 pts)")
ax1.axvline(np.median(counts), color="red", linestyle="-", label=f"Médiane ({int(np.median(counts))})")
ax1.set_title("Distribution des tailles de cohortes (Échelle Log)")
ax1.set_xlabel("Nombre de patients cibles")
ax1.set_ylabel("Nombre de molécules")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fonction de répartition empirique (ECDF inversée : molécules >= X)
sorted_counts = np.sort(counts)
survival_prob = (len(sorted_counts) - np.arange(len(sorted_counts)))
ax2.plot(sorted_counts, survival_prob, color="#2ca02c", lw=2)
ax2.set_xscale("log")
ax2.set_title("Courbe de survie du catalogue (Molécules >= N patients)")
ax2.set_xlabel("Seuil de patients requis (N)")
ax2.set_ylabel("Nombre de molécules exploitables")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_plot = Path("data/cohorts/cohort_distribution.png")
plt.savefig(out_plot, dpi=200)
print(f"Graphique sauvegardé dans : {out_plot}")