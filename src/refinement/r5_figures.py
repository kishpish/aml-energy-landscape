"""R5 — publication figures.

Generates the main figures as PNG (300 dpi) into outputs/figures/:
  Fig1  landscape UMAP colored by basin + by Van Galen cell type
  Fig2  barrier heatmap + critical-point energies
  Fig3  augmentation validation (5-check radar-ish bar + benchmark jaccard)
  Fig4  LSC subtypes (cell-cycle composition + D_trace)
  Fig5  drug concordance (Beat AML monocyte→MEK/HDAC) + connectivity heatmap
  Fig6  survival: held-out C-index comparison + a KM curve for a nominal state
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
AUG = ROOT / "data/augmented"
FIG = ROOT / "outputs/figures"
FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---------- Fig 1: landscape UMAP ----------
print("[r5] Fig1 landscape UMAP…")
a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad", backed="r")
um = a.obsm["X_umap"][:]
basin = a.obs["basin"].astype(str).values
vg = a.obs["VanGalen_CellType"].astype(str).values
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, lab, title in [(axes[0], basin, "Basin (score-field attractor)"),
                       (axes[1], vg, "Van Galen cell type")]:
    cats = pd.Series(lab).value_counts().index[:12]
    cmap = plt.cm.tab20(np.linspace(0, 1, len(cats)))
    for c, col in zip(cats, cmap):
        m = lab == c
        ax.scatter(um[m,0], um[m,1], s=1, c=[col], label=str(c)[:14], rasterized=True)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=5, loc="best", ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(FIG/"Fig1_landscape_umap.png", bbox_inches="tight"); plt.close()

# ---------- Fig 2: barriers + critical points ----------
print("[r5] Fig2 barriers…")
bar = pd.read_csv(ROOT/"outputs/landscape/barriers.csv")
cp = pd.read_csv(ROOT/"outputs/landscape/critical_points.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
# barrier matrix
basins = sorted(set(bar["from"]) | set(bar["to"]))
M = np.full((len(basins), len(basins)), np.nan)
bi = {b:i for i,b in enumerate(basins)}
for _, r in bar.iterrows():
    M[bi[r["from"]], bi[r["to"]]] = r["barrier_i_to_j"]
    M[bi[r["to"]], bi[r["from"]]] = r["barrier_j_to_i"]
im = axes[0].imshow(M, cmap="viridis")
axes[0].set_xticks(range(len(basins))); axes[0].set_xticklabels(basins)
axes[0].set_yticks(range(len(basins))); axes[0].set_yticklabels(basins)
axes[0].set_title("Barrier height ΔU (from→to)")
plt.colorbar(im, ax=axes[0], fraction=0.046)
# critical point energies
attr = cp[cp["kind"]=="attractor"]
axes[1].bar(attr["id"], attr["phi"], color="steelblue")
axes[1].set_ylabel("φ (potential at attractor)"); axes[1].set_title("Attractor energies")
fig.tight_layout(); fig.savefig(FIG/"Fig2_barriers.png", bbox_inches="tight"); plt.close()

# ---------- Fig 3: augmentation validation + benchmark ----------
print("[r5] Fig3 augmentation…")
val = pd.read_csv(ROOT/"outputs/augmentation/validation_per_state.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
checks = ["range_frac_avg","corr_pearson","marker_KS_p_mean","basin_fidelity"]
val_plot = val.set_index("state")[checks].astype(float)
val_plot.plot(kind="bar", ax=axes[0], width=0.8)
axes[0].axhline(0.5, ls="--", c="grey", lw=0.8)
axes[0].set_title("Synthetic-cell validation (per state)"); axes[0].set_ylabel("score")
axes[0].legend(fontsize=5, ncol=2); axes[0].tick_params(axis='x', labelsize=5, rotation=45)
bpath = ROOT/"outputs/phase4/augmentation_benchmark.csv"
if bpath.exists():
    bm = pd.read_csv(bpath)
    g = bm.groupby("n")[["jaccard_real","jaccard_aug"]].mean()
    g.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Marker recovery vs ground truth"); axes[1].set_xlabel("n real cells sampled")
    axes[1].set_ylabel("Jaccard with full-data markers"); axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG/"Fig3_augmentation.png", bbox_inches="tight"); plt.close()

# ---------- Fig 4: LSC subtypes ----------
print("[r5] Fig4 LSC…")
lsc = pd.read_csv(ROOT/"outputs/phase5/lsc_subtypes.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
x = np.arange(len(lsc))
axes[0].bar(x, lsc["G1_frac"], label="G1", color="#4575b4")
axes[0].bar(x, lsc["S_frac"], bottom=lsc["G1_frac"], label="S", color="#fee090")
axes[0].bar(x, lsc["G2M_frac"], bottom=lsc["G1_frac"]+lsc["S_frac"], label="G2M", color="#d73027")
axes[0].set_xticks(x); axes[0].set_xticklabels(lsc["phase_tag"], rotation=0)
axes[0].set_ylabel("cell-cycle fraction"); axes[0].set_title("LSC subtypes"); axes[0].legend(fontsize=6)
axes[1].bar(lsc["phase_tag"], lsc["D_trace_mean"], color="slateblue")
axes[1].set_ylabel("D_trace (transcriptional noise)"); axes[1].set_title("LSC noise scale")
fig.tight_layout(); fig.savefig(FIG/"Fig4_lsc.png", bbox_inches="tight"); plt.close()

# ---------- Fig 5: Beat AML drug concordance ----------
print("[r5] Fig5 drug concordance…")
beat = pd.read_csv(ROOT/"outputs/phase6/beataml_drug_state_correlation.csv")
top = beat[beat["q_value"]<0.05].nsmallest(15, "spearman_rho")
fig, ax = plt.subplots(figsize=(6, 4))
lbl = (top["state_id"]+" ↔ "+top["drug"]).str[:40]
ax.barh(range(len(top)), top["spearman_rho"], color="firebrick")
ax.set_yticks(range(len(top))); ax.set_yticklabels(lbl, fontsize=5)
ax.invert_yaxis(); ax.set_xlabel("Spearman ρ (state fraction vs drug AUC)")
ax.set_title("Beat AML: state→drug sensitivity (negative = sensitive)")
fig.tight_layout(); fig.savefig(FIG/"Fig5_drug_concordance.png", bbox_inches="tight"); plt.close()

# ---------- Fig 6: survival held-out C-index ----------
print("[r5] Fig6 survival…")
ho = json.load(open(ROOT/"outputs/phase8/prognostic_heldout.json"))
fig, ax = plt.subplots(figsize=(4, 3.6))
names = ["Our panel", "LSC17"]
means = [ho["panel_heldout_cindex_mean"], ho["lsc17_heldout_cindex_mean"]]
sds = [ho["panel_heldout_cindex_sd"], ho["lsc17_heldout_cindex_sd"]]
ax.bar(names, means, yerr=sds, capsize=4, color=["teal","grey"])
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="random")
ax.set_ylim(0.4, 0.7); ax.set_ylabel("Held-out C-index (5-fold nested CV)")
ax.set_title("Prognostic concordance (out-of-sample)"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG/"Fig6_survival.png", bbox_inches="tight"); plt.close()

print(f"[r5] wrote 6 figures to {FIG}")
print([p.name for p in sorted(FIG.glob('*.png'))])
