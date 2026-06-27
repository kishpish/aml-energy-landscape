"""Phase 3.5 — rare cell triage.

Classify cells into 4 categories based on biophysical criteria:

  * `core`        — cell is deep in its basin (high margin, low phi)
  * `high_U`      — phi(x) > P99 within the cell's basin (top 1% high-energy
                    cells in their basin → quasi-metastable rare states)
  * `basin_edge`  — basin_margin < P5 (the cell is close to a basin
                    boundary, candidate for fate plasticity)
  * `near_saddle` — if any saddles were found, distance to nearest saddle
                    < median pairwise inter-attractor distance / 4

The rare-cell flag drives Phase 4 augmentation targets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTL = ROOT / "outputs/landscape"

print("[p3_05] loading landscape AnnData…")
a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad")
n = a.n_obs

# Read attractor + saddle positions (in normalized Z-space)
attr = np.load(OUTL / "attractors.npy")
saddles = np.load(OUTL / "saddles.npy") if (OUTL / "saddles.npy").stat().st_size > 200 else np.zeros((0, attr.shape[1]))
print(f"[p3_05] {len(attr)} attractors, {len(saddles)} saddles")

# Median pairwise attractor distance, /4 = saddle-proximity threshold
from scipy.spatial.distance import pdist
if len(attr) > 1:
    med_pair = float(np.median(pdist(attr)))
    saddle_thr = med_pair / 4
else:
    saddle_thr = np.inf
print(f"[p3_05] median attractor-pair distance: {med_pair:.2f}, "
      f"saddle proximity threshold: {saddle_thr:.2f}")

# ---------------------------------------------------------------------------
# Rare-cell categorization
# ---------------------------------------------------------------------------
phi = a.obs["phi"].values
basin = a.obs["basin"].astype(str).values
margin = a.obs["basin_margin"].values

# 1. high_U per basin (top 1% within each basin)
high_U_flag = np.zeros(n, dtype=bool)
for b in np.unique(basin):
    idx = np.where(basin == b)[0]
    thr_phi = np.percentile(phi[idx], 99)
    high_U_flag[idx[phi[idx] > thr_phi]] = True
print(f"[p3_05] high_U cells: {high_U_flag.sum()}")

# 2. basin_edge (margin < P5)
edge_thr = np.percentile(margin, 5)
basin_edge_flag = margin < edge_thr
print(f"[p3_05] basin_edge cells (margin < {edge_thr:.3f}): {basin_edge_flag.sum()}")

# 3. near_saddle (if saddles found)
near_saddle_flag = np.zeros(n, dtype=bool)
if len(saddles) > 0:
    Z = a.obsm["X_scVI"].astype(np.float32)
    # Use the same Z-normalization as score model
    mu = a.obsm["X_scVI"].mean(axis=0)
    sd = a.obsm["X_scVI"].std(axis=0) + 1e-6
    Zn = (Z - mu) / sd
    # distance to nearest saddle
    d_to_saddle = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d_to_saddle[i] = np.min(np.linalg.norm(Zn[i] - saddles, axis=1))
    near_saddle_flag = d_to_saddle < saddle_thr
print(f"[p3_05] near_saddle cells: {near_saddle_flag.sum()}")

# ---------------------------------------------------------------------------
# Compose rare_category
# ---------------------------------------------------------------------------
rare_cat = np.array(["core"] * n, dtype=object)
rare_cat[high_U_flag] = "high_U"
rare_cat[basin_edge_flag] = "basin_edge"
rare_cat[near_saddle_flag] = "near_saddle"
# Multi-flag cells: prefer "near_saddle" > "basin_edge" > "high_U"
a.obs["high_U"] = high_U_flag.astype("uint8")
a.obs["basin_edge"] = basin_edge_flag.astype("uint8")
a.obs["near_saddle"] = near_saddle_flag.astype("uint8")
a.obs["rare_category"] = pd.Categorical(rare_cat)

print(f"[p3_05] rare_category counts: {a.obs['rare_category'].value_counts().to_dict()}")

# Subcluster within each rare category & basin to get specific "rare states"
from sklearn.cluster import DBSCAN
rare_state_id = np.full(n, "", dtype=object)
for cat in ["high_U", "basin_edge", "near_saddle"]:
    for b in np.unique(basin):
        mask = (a.obs["rare_category"].values == cat) & (basin == b)
        if mask.sum() < 30:
            continue
        Zsub = a.obsm["X_scVI"][mask]
        db = DBSCAN(eps=2.0, min_samples=5).fit(Zsub)
        for k in range(db.labels_.max() + 1):
            sub_mask = (db.labels_ == k)
            real_idx = np.where(mask)[0][sub_mask]
            sid = f"{b}_{cat}_{k}"
            rare_state_id[real_idx] = sid
a.obs["rare_state_id"] = pd.Categorical(rare_state_id)
state_counts = a.obs["rare_state_id"].value_counts()
state_counts = state_counts[state_counts.index != ""]
print(f"[p3_05] {len(state_counts)} rare states discovered:")
for sid, c in state_counts.head(20).items():
    print(f"  {sid}: {c} cells")

a.write_h5ad(PROC / "van_galen_phase3_landscape.h5ad", compression="gzip")

# Also save a rare-state summary table
ssum = []
for sid in state_counts.index:
    mask = a.obs["rare_state_id"].values == sid
    if not mask.any():
        continue
    row = {
        "state_id": sid,
        "n_cells": int(mask.sum()),
        "basin": basin[mask][0],
        "category": sid.split("_", 2)[1] + ("_" + sid.split("_")[2] if "_" in sid.split("_", 2)[2] else ""),
        "mean_phi": float(np.mean(phi[mask])),
        "mean_basin_margin": float(np.mean(margin[mask])),
    }
    # top Van Galen cell type if available
    if "VanGalen_CellType" in a.obs.columns:
        ct = a.obs.loc[mask, "VanGalen_CellType"].value_counts()
        row["top_vangalen_type"] = ct.index[0] if len(ct) else "nan"
        row["top_vangalen_frac"] = float(ct.iloc[0] / mask.sum()) if len(ct) else 0.0
    # top Author Cell Type
    if "Author Cell Type" in a.obs.columns:
        at = a.obs.loc[mask, "Author Cell Type"].value_counts()
        row["top_atlas_type"] = at.index[0] if len(at) else "nan"
        row["top_atlas_frac"] = float(at.iloc[0] / mask.sum()) if len(at) else 0.0
    ssum.append(row)

df = pd.DataFrame(ssum)
df.to_csv(OUTL / "rare_states_catalog.csv", index=False)
print(f"[p3_05] wrote {OUTL / 'rare_states_catalog.csv'}")
print(df.head(30).to_string())
