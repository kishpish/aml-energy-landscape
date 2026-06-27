"""Phase 4.5 — assemble the augmented AnnData.

Concatenate:
  * all 38,193 real cells from van_galen_phase3_landscape.h5ad
  * validated synthetic cells (states with verdict PASS or PASS-SMALL)

For each synthetic cell we store:
  obs['is_synthetic']        True/False
  obs['source_state']        rare_state_id that seeded it
  obs['source_barcode']      real cell barcode it was integrated from
  obs['trajectory_index']    which trajectory replica
  obs['validation_verdict']  state-level verdict

The augmented AnnData uses ONLY the HVG gene set (3,000 genes), because:
  (a) the scVI decoder operates on HVG, so synthetic counts only exist for those
      genes,
  (b) every downstream Phase 3+ analysis operates in scVI latent space anyway,
      and full-gene reconstructions would require model.posterior_predictive
      which is much heavier.

We re-run sc.pp.neighbors + sc.tl.umap on the concatenated X_scVI so the new
synthetic cells are visible in the embedding.

Output:
  data/augmented/van_galen_phase4_augmented.h5ad
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
AUG = ROOT / "data/augmented"
OUTA = ROOT / "outputs/augmentation"

print("[p4_04] loading Phase 3 deliverable + synthetic artifacts…")
a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad")

synth_latent_X_scVI = np.load(AUG / "synthetic_X_scVI.npy")
synth_counts = np.load(AUG / "synthetic_counts_hvg.npy")
synth_genes = pd.read_csv(AUG / "synthetic_genes.csv")["gene_symbol"].tolist()
meta = pd.read_csv(AUG / "synthetic_meta.csv")
val = pd.read_csv(OUTA / "validation_per_state.csv")
print(f"[p4_04] synthetic: {synth_counts.shape}, real: {a.shape}")

# --- Decide which synthetic cells to keep (verdict in PASS / PASS-SMALL) ---
passing_states = val[val["verdict"].isin(["PASS", "PASS-SMALL"])]["state"].tolist()
print(f"[p4_04] passing states: {passing_states}")
keep_mask = meta["rare_state_id"].isin(passing_states).values
n_keep = int(keep_mask.sum())
n_drop = int((~keep_mask).sum())
print(f"[p4_04] keep {n_keep} synthetic; drop {n_drop} (states with REVIEW/FAIL)")

synth_latent_X_scVI = synth_latent_X_scVI[keep_mask]
synth_counts = synth_counts[keep_mask]
meta = meta[keep_mask].reset_index(drop=True)

# --- Build the real-cell subset restricted to the same HVG gene set ---
print("[p4_04] subsetting real cells to HVG gene set…")
gene_idx = np.array([a.var_names.get_loc(g) for g in synth_genes])
real_counts_hvg = a.layers["counts"][:, gene_idx]
if not sp.issparse(real_counts_hvg):
    real_counts_hvg = sp.csr_matrix(real_counts_hvg)
print(f"[p4_04] real HVG counts shape: {real_counts_hvg.shape}")

# Build real AnnData
real_var = a.var.iloc[gene_idx].copy()
real_ad = ad.AnnData(
    X=real_counts_hvg.astype(np.float32),
    obs=a.obs.copy(),
    var=real_var,
    obsm={"X_scVI": a.obsm["X_scVI"].copy()},
    layers={"counts": real_counts_hvg.copy()},
)
real_ad.obs["is_synthetic"] = False
real_ad.obs["source_state"] = ""
real_ad.obs["source_barcode"] = ""
real_ad.obs["trajectory_index"] = -1
real_ad.obs["validation_verdict"] = ""
print(f"[p4_04] real AnnData (HVG-only): {real_ad.shape}")

# --- Build the synthetic AnnData ---
print("[p4_04] building synthetic AnnData…")
n_synth = len(meta)
synth_counts_sp = sp.csr_matrix(synth_counts.astype(np.float32))

# State-level verdict per cell
state_verdict_map = dict(zip(val["state"], val["verdict"]))
synth_obs = pd.DataFrame({
    "is_synthetic": True,
    "source_state": meta["rare_state_id"].values,
    "source_barcode": meta["source_barcode"].astype(str).values,
    "trajectory_index": meta["trajectory"].values.astype(int),
    "validation_verdict": meta["rare_state_id"].map(state_verdict_map).values,
})
# Inherit per-state median obs values from the source cells where possible
# (patient_id, basin, treatment_status, etc.) — pulled from the source_barcode
print("[p4_04] inheriting per-cell metadata from source cells…")
src_idx = a.obs_names.get_indexer(meta["source_barcode"].astype(str).values)
inherited_cols = ["sample_id", "patient_id", "timepoint", "treatment_status",
                   "disease_state", "VanGalen_CellType", "VanGalen_malignant_call",
                   "Author Cell Type", "basin", "phase"]
for col in inherited_cols:
    if col in a.obs.columns:
        synth_obs[col] = a.obs[col].values[src_idx]

# unique barcodes for synthetic
synth_barcodes = [f"synth_{meta.loc[i, 'rare_state_id']}_{i:06d}" for i in range(n_synth)]
synth_obs.index = synth_barcodes

synth_ad = ad.AnnData(
    X=synth_counts_sp,
    obs=synth_obs,
    var=real_var,
    obsm={"X_scVI": synth_latent_X_scVI.astype(np.float32)},
    layers={"counts": synth_counts_sp.copy()},
)
print(f"[p4_04] synthetic AnnData: {synth_ad.shape}")

# Align obs columns: union of real_ad.obs.columns and synth_ad.obs.columns
all_cols = list(real_ad.obs.columns)
for c in synth_ad.obs.columns:
    if c not in all_cols:
        all_cols.append(c)
for c in all_cols:
    if c not in real_ad.obs.columns:
        real_ad.obs[c] = ""
    if c not in synth_ad.obs.columns:
        synth_ad.obs[c] = ""
real_ad.obs = real_ad.obs[all_cols]
synth_ad.obs = synth_ad.obs[all_cols]

# --- Concatenate ---
print("[p4_04] concatenating real + synthetic…")
augmented = ad.concat([real_ad, synth_ad], join="inner", merge="same")
print(f"[p4_04] augmented: {augmented.shape}  is_synthetic counts: "
      f"{augmented.obs['is_synthetic'].value_counts().to_dict()}")

# Coerce object cols to str/category for h5ad
for col in augmented.obs.columns:
    s = augmented.obs[col]
    if s.dtype == object:
        augmented.obs[col] = s.astype(str)
    elif s.dtype == bool:
        augmented.obs[col] = s.astype("uint8")

# --- Re-compute neighbor graph + UMAP on the combined X_scVI ---
print("[p4_04] re-running neighbors + UMAP on combined X_scVI…")
sc.pp.neighbors(augmented, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(augmented, min_dist=0.3, spread=1.0)

# Normalize + log for downstream
sc.pp.normalize_total(augmented, target_sum=1e4)
sc.pp.log1p(augmented)

# --- Save ---
out = AUG / "van_galen_phase4_augmented.h5ad"
augmented.write_h5ad(out, compression="gzip")
print(f"[p4_04] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

# Augmentation summary
summary = {
    "n_real": int((augmented.obs["is_synthetic"].astype(str) == "0").sum() +
                  (augmented.obs["is_synthetic"].astype(str) == "False").sum()),
    "n_synth": int((augmented.obs["is_synthetic"].astype(str) == "1").sum() +
                   (augmented.obs["is_synthetic"].astype(str) == "True").sum()),
    "n_total": augmented.n_obs,
    "passing_states": passing_states,
    "dropped_states": val[~val["verdict"].isin(["PASS", "PASS-SMALL"])]["state"].tolist(),
    "augmentation_factors": (
        meta["rare_state_id"].value_counts().to_dict()
    ),
}
import json
with open(OUTA / "phase4_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[p4_04] summary: {summary}")
