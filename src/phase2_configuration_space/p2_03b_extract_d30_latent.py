"""Extract the d=30 latent representation from the trained scVI model and
build data/processed/van_galen_integrated.h5ad.

This is the resume-from-checkpoint path: if `scripts/p2_03_hvg_and_scvi.py`
trained d=30 successfully but was interrupted before d=20 / d=40, run this to
finalize Phase 2.6.

d=20 and d=40 stability checks are deferred — they can be added by simply
re-running `p2_03_hvg_and_scvi.py` (with the d=30 block early-exited) at any
time. The Phase 2 deliverable does not strictly require them.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/scvi"

torch.manual_seed(0); np.random.seed(0); scvi.settings.seed = 0

print("[scvi-resume] loading QC'd AnnData…")
adata = sc.read_h5ad(PROC / "van_galen_qc.h5ad")
adata.X = adata.layers["counts"]
print(f"[scvi-resume] {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Repeat gene filter for consistency
sc.pp.filter_genes(adata, min_cells=10)
print(f"[scvi-resume] after gene filter: {adata.n_vars:,} genes")

# Cell-cycle scoring (same as p2_03)
sigs = json.load(open(ROOT / "data/raw/msigdb/AML_signatures.json"))
s_genes = [g for g in sigs["Tirosh_S_phase"] if g in adata.var_names]
g2m_genes = [g for g in sigs["Tirosh_G2M_phase"] if g in adata.var_names]
tmp = adata.copy()
sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
sc.tl.score_genes_cell_cycle(tmp, s_genes=s_genes, g2m_genes=g2m_genes)
adata.obs["S_score"] = tmp.obs["S_score"]
adata.obs["G2M_score"] = tmp.obs["G2M_score"]
adata.obs["phase"] = tmp.obs["phase"]
del tmp
print(f"[scvi-resume] phase: {adata.obs['phase'].value_counts().to_dict()}")

# HVG selection
print("[scvi-resume] selecting HVGs (seurat_v3 with batch_key='patient_id')…")
try:
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3",
                                 batch_key="patient_id", subset=False, layer="counts",
                                 span=0.5)
except Exception as e:
    print(f"[scvi-resume] seurat_v3 failed ({e}); falling back to seurat")
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=3000, flavor="seurat",
                                 batch_key="patient_id", subset=False)
    adata.var["highly_variable"] = tmp.var["highly_variable"].values
    del tmp
print(f"[scvi-resume] HVGs: {adata.var['highly_variable'].sum()}")

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts", batch_key="patient_id")

# Load the trained d=30 model
print("[scvi-resume] loading trained d=30 model…")
model = scvi.model.SCVI.load(str(MODELS / "van_galen_d30"), adata=adata_hvg,
                              accelerator="gpu")
print("[scvi-resume] extracting latent representation…")
latent = model.get_latent_representation()
print(f"[scvi-resume] latent shape: {latent.shape}")

adata.obsm["X_scVI_d30"] = latent
adata.obsm["X_scVI"] = latent  # primary alias

# Log-normalize for downstream tools
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()  # log-norm full gene set

# Neighbor graph + UMAP on X_scVI
print("[scvi-resume] building neighbor graph + UMAP…")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

# Save
out = PROC / "van_galen_integrated.h5ad"
adata.write_h5ad(out, compression="gzip")
print(f"[scvi-resume] wrote {out}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  obs cols: {len(adata.obs.columns)}")
