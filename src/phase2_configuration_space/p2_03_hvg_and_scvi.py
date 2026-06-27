"""Phase 2.5–2.6 — global gene filter, HVG selection, scVI training at three
latent dimensionalities (20, 30, 40). The d=30 result is the primary
configuration space for Phase 3; d=20 and d=40 are kept so Phase 3 can run
landscape topology robustness checks (stability of attractor counts across
latent dim).

Inputs:    data/processed/van_galen_qc.h5ad
Outputs:   data/processed/van_galen_integrated.h5ad
           models/scvi/van_galen_d{20,30,40}/
           outputs/integration/scvi_training.log
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import torch
import time
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/scvi"
OUTI = ROOT / "outputs/integration"
MODELS.mkdir(exist_ok=True, parents=True)
OUTI.mkdir(exist_ok=True, parents=True)

torch.manual_seed(0)
np.random.seed(0)
scvi.settings.seed = 0

print("[scvi] loading QC'd AnnData…")
adata = sc.read_h5ad(PROC / "van_galen_qc.h5ad")
adata.X = adata.layers["counts"]  # raw counts for scVI
print(f"[scvi] {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# ---------------------------------------------------------------------------
# 1. Gene-level filter (≥10 cells across the cohort)
# ---------------------------------------------------------------------------
sc.pp.filter_genes(adata, min_cells=10)
print(f"[scvi] after gene filter (≥10 cells): {adata.n_vars:,} genes")

# ---------------------------------------------------------------------------
# 2. Tirosh cell-cycle scores (stored for downstream — NOT regressed out here)
# ---------------------------------------------------------------------------
sigs = json.load(open(ROOT / "data/raw/msigdb/AML_signatures.json"))
s_genes = [g for g in sigs["Tirosh_S_phase"] if g in adata.var_names]
g2m_genes = [g for g in sigs["Tirosh_G2M_phase"] if g in adata.var_names]
print(f"[scvi] cell-cycle gene coverage: S={len(s_genes)}/{len(sigs['Tirosh_S_phase'])}, "
      f"G2M={len(g2m_genes)}/{len(sigs['Tirosh_G2M_phase'])}")

# Score requires log-normalized data; build a temporary normalized copy
tmp = adata.copy()
sc.pp.normalize_total(tmp, target_sum=1e4)
sc.pp.log1p(tmp)
sc.tl.score_genes_cell_cycle(tmp, s_genes=s_genes, g2m_genes=g2m_genes)
adata.obs["S_score"] = tmp.obs["S_score"]
adata.obs["G2M_score"] = tmp.obs["G2M_score"]
adata.obs["phase"] = tmp.obs["phase"]
del tmp
print(f"[scvi] cell cycle phases: {adata.obs['phase'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 3. HVG selection
# Try seurat_v3 first (preferred, uses raw counts), fallback to seurat (log)
# ---------------------------------------------------------------------------
print("[scvi] selecting HVGs (seurat_v3 with batch_key='patient_id')…")
try:
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3",
                                 batch_key="patient_id", subset=False, layer="counts",
                                 span=0.5)
except (ValueError, Exception) as e:
    print(f"[scvi] seurat_v3 failed ({e}); falling back to seurat on log-normalized")
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=3000, flavor="seurat",
                                 batch_key="patient_id", subset=False)
    adata.var["highly_variable"] = tmp.var["highly_variable"].values
    adata.var["means"] = tmp.var["means"].values
    adata.var["dispersions_norm"] = tmp.var["dispersions_norm"].values
    del tmp
print(f"[scvi] HVGs: {adata.var['highly_variable'].sum()}")

# Use HVG subset for training; keep full AnnData for later
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
print(f"[scvi] training subset: {adata_hvg.n_obs:,} × {adata_hvg.n_vars:,}")

# ---------------------------------------------------------------------------
# 4. Train scVI at three latent dimensionalities
# ---------------------------------------------------------------------------
scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts", batch_key="patient_id")

results = {}
for d in [30, 20, 40]:
    t0 = time.time()
    print(f"\n[scvi] training d={d}…")
    model = scvi.model.SCVI(
        adata_hvg,
        n_latent=d,
        n_layers=2,
        n_hidden=128,
        gene_likelihood="nb",
        dropout_rate=0.1,
    )
    model.train(
        max_epochs=400,
        early_stopping=True,
        early_stopping_patience=45,
        validation_size=0.1,
        check_val_every_n_epoch=2,
        accelerator="gpu",
        devices=1,
        plan_kwargs={"lr": 1e-3},
    )
    elapsed = time.time() - t0
    model.save(str(MODELS / f"van_galen_d{d}"), overwrite=True)
    latent = model.get_latent_representation()
    results[d] = latent
    print(f"[scvi] d={d} done in {elapsed:.0f}s, latent shape={latent.shape}")

# ---------------------------------------------------------------------------
# 5. Store latents back onto the full AnnData
# ---------------------------------------------------------------------------
for d, latent in results.items():
    adata.obsm[f"X_scVI_d{d}"] = latent

# Primary representation = d=30
adata.obsm["X_scVI"] = adata.obsm["X_scVI_d30"]

# ---------------------------------------------------------------------------
# 6. Normalize + log for downstream non-scVI tools
# ---------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()  # store log-normalized full gene set

# ---------------------------------------------------------------------------
# 7. Neighbors + UMAP on the d=30 latent
# ---------------------------------------------------------------------------
print("[scvi] building neighbor graph (k=15, X_scVI)…")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
print("[scvi] running UMAP…")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

print("[scvi] writing integrated AnnData…")
adata.write_h5ad(PROC / "van_galen_integrated.h5ad", compression="gzip")
print(f"[scvi] DONE. {adata.n_obs:,} cells × {adata.n_vars:,} genes; "
      f"obsm: {list(adata.obsm.keys())}")
