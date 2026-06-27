"""Phase 4.3 — decode synthetic latent positions to gene-expression counts.

We use the trained scVI model from Phase 2.6 to decode synthetic X_scVI
positions back into per-cell expected gene counts on the 3,000-HVG feature
set. Then we sample integer counts from the NB distribution with scVI's
inferred dispersion.

Important: scVI's `get_likelihood_parameters()` / `posterior_predictive_sample()`
work cleanly only on AnnData-aligned inputs. Since we are decoding *new* latent
positions (not real cells), we use the lower-level generative path:

    z      = synthetic_latent     (n_synth, d_latent)
    library = mean(library of source cell)        (n_synth,)
    px     = decoder(z, library)
    counts ~ NB(px.mean, px.scale)

scVI's `module.generative(z, library, ...)` returns the per-gene rate and
dispersion. We pull those and sample.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import torch
import scipy.sparse as sp

ROOT = Path(".")
PROC = ROOT / "data/processed"
AUG = ROOT / "data/augmented"
MODELS = ROOT / "models/scvi"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p4_02] device: {device}")

# --- load synthetic latent (in original X_scVI scale, NOT Z-normalized) ---
synth_X_scVI = np.load(AUG / "synthetic_X_scVI.npy")
meta = pd.read_csv(AUG / "synthetic_meta.csv")
n_synth = synth_X_scVI.shape[0]
print(f"[p4_02] {n_synth} synthetic latent points to decode")

# --- load the QC'd AnnData + scVI model exactly as in Phase 2.6 ---
print("[p4_02] loading van_galen_qc.h5ad…")
a = sc.read_h5ad(PROC / "van_galen_qc.h5ad")
a.X = a.layers["counts"]
import scanpy as sc
sc.pp.filter_genes(a, min_cells=10)

# HVG selection identical to p2_03
try:
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat_v3",
                                 batch_key="patient_id", subset=False,
                                 layer="counts", span=0.5)
except Exception:
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=3000, flavor="seurat",
                                 batch_key="patient_id", subset=False)
    a.var["highly_variable"] = tmp.var["highly_variable"].values
    del tmp

a_hvg = a[:, a.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(a_hvg, layer="counts", batch_key="patient_id")
print(f"[p4_02] HVG subset: {a_hvg.shape}")

print("[p4_02] loading trained scVI d=30 model…")
model = scvi.model.SCVI.load(str(MODELS / "van_galen_d30"), adata=a_hvg,
                              accelerator="gpu")
print("[p4_02] scVI model loaded")

# --- per-source-cell library size lookup ---
# scVI library is just log(total_counts) for each cell; we use the source cell's
# library size for each synthetic cell (so synthetic cells inherit a realistic
# UMI budget from the rare-state population they were seeded from).
source_bc = meta["source_barcode"].astype(str).values
src_idx = a_hvg.obs_names.get_indexer(source_bc)
assert (src_idx >= 0).all(), "some source barcodes not in HVG AnnData"
src_lib = np.asarray(a_hvg.layers["counts"][src_idx].sum(axis=1)).ravel()
print(f"[p4_02] source library sizes: mean={src_lib.mean():.0f}  "
      f"min={src_lib.min():.0f}  max={src_lib.max():.0f}")

# --- decode in batches via scVI module ---
print("[p4_02] decoding synthetic latent → expected gene rates…")
z = torch.tensor(synth_X_scVI, device=device, dtype=torch.float32)
lib = torch.tensor(np.log(src_lib + 1).astype(np.float32), device=device)
# batch labels — use the source cell's batch (patient)
src_patient = a_hvg.obs["patient_id"].values[src_idx]
batch_mapping = a_hvg.obs["_scvi_batch"].values[src_idx].astype(np.int64)
batch_t = torch.tensor(batch_mapping, device=device)

# scvi-tools 1.3 module API
module = model.module
module.eval()
B = 1024
mu_genes = np.zeros((n_synth, a_hvg.n_vars), dtype=np.float32)
disp_genes = np.zeros((n_synth, a_hvg.n_vars), dtype=np.float32)

with torch.no_grad():
    for b0 in range(0, n_synth, B):
        b1 = min(b0 + B, n_synth)
        z_b = z[b0:b1]
        lib_b = lib[b0:b1]
        batch_b = batch_t[b0:b1]
        # decoder forward
        try:
            # scvi-tools 1.3+ API
            px_rate, px_r, px_dropout = module.decoder(
                module.dispersion, z_b, lib_b.unsqueeze(1), batch_b.unsqueeze(1)
            )
        except Exception as e1:
            # try alternate API
            try:
                gen_out = module.generative(z=z_b, library=lib_b.unsqueeze(1),
                                             batch_index=batch_b.unsqueeze(1))
                px_rate = gen_out["px"].mean
                px_r = gen_out["px"].theta if hasattr(gen_out["px"], "theta") else None
            except Exception as e2:
                print(f"decoder error: {e1} / {e2}")
                raise
        mu_genes[b0:b1] = px_rate.cpu().numpy()
        if px_r is not None:
            r = px_r if px_r.ndim > 1 else px_r.unsqueeze(0).expand_as(px_rate)
            disp_genes[b0:b1] = r.cpu().numpy()
        if b0 % (B * 5) == 0:
            print(f"  decoded {b1}/{n_synth}")

print(f"[p4_02] mu_genes shape: {mu_genes.shape}  "
      f"range: [{mu_genes.min():.3f}, {mu_genes.max():.3f}]")

# --- sample NB counts ---
# Mean μ, dispersion r → NB(r, p) where p = μ/(μ+r); torch implements this via
# negative_binomial. For numerical stability we cap r at 1e3.
print("[p4_02] sampling NB counts…")
r_safe = np.clip(disp_genes, 1e-2, 1e3)
mu_safe = np.clip(mu_genes, 1e-6, None)
# variance σ² = μ + μ²/r ; we generate by gamma-Poisson
synth_counts = np.zeros_like(mu_safe)
rng = np.random.default_rng(0)
# Sample in a vectorized way: gamma(shape=r, scale=μ/r) then Poisson
gamma_shape = r_safe
gamma_scale = mu_safe / r_safe
lam = rng.gamma(gamma_shape, gamma_scale)
synth_counts = rng.poisson(lam).astype(np.int32)
print(f"[p4_02] synthetic counts: total={synth_counts.sum():,}  "
      f"mean per cell={synth_counts.sum(axis=1).mean():.0f}  "
      f"mean per gene={synth_counts.sum(axis=0).mean():.2f}")

# --- save outputs ---
np.save(AUG / "synthetic_counts_hvg.npy", synth_counts)
# Also save the gene list (HVG order) so downstream knows the gene axis
gene_names = a_hvg.var_names.astype(str).tolist()
pd.Series(gene_names).to_csv(AUG / "synthetic_genes.csv", index=False, header=["gene_symbol"])
print(f"[p4_02] saved {AUG / 'synthetic_counts_hvg.npy'} "
      f"({synth_counts.shape[0]} cells × {synth_counts.shape[1]} HVGs)")
