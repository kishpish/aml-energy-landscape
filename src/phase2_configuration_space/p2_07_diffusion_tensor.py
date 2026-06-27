"""Phase 2.10 — local diffusion tensor estimation in the scVI latent space.

For each cell, the local diffusion tensor D_i is the empirical covariance of
its k=30 nearest neighbors in the d=30 scVI latent. We store:

  obs['D_local_trace']       trace of D_i — overall noise magnitude
  obs['D_local_anisotropy']  λ_max / λ_min of D_i — directional preference
  obs['D_local_logdet']      log-det of D_i — generalized "volume" of D_i
  obsm['D_local']            full d×d covariance per cell (saved separately
                             in .h5 as a 3-D array)

Phase 3's Langevin integrator will read obsm['D_local'] directly; the scalar
summaries in obs are for diagnostic plotting (high-trace regions should
correspond to cycling populations, low-trace to quiescent stem cells).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from sklearn.neighbors import NearestNeighbors

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUT  = ROOT / "outputs/diffusion_tensor"
OUT.mkdir(exist_ok=True, parents=True)

K = 30

print("[diff] loading CNV-classified AnnData…")
a = sc.read_h5ad(PROC / "van_galen_cnv.h5ad")
Z = a.obsm["X_scVI"]  # d=30 latent
n, d = Z.shape
print(f"[diff] computing local covariance at k={K} for {n} cells in d={d}")

print("[diff] fitting NearestNeighbors…")
nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=-1).fit(Z)
_, idx = nn.kneighbors(Z)
idx = idx[:, 1:]  # drop self

# Compute covariances in batches to manage memory
trace = np.zeros(n, dtype=np.float32)
aniso = np.zeros(n, dtype=np.float32)
logdet = np.zeros(n, dtype=np.float32)
D_all = np.zeros((n, d, d), dtype=np.float32)

BATCH = 1000
for b0 in range(0, n, BATCH):
    b1 = min(b0 + BATCH, n)
    # neighbors positions: shape (batch, K, d)
    neigh = Z[idx[b0:b1]]  # (batch, K, d)
    mu = neigh.mean(axis=1, keepdims=True)  # (batch, 1, d)
    centered = neigh - mu  # (batch, K, d)
    # covariance: (batch, d, d)
    cov = np.einsum("bnj,bnk->bjk", centered, centered) / (K - 1)
    D_all[b0:b1] = cov
    for i in range(b1 - b0):
        eigs = np.linalg.eigvalsh(cov[i])
        eigs = np.clip(eigs, 1e-8, None)
        trace[b0 + i] = eigs.sum()
        aniso[b0 + i] = eigs[-1] / max(eigs[0], 1e-8)
        logdet[b0 + i] = np.log(eigs).sum()
    if b0 % 5000 == 0:
        print(f"  processed {b1}/{n}")

a.obs["D_local_trace"] = trace
a.obs["D_local_anisotropy"] = aniso
a.obs["D_local_logdet"] = logdet

print(f"[diff] trace: mean={trace.mean():.3f}  median={np.median(trace):.3f}  "
      f"min={trace.min():.3f}  max={trace.max():.3f}")
print(f"[diff] anisotropy: median={np.median(aniso):.2f}")

# Save full per-cell covariances separately (h5 for compactness)
out_h5 = OUT / "D_local.h5"
with h5py.File(out_h5, "w") as h:
    h.create_dataset("D", data=D_all, compression="gzip")
    h.create_dataset("obs_names", data=np.array(a.obs_names.astype(str), dtype="S"))
    h.attrs["k"] = K
    h.attrs["latent_dim"] = d
print(f"[diff] wrote per-cell covariances to {out_h5}")

# Save AnnData with scalar columns
a.write_h5ad(PROC / "van_galen_diff.h5ad", compression="gzip")
print(f"[diff] wrote {PROC / 'van_galen_diff.h5ad'}")
