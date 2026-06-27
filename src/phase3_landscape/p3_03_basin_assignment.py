"""Phase 3.3 — basin assignment.

For every real cell, integrate dx/dt = s_θ(x) for MAX_STEPS=400 (RK2, dt=0.05)
and assign the cell to the nearest attractor of its terminal position. This is
the physically principled clustering: cells are in the same basin iff their
deterministic flow descends to the same attractor.

We also compute, for every cell:
  obs['basin']                 attractor id (A0..A_{k-1}) or 'orphan'
  obs['basin_dist_to_attr']    Euclidean distance to assigned attractor
  obs['basin_2nd_attr']        second-nearest attractor id
  obs['basin_margin']          dist(2nd attractor) − dist(1st attractor)
  obs['phi']                   φ_θ(x), the learned potential
  obs['score_norm']            ‖s_θ(x)‖

The 'margin' tells us which cells are deep in their basin (large margin) vs
on a watershed boundary (small margin) — these latter are the 'basin_edge'
rare-cell candidates for Phase 4.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import time

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/score_based"
OUTL = ROOT / "outputs/landscape"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p3_03] device: {device}")

class PotentialMLP(nn.Module):
    def __init__(self, d, h=256, depth=4):
        super().__init__()
        layers = [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)
    def score(self, x):
        x = x.detach().requires_grad_(True)
        phi = self.forward(x).sum()
        return -torch.autograd.grad(phi, x, create_graph=False)[0]
    def phi(self, x): return self.forward(x).detach()

ckpt = torch.load(MODELS / "score_model.pt", weights_only=False, map_location=device)
model = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mu = np.array(ckpt["mu"], dtype=np.float32)
sd = np.array(ckpt["sd"], dtype=np.float32)

# Load attractors found in p3_02
attr = np.load(OUTL / "attractors.npy")
n_attr = attr.shape[0]
print(f"[p3_03] {n_attr} attractors loaded")

# Filter to true attractors (n_negative_eigs == 0) from critical_points.csv
cp = pd.read_csv(OUTL / "critical_points.csv")
true_attr_mask = (cp["kind"] == "attractor").values
attr_true = attr[true_attr_mask]
cp_true = cp[true_attr_mask].reset_index(drop=True)
print(f"[p3_03] {len(attr_true)} true attractors (positive-definite Hessian)")
attr_t = torch.tensor(attr_true, device=device, dtype=torch.float32)

# Load real cells
print("[p3_03] loading X_scVI…")
a = sc.read_h5ad(PROC / "van_galen_phase2_complete.h5ad")
Z_raw = a.obsm["X_scVI"].astype(np.float32)
Z = (Z_raw - mu) / sd
n, d = Z.shape
x_all = torch.tensor(Z, device=device, dtype=torch.float32)
print(f"[p3_03] {n} cells in d={d}")

# ---------------------------------------------------------------------------
# Integrate flow batch-wise
# ---------------------------------------------------------------------------
MAX_STEPS = 400
DT = 0.05
TOL = 1.0       # converge if ‖s_θ‖ < TOL OR very close to an attractor

BATCH = 4096
final_pos = np.zeros((n, d), dtype=np.float32)
print(f"[p3_03] integrating {n} cells, batch={BATCH}, steps={MAX_STEPS}…")
t0 = time.time()

for b0 in range(0, n, BATCH):
    b1 = min(b0 + BATCH, n)
    x = x_all[b0:b1].clone()
    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    for step in range(MAX_STEPS):
        s = model.score(x)
        # Check proximity to any attractor
        d_attr = (x.unsqueeze(1) - attr_t.unsqueeze(0)).norm(dim=-1)  # (B, n_attr)
        min_d, _ = d_attr.min(dim=1)
        is_close = min_d < 0.4
        is_low = s.norm(dim=1) < TOL
        new_conv = (is_close | is_low) & (~converged)
        converged |= new_conv
        if converged.all():
            break
        # only move non-converged cells
        s_eff = s.clone()
        s_eff[converged] = 0
        x = x + DT * s_eff
    final_pos[b0:b1] = x.detach().cpu().numpy()
    if b0 % (BATCH * 4) == 0:
        print(f"  cells {b1}/{n}  [{time.time()-t0:.0f}s]")

print(f"[p3_03] integration done in {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# Assign each final position to nearest attractor; compute margins
# ---------------------------------------------------------------------------
print("[p3_03] assigning basins…")
final_t = torch.tensor(final_pos, device=device, dtype=torch.float32)
d_to_attr = (final_t.unsqueeze(1) - attr_t.unsqueeze(0)).norm(dim=-1).cpu().numpy()
sorted_idx = np.argsort(d_to_attr, axis=1)
sorted_d   = np.take_along_axis(d_to_attr, sorted_idx, axis=1)
basin_id    = sorted_idx[:, 0]
basin_id_2  = sorted_idx[:, 1]
basin_dist  = sorted_d[:, 0]
basin_margin = sorted_d[:, 1] - sorted_d[:, 0]

# Compute φ and ‖s‖ at every real cell
phi_per_cell = np.zeros(n, dtype=np.float32)
score_norm  = np.zeros(n, dtype=np.float32)
for b0 in range(0, n, BATCH):
    b1 = min(b0 + BATCH, n)
    xb = x_all[b0:b1]
    phi_per_cell[b0:b1] = model.phi(xb).cpu().numpy()
    score_norm[b0:b1]  = model.score(xb).norm(dim=1).detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Save into AnnData
# ---------------------------------------------------------------------------
a.obs["basin"] = pd.Categorical([f"A{i}" for i in basin_id])
a.obs["basin_2nd"] = pd.Categorical([f"A{i}" for i in basin_id_2])
a.obs["basin_dist_to_attractor"] = basin_dist
a.obs["basin_margin"] = basin_margin
a.obs["phi"] = phi_per_cell
a.obs["score_norm"] = score_norm

basin_counts = a.obs["basin"].value_counts()
print(f"[p3_03] basin counts: {basin_counts.to_dict()}")

# Cross-tab basin × Van Galen CellType to sanity-check
if "VanGalen_CellType" in a.obs.columns:
    crosstab = pd.crosstab(a.obs["basin"], a.obs["VanGalen_CellType"])
    crosstab.to_csv(OUTL / "basin_vs_vangalen_celltype.csv")
    print(f"[p3_03] wrote basin × Van Galen crosstab")

# Save AnnData
out = PROC / "van_galen_phase3_landscape.h5ad"
a.write_h5ad(out, compression="gzip")
print(f"[p3_03] wrote {out}")
