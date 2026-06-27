"""Phase 4.1–4.2 — Langevin augmentation in the Z-normalized scVI latent space.

The Langevin SDE governing cell-state dynamics is

    dx = s_θ(x) dt  +  √(2 D(x) dt) · ξ,        ξ ~ N(0, I_d)

with:
  * s_θ(x) = -∇φ_θ(x)        the learned conservative score (Phase 3)
  * D_i ∈ R^{d×d}            per-cell anisotropic diffusion tensor (Phase 2.10)

For each rare state s ∈ {A1_basin_edge_0, A1_basin_edge_3, ...}:
  * For each real cell r in state s:
      run K_r trajectories starting at r, integrate for tau steps, save
      synthetic_cells.
  * K_r is chosen so the aggregate per-state augmentation factor ≈ 5
    (with a minimum of 50 synthetic per state to make tiny rare states
    statistically useful).

Integration:
  * Time step dt = 0.02 (Z-normalized units; matches Phase 3's RK2 scale)
  * Trajectory length tau = 20 steps (≈ short within-basin exploration)
  * Per-cell noise: ξ_i ~ N(0, I_d), scaled at each step by L_i where
    L_i L_iᵀ = D_i (Cholesky of D_i). For numerical safety we add
    epsilon * I_d before Cholesky.

Outputs:
  data/augmented/synthetic_latent.npy             (n_synth, d) Z-normed coords
  data/augmented/synthetic_meta.csv               per-cell metadata (state, source, traj_step)
  outputs/augmentation/per_state_counts.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import h5py
import time

ROOT = Path(".")
PROC = ROOT / "data/processed"
AUG = ROOT / "data/augmented"
OUTA = ROOT / "outputs/augmentation"
MODELS = ROOT / "models/score_based"
AUG.mkdir(exist_ok=True, parents=True)
OUTA.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p4_01] device: {device}")

# --- Reload the conservative potential ---
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

ckpt = torch.load(MODELS / "score_model.pt", weights_only=False, map_location=device)
model = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mu = np.array(ckpt["mu"], dtype=np.float32)
sd = np.array(ckpt["sd"], dtype=np.float32)
print(f"[p4_01] potential model loaded, d={ckpt['d']}")

# --- Load the Phase 3 deliverable + diffusion tensors ---
print("[p4_01] loading van_galen_phase3_landscape.h5ad…")
a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad")
Z_raw = a.obsm["X_scVI"].astype(np.float32)
Z = (Z_raw - mu) / sd
n, d = Z.shape
print(f"[p4_01] {n} cells in d={d}")

print("[p4_01] loading per-cell D_local…")
with h5py.File(ROOT / "outputs/diffusion_tensor/D_local.h5") as h:
    D_local = h["D"][:].astype(np.float32)  # (n, d, d)
    D_obs = [b.decode() for b in h["obs_names"][:]]
# Verify alignment
assert D_obs[:5] == a.obs_names[:5].tolist(), "D_local order mismatch"
print(f"[p4_01] D_local shape: {D_local.shape}")

# Also need to scale D by Phase 3's Z-normalization (D was computed in raw scVI
# space; the score model lives in Z-normalized space). The scaling: D_norm =
# diag(1/sd) D diag(1/sd) for any single-cell D, because Z = (X - mu)/sd.
print("[p4_01] re-normalizing D to Z-space…")
inv_sd = 1.0 / sd
D_norm = D_local * (inv_sd[None, :, None]) * (inv_sd[None, None, :])
# Add small ridge for numerical Cholesky stability
ridge = 1e-3 * np.eye(d, dtype=np.float32)
D_norm = D_norm + ridge[None, :, :]

# --- Identify augmentation targets ---
rare_states = a.obs["rare_state_id"].astype(str)
state_counts = rare_states.value_counts()
state_counts = state_counts[state_counts.index != ""]
print(f"[p4_01] {len(state_counts)} rare states to augment:")
for sid, c in state_counts.items():
    print(f"  {sid}: {c} cells")

# Decide K per state: aim for total 5×|real|, minimum 50 synthetic per state
TARGET_FACTOR = 5.0
MIN_SYNTH = 50
target_per_state = {}
for sid, c in state_counts.items():
    target = max(MIN_SYNTH, int(np.ceil(TARGET_FACTOR * c)))
    target_per_state[sid] = target
print(f"\n[p4_01] target synthetic counts: {target_per_state}")

# --- SDE integrator ---
TAU = 20         # trajectory length (steps)
DT = 0.02        # time step (Z-normalized units)

# Move D-Cholesky to GPU only for needed cells (memory: each cell ~3.5 KB)
def chol_per_cell(D_subset):
    """Cholesky of (B, d, d) with regularization fallback."""
    out = np.zeros_like(D_subset)
    for i, M in enumerate(D_subset):
        try:
            out[i] = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # add more regularization
            M2 = M + 0.01 * np.eye(d, dtype=np.float32)
            out[i] = np.linalg.cholesky(M2)
    return out

np.random.seed(0)
torch.manual_seed(0)

all_synth_latent = []
all_synth_meta = []
synth_counts = []
t0 = time.time()
SOURCE_FACTOR = 5  # how many trajectories per source cell

for sid, target_n in target_per_state.items():
    src_idx = np.where(rare_states.values == sid)[0]
    n_src = len(src_idx)
    # Number of trajectories per source cell
    K_per_cell = max(1, int(np.ceil(target_n / n_src)))
    actual_total = K_per_cell * n_src
    print(f"\n[p4_01] {sid}: {n_src} sources × {K_per_cell} trajectories "
          f"→ {actual_total} synthetic ({TAU} steps each)")

    # Build initial conditions: repeat each source K times
    src_Z = torch.tensor(Z[src_idx], device=device, dtype=torch.float32)  # (n_src, d)
    src_L_np = chol_per_cell(D_norm[src_idx])  # (n_src, d, d)
    src_L = torch.tensor(src_L_np, device=device, dtype=torch.float32)

    init = src_Z.repeat_interleave(K_per_cell, dim=0)             # (actual_total, d)
    L = src_L.repeat_interleave(K_per_cell, dim=0)                # (actual_total, d, d)
    source_barcodes = np.repeat(a.obs_names.values[src_idx], K_per_cell)
    traj_index = np.tile(np.arange(K_per_cell), n_src)

    x = init.clone()
    # Integrate Langevin for TAU steps
    for step in range(TAU):
        s = model.score(x)                                        # (B, d)
        eps = torch.randn_like(x)                                 # (B, d)
        # Anisotropic diffusion increment: L @ eps  (using src cell's Cholesky)
        diffusion = torch.einsum("bij,bj->bi", L, eps) * np.sqrt(2 * DT)
        drift = s * DT
        x = x + drift + diffusion

    final = x.detach().cpu().numpy()
    all_synth_latent.append(final)
    for i in range(actual_total):
        all_synth_meta.append({
            "synth_index": len(all_synth_meta),
            "rare_state_id": sid,
            "source_barcode": source_barcodes[i],
            "trajectory": int(traj_index[i]),
            "tau_steps": TAU,
        })
    synth_counts.append({"state": sid, "n_real": n_src,
                          "n_synth": actual_total, "K_per_cell": K_per_cell})

all_synth_latent = np.concatenate(all_synth_latent, axis=0)
meta = pd.DataFrame(all_synth_meta)
print(f"\n[p4_01] generated {len(all_synth_latent)} synthetic latent points "
      f"in {time.time() - t0:.1f}s")

# Save Z-normalized synthetic positions
np.save(AUG / "synthetic_latent.npy", all_synth_latent)
# Also save in the original X_scVI scale (un-normalized) for downstream tools
synth_X_scVI = all_synth_latent * sd + mu
np.save(AUG / "synthetic_X_scVI.npy", synth_X_scVI.astype(np.float32))
meta.to_csv(AUG / "synthetic_meta.csv", index=False)
pd.DataFrame(synth_counts).to_csv(OUTA / "per_state_counts.csv", index=False)
print(f"[p4_01] saved {AUG / 'synthetic_latent.npy'} and meta")
