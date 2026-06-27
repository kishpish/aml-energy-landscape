"""Phase 5.2 — biophysical fingerprint per state.

For each state we compute the *dynamical* descriptors that turn cell states
from static gene-marker sets into testable biophysical objects:

  1. attractor_basin      the basin (A0..A3) the state lives in
  2. centroid             cell-state centroid in Z-normalized X_scVI
  3. basin_depth_φ        φ(centroid) − φ(basin's attractor)
                            (state-level energy above the attractor's minimum)
  4. basin_width_logdet   tr-log of local Hessian of φ at centroid;
                            small width = sharp narrow state, large = diffuse
  5. local_score_norm     ‖s_θ(centroid)‖ — small near critical points
  6. D_trace_mean         mean of D_local_trace over real cells in state
                            (transcriptional noise scale)
  7. MFPT_to_A0           mean first-passage time to mature myeloid basin
                            (estimated by ensemble Langevin trajectories)
  8. MFPT_to_A2           mean first-passage time to primitive-blast basin
  9. committor_to_A0      P(hit A0 before all other basins | start here)
 10. committor_to_A2      same for the primitive blast basin

Clinical fates chosen for MFPT/committor:
  * "mature_monocyte"    = basin A0 (myeloid mature attractor)
  * "primitive_blast"    = basin A2 (HSC/GMP-like primitive attractor)

The MFPT computation uses 256 stochastic trajectories per state seed,
integrated with the full Langevin SDE (drift = s_θ, anisotropic diffusion
from per-cell D_local). Trajectories that have not reached any clinical
fate within MAX_STEPS contribute a censored time = MAX_STEPS.

Outputs:
  outputs/phase5/biophysical_fingerprints.csv
  outputs/phase5/mfpt_committor_diag.csv     (per-cell-seed diagnostics)
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
AUG = ROOT / "data/augmented"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/score_based"
OUTL = ROOT / "outputs/landscape"
OUTP5 = ROOT / "outputs/phase5"
OUTP5.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p5_02] device: {device}")

# --- Reload conservative potential model ---
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
        return -torch.autograd.grad(self.forward(x).sum(), x, create_graph=False)[0]
    def phi(self, x): return self.forward(x).detach()
    def hessian_eig(self, x):
        x = x.detach().requires_grad_(True)
        phi = self.forward(x.unsqueeze(0)).sum()
        g = torch.autograd.grad(phi, x, create_graph=True)[0]
        d = x.shape[0]
        H = torch.zeros(d, d, device=x.device)
        for i in range(d):
            H[i] = torch.autograd.grad(g[i], x, retain_graph=(i < d - 1))[0]
        H = 0.5 * (H + H.T)
        return torch.linalg.eigvalsh(H).detach()

ckpt = torch.load(MODELS / "score_model.pt", weights_only=False, map_location=device)
model = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mu = np.array(ckpt["mu"], dtype=np.float32)
sd = np.array(ckpt["sd"], dtype=np.float32)
d = ckpt["d"]
print(f"[p5_02] potential model loaded, d={d}")

# --- Load augmented AnnData with state_id ---
print("[p5_02] loading augmented + state-id AnnData…")
a = sc.read_h5ad(AUG / "van_galen_phase5_states.h5ad")
print(f"[p5_02] {a.shape}, states: {a.obs['state_id'].nunique()}")

# Load full per-cell D_local from Phase 2 (39193 cells)
print("[p5_02] loading per-cell D_local…")
with h5py.File(ROOT / "outputs/diffusion_tensor/D_local.h5") as h:
    D_local = h["D"][:].astype(np.float32)
    D_obs = [b.decode() for b in h["obs_names"][:]]
D_obs_map = {b: i for i, b in enumerate(D_obs)}
# Re-normalize D into Z-space, same as in p4_01
inv_sd = 1.0 / sd
D_norm_global = D_local * (inv_sd[None, :, None]) * (inv_sd[None, None, :]) + \
                 (1e-3 * np.eye(d, dtype=np.float32))[None, :, :]

# Attractors (Z-normalized)
attractors_raw = np.load(OUTL / "attractors.npy")
cp = pd.read_csv(OUTL / "critical_points.csv")
true_attr_mask = (cp["kind"] == "attractor").values
attractors = attractors_raw[true_attr_mask].astype(np.float32)
# NOTE: in the cell-level data, p3_03 relabeled true attractors as A0..A3 in
# sequence after filtering. critical_points.csv has 6 entries (4 attractor + 2
# saddle); after filtering, the 4 attractors become basins A0..A3 in order.
cp_attr_ids = cp.loc[true_attr_mask, "id"].tolist()
# basin "Ak" in cell data → attractor row k of `attractors`
basin_to_idx = {f"A{i}": i for i in range(len(attractors))}
print(f"[p5_02] {len(attractors)} true attractors; mapping basin→idx: {basin_to_idx}")
print(f"        (cp_attr_ids used for original Hessian: {cp_attr_ids})")
attr_t = torch.tensor(attractors, device=device, dtype=torch.float32)
phi_attractors = model.phi(attr_t).cpu().numpy()

# --- Identify clinical fate basins (in cell-data labels) ---
# basin A0 = first attractor in cell space = mature myeloid (per Phase 3 crosstab)
# basin A2 = third = primitive blast
fate_basins = {"A0": "mature_monocyte", "A2": "primitive_blast"}
absorbing_idx = {bid: basin_to_idx[bid] for bid in fate_basins.keys()}
print(f"[p5_02] absorbing fates: {fate_basins}  → attractor indices: {absorbing_idx}")

# Real-cell mask (for source seeding)
is_syn = a.obs["is_synthetic"].astype(str).isin(["True", "1", "1.0"]).values
real_obs_names = a.obs_names.values[~is_syn]
real_basins = a.obs["basin"].astype(str).values[~is_syn]

# Subset Z-normalized latent for the real cells (synth cells used only for
# state-defining centroid; MFPT/committor seeded from real-cell positions).
Z_all = (a.obsm["X_scVI"].astype(np.float32) - mu) / sd

# --- MFPT / committor function ---
N_TRAJ = 128       # trajectories per seed cell
MAX_STEPS = 1500   # cap (30 time units at dt=0.02)
DT = 0.02
SEEDS_PER_STATE = 8   # use up to 8 source cells per state
HIT_RADIUS = 1.5   # within this Z-space distance to attractor = "absorbed"
                  # (pairwise attractor distances are 0.85-3.4; cell-to-attractor
                  #  distances are 5+, so 1.5 is a meaningful watershed threshold)

def run_mfpt(seeds_Z, L_chol, attr_t):
    """Run N_TRAJ trajectories per seed; report:
      * fpt to each absorbing fate (first step within HIT_RADIUS of attractor)
      * final-state committor (which attractor closest at trajectory end)
      * censored fraction (never absorbed)
    """
    n_seed = len(seeds_Z)
    x = torch.tensor(np.repeat(seeds_Z, N_TRAJ, axis=0), device=device, dtype=torch.float32)
    L = torch.tensor(np.repeat(L_chol, N_TRAJ, axis=0), device=device, dtype=torch.float32)
    absorbed = torch.full((len(x),), -1, dtype=torch.long, device=device)
    fpt = torch.full((len(x),), MAX_STEPS, dtype=torch.float32, device=device)

    for step in range(MAX_STEPS):
        s = model.score(x)
        eps = torch.randn_like(x)
        x = x + s * DT + torch.einsum("bij,bj->bi", L, eps) * np.sqrt(2 * DT)
        d_attr = (x.unsqueeze(1) - attr_t.unsqueeze(0)).norm(dim=-1)
        min_d, nearest = d_attr.min(dim=1)
        hit = (min_d < HIT_RADIUS) & (absorbed == -1)
        absorbed[hit] = nearest[hit]
        fpt[hit] = step
        if (absorbed != -1).all():
            break

    # Final-state committor: which attractor closest at MAX_STEPS
    d_attr_final = (x.unsqueeze(1) - attr_t.unsqueeze(0)).norm(dim=-1)
    final_nearest = d_attr_final.argmin(dim=1)

    absorbed = absorbed.cpu().numpy().reshape(n_seed, N_TRAJ)
    fpt = fpt.cpu().numpy().reshape(n_seed, N_TRAJ)
    final_nearest = final_nearest.cpu().numpy().reshape(n_seed, N_TRAJ)
    return absorbed, fpt, final_nearest

# --- Iterate over states ---
state_defs = pd.read_csv(OUTP5 / "state_definitions.csv")
# Filter to states with ≥ 30 real cells (smaller states get inherited
# fingerprints; computing MFPT on 6 cells × 256 trajectories isn't more
# informative than 1 cell × 1500 trajectories).
state_defs_eval = state_defs[state_defs["n_real"] >= 30].reset_index(drop=True)
print(f"[p5_02] computing fingerprints on {len(state_defs_eval)} states "
      f"(n_real ≥ 30) out of {len(state_defs)} total")

rows = []
diag_rows = []
t0 = time.time()

for ridx, row in state_defs_eval.iterrows():
    sid = row["state_id"]
    mask = (a.obs["state_id"].astype(str).values == sid) & (~is_syn)
    if mask.sum() == 0:
        continue
    cell_idx = np.where(mask)[0]
    cell_Z = Z_all[cell_idx]   # (n_state, d) Z-normalized

    # Centroid in Z-space
    centroid_Z = cell_Z.mean(axis=0).astype(np.float32)
    # phi + score at centroid
    cent_t = torch.tensor(centroid_Z, device=device, dtype=torch.float32)
    phi_cent = float(model.phi(cent_t.unsqueeze(0)).item())
    score_cent = float(model.score(cent_t.unsqueeze(0)).norm().item())

    # Basin depth: φ(centroid) - φ(basin attractor)
    basin = row["basin"]
    if basin in basin_to_idx:
        b_idx = basin_to_idx[basin]
        depth = phi_cent - float(phi_attractors[b_idx])
    else:
        depth = np.nan

    # Width via Hessian eigenvalues at centroid
    eigs = model.hessian_eig(cent_t).cpu().numpy()
    eigs_pos = np.clip(eigs, 1e-6, None)
    width_logdet = float(np.log(eigs_pos).sum())
    width_minneg = float((eigs < 0).sum())

    # Local D_trace mean over real cells (real → look up in D_local)
    barcodes = a.obs_names.values[cell_idx]
    obs_idx = np.array([D_obs_map.get(b, -1) for b in barcodes])
    obs_idx = obs_idx[obs_idx >= 0]
    D_trace = float(np.trace(D_norm_global[obs_idx], axis1=1, axis2=2).mean()) if len(obs_idx) else np.nan

    # MFPT / committor — sample SEEDS_PER_STATE source cells (or all if fewer)
    n_avail = len(cell_idx)
    if n_avail == 0:
        continue
    rng = np.random.RandomState(0)
    seed_local_idx = rng.choice(n_avail, min(SEEDS_PER_STATE, n_avail), replace=False)
    seed_Z = cell_Z[seed_local_idx]
    seed_bc = barcodes[seed_local_idx]
    # Cholesky of per-cell D for these seeds; fall back to global mean if missing
    seed_obs_idx = np.array([D_obs_map.get(b, -1) for b in seed_bc])
    seed_D = D_norm_global[seed_obs_idx[seed_obs_idx >= 0]]
    if len(seed_D) < len(seed_Z):
        # pad with mean D
        mean_D = D_norm_global[obs_idx].mean(axis=0) if len(obs_idx) else np.eye(d, dtype=np.float32) * 0.1
        pad = np.broadcast_to(mean_D, (len(seed_Z) - len(seed_D), d, d)).copy()
        seed_D = np.concatenate([seed_D, pad], axis=0)
    L_chol = np.linalg.cholesky(seed_D + 0.005 * np.eye(d, dtype=np.float32)[None])

    absorbed, fpt, final_nearest = run_mfpt(seed_Z, L_chol, attr_t)
    # MFPT per absorbing fate
    fate_mfpt = {}
    fate_committor = {}    # final-state-based committor (more robust)
    fate_first_hit = {}    # first-passage-based committor
    for fid, fname in fate_basins.items():
        fate_attr_idx = basin_to_idx[fid]
        absorbed_at_fate = (absorbed == fate_attr_idx)
        if absorbed_at_fate.any():
            mean_fpt = float(fpt[absorbed_at_fate].mean()) * DT
        else:
            mean_fpt = float(MAX_STEPS * DT)
        # Committor via final state (closest attractor after MAX_STEPS)
        committor = float((final_nearest == fate_attr_idx).mean())
        first_hit = float(absorbed_at_fate.mean())
        fate_mfpt[fname] = round(mean_fpt, 2)
        fate_committor[fname] = round(committor, 3)
        fate_first_hit[fname] = round(first_hit, 3)
    censored_frac = float((absorbed == -1).mean())

    rows.append({
        "state_id": sid,
        "basin": basin,
        "n_real": int(mask.sum()),
        "centroid_phi": round(phi_cent, 3),
        "centroid_score_norm": round(score_cent, 3),
        "basin_depth": round(depth, 3) if not np.isnan(depth) else None,
        "width_logdet": round(width_logdet, 3),
        "hessian_n_negative_eigs": int(width_minneg),
        "D_trace_mean": round(D_trace, 3) if not np.isnan(D_trace) else None,
        "MFPT_to_mature_monocyte": fate_mfpt["mature_monocyte"],
        "MFPT_to_primitive_blast": fate_mfpt["primitive_blast"],
        "committor_to_mature_monocyte": fate_committor["mature_monocyte"],
        "committor_to_primitive_blast": fate_committor["primitive_blast"],
        "first_hit_mature_monocyte": fate_first_hit["mature_monocyte"],
        "first_hit_primitive_blast": fate_first_hit["primitive_blast"],
        "censored_frac": round(censored_frac, 3),
    })
    diag_rows.append({"state_id": sid, "seed_barcodes": ";".join(seed_bc[:8].tolist())})

    if ridx % 5 == 0:
        elapsed = time.time() - t0
        print(f"  {ridx+1}/{len(state_defs_eval)} {sid}: depth={depth:.2f} "
              f"MFPT(A0)={fate_mfpt['mature_monocyte']:.1f} "
              f"committor(A0)={fate_committor['mature_monocyte']:.2f} "
              f"censored={censored_frac:.2f}  [{elapsed:.0f}s]")

df = pd.DataFrame(rows)

# Inherit basin-level numbers for states with n_real < 30
print(f"\n[p5_02] inheriting basin defaults for small states…")
small_states = state_defs[state_defs["n_real"] < 30].copy()
basin_means = df.groupby("basin").mean(numeric_only=True)
small_rows = []
for _, srow in small_states.iterrows():
    sid = srow["state_id"]
    bid = srow["basin"]
    if bid not in basin_means.index:
        continue
    inherited = basin_means.loc[bid].to_dict()
    inherited["state_id"] = sid
    inherited["basin"] = bid
    inherited["n_real"] = int(srow["n_real"])
    inherited["inherited_from_basin_mean"] = True
    small_rows.append(inherited)
df_small = pd.DataFrame(small_rows)
df["inherited_from_basin_mean"] = False
df_full = pd.concat([df, df_small], ignore_index=True)

df_full.to_csv(OUTP5 / "biophysical_fingerprints.csv", index=False)
print(f"[p5_02] wrote {OUTP5 / 'biophysical_fingerprints.csv'}  "
      f"({len(df_full)} states, {len(df)} computed + {len(df_small)} inherited)")

pd.DataFrame(diag_rows).to_csv(OUTP5 / "mfpt_committor_diag.csv", index=False)

# Quick summary
print("\n  Top 10 by basin depth (most stable states)  ")
print(df.sort_values("basin_depth", ascending=True).head(10)[
    ["state_id","basin","n_real","basin_depth","D_trace_mean",
     "MFPT_to_mature_monocyte","committor_to_mature_monocyte"]].to_string(index=False))
