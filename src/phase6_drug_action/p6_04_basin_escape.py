"""Phase 6.4 — basin-escape simulation under the drug-perturbed landscape.

Drug action is modeled as a constant force added to the score field:
    s_drug(x) = s_θ(x) + λ · Δlatent_drug_normalized      (drift)
This corresponds to a tilted potential U(x) − λ · <Δlatent_drug, x>.

For each (state, drug) candidate we:
  1. seed N_TRAJ Langevin trajectories from real cells of the state,
  2. integrate under s_drug for T steps (drug-exposure window),
  3. record the fraction of trajectories that LEAVE the source basin
     (their nearest attractor changes from the source basin to another).

A high escape fraction = the drug dynamically destabilizes the state.

To keep runtime bounded we evaluate only the *top-K drugs per state* selected
by the union of (a) most negative connectivity, (b) largest |Hessian
projection|. Each candidate runs N_TRAJ=256 trajectories.

Outputs:
  outputs/phase6/basin_escape_per_state_drug.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
OUTL = ROOT / "outputs/landscape"
OUTP5 = ROOT / "outputs/phase5"
OUTP6 = ROOT / "outputs/phase6"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p6_04] device: {device}")

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

ckpt = torch.load(MODELS / "score_based/score_model.pt", weights_only=False, map_location=device)
pot = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device); pot.eval()
mu = np.array(ckpt["mu"], dtype=np.float32); sd = np.array(ckpt["sd"], dtype=np.float32)
d = ckpt["d"]

# drug latent directions (Z-normalized, unit vectors)
drug_dirs = np.load(PROC / "drug_latent_directions.npy")
drug_names = pd.read_csv(PROC / "drug_latent_names.csv")["drug"].tolist()
drug_dirs_Z = drug_dirs / sd[None, :]
drug_dirs_Z = drug_dirs_Z / (np.linalg.norm(drug_dirs_Z, axis=1, keepdims=True) + 1e-8)
drug_idx = {dn: i for i, dn in enumerate(drug_names)}

attractors = np.load(OUTL / "attractors.npy")
cp = pd.read_csv(OUTL / "critical_points.csv")
attr_true = attractors[(cp["kind"] == "attractor").values].astype(np.float32)
attr_t = torch.tensor(attr_true, device=device)

# load cells
a = sc.read_h5ad(PROC / "van_galen_phase5_complete.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True","1","1.0"]).values
Z_all = (a.obsm["X_scVI"].astype(np.float32) - mu) / sd

# candidate (state, drug) pairs: top-8 by connectivity ∪ top-8 by |projection|
conn = pd.read_csv(OUTP6 / "connectivity_per_state_drug.csv")
proj = pd.read_csv(OUTP6 / "hessian_projection_per_state_drug.csv")
state_def = pd.read_csv(OUTP5 / "state_definitions.csv")
state_basin = dict(zip(state_def["state_id"], state_def["basin"]))

# Evaluate escape only for states with ≥ 50 real cells (enough seeds)
eval_states = state_def[state_def["n_real"] >= 50]["state_id"].tolist()
print(f"[p6_04] evaluating basin escape on {len(eval_states)} states")

N_TRAJ = 256
T_STEPS = 240   # ~24h drug-exposure window (arbitrary units)
DT = 0.02
LAMBDA = 8.0    # drug force magnitude (calibrated so escape is sensitive)

def basin_escape(seed_Z, source_basin_idx, drug_vec):
    n_seed = len(seed_Z)
    x = torch.tensor(np.repeat(seed_Z, N_TRAJ // n_seed + 1, axis=0)[:N_TRAJ],
                     device=device, dtype=torch.float32)
    fvec = torch.tensor(drug_vec, device=device, dtype=torch.float32)
    for _ in range(T_STEPS):
        s = pot.score(x) + LAMBDA * fvec
        x = x + s * DT + torch.randn_like(x) * np.sqrt(2 * 0.05 * DT)
    d_attr = (x.unsqueeze(1) - attr_t.unsqueeze(0)).norm(dim=-1)
    final_basin = d_attr.argmin(dim=1).cpu().numpy()
    escaped = (final_basin != source_basin_idx).mean()
    return float(escaped)

rows = []
for sid in eval_states:
    basin = state_basin[sid]
    basin_idx = int(basin[1:])
    mask = (a.obs["state_id"].astype(str).values == sid) & (~is_syn)
    cell_Z = Z_all[mask]
    if len(cell_Z) < 8:
        continue
    rng = np.random.RandomState(0)
    seed_Z = cell_Z[rng.choice(len(cell_Z), min(8, len(cell_Z)), replace=False)]

    # baseline escape (no drug)
    base_escape = basin_escape(seed_Z, basin_idx, np.zeros(d, dtype=np.float32))

    # candidate drugs
    c_top = conn[conn["state_id"] == sid].nsmallest(8, "conn_score")["drug"].tolist()
    p_sub = proj[proj["state_id"] == sid].copy()
    p_sub["absproj"] = p_sub["hessian_projection"].abs()
    p_top = p_sub.nlargest(8, "absproj")["drug"].tolist()
    candidates = list(dict.fromkeys(c_top + p_top))  # unique, preserve order

    for drug in candidates:
        if drug not in drug_idx:
            continue
        esc = basin_escape(seed_Z, basin_idx, drug_dirs_Z[drug_idx[drug]])
        rows.append({"state_id": sid, "basin": basin, "drug": drug,
                     "baseline_escape": round(base_escape, 3),
                     "drug_escape": round(esc, 3),
                     "escape_gain": round(esc - base_escape, 3)})

esc_df = pd.DataFrame(rows)
esc_df.to_csv(OUTP6 / "basin_escape_per_state_drug.csv", index=False)
print(f"[p6_04] wrote basin_escape_per_state_drug.csv ({len(esc_df)} rows)")
# top escape gains
print("\n[p6_04] top escape-gain (state, drug) pairs:")
print(esc_df.nlargest(15, "escape_gain")[
    ["state_id","basin","drug","baseline_escape","drug_escape","escape_gain"]
].to_string(index=False))
