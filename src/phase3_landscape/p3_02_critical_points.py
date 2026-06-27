"""Phase 3.2 — find attractors (sinks) and saddles of the score field.

Method:
  1. Initialize N_INIT=5000 trial points: half from a random sample of real
     cells (so we start inside the data manifold), half from random noise
     within the data convex hull.
  2. Integrate dx/dt = s_θ(x) deterministically (RK2, adaptive dt=0.05) for
     up to MAX_STEPS=400, stopping when ‖s_θ‖ < TOL.
  3. Cluster the converged terminal points with DBSCAN. Each cluster center
     is an *attractor*.
  4. For saddle finding: at each candidate point along the trajectory where
     the Hessian of φ has a *single negative eigenvalue* and ‖s_θ‖ is small,
     record as candidate saddle. (Saddles are unstable in 1 direction.)

Outputs:
  outputs/landscape/critical_points.csv     attractor + saddle catalog
  outputs/landscape/attractors.npy          (n_attr, d) attractor positions
  outputs/landscape/saddles.npy             (n_saddle, d) saddle positions
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import time

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/score_based"
OUTL = ROOT / "outputs/landscape"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p3_02] device: {device}")

# -----------------------------------------------------------------------------
# 1. Reload model + normalize Z exactly as in p3_01
# -----------------------------------------------------------------------------
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
    def phi(self, x):
        return self.forward(x).detach()
    def hessian_eig(self, x):
        """Return (eigvals, eigvecs) of Hessian of φ at x (single point)."""
        x = x.detach().requires_grad_(True)
        phi = self.forward(x.unsqueeze(0)).sum()
        g = torch.autograd.grad(phi, x, create_graph=True)[0]
        d = x.shape[0]
        H = torch.zeros(d, d, device=x.device)
        for i in range(d):
            H[i] = torch.autograd.grad(g[i], x, retain_graph=(i < d - 1))[0]
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = torch.linalg.eigh(H)
        return eigvals.detach(), eigvecs.detach()

ckpt = torch.load(MODELS / "score_model.pt", weights_only=False, map_location=device)
model = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mu = np.array(ckpt["mu"], dtype=np.float32)
sd = np.array(ckpt["sd"], dtype=np.float32)
print(f"[p3_02] model loaded, d={ckpt['d']}")

# -----------------------------------------------------------------------------
# 2. Initial conditions
# -----------------------------------------------------------------------------
print("[p3_02] loading X_scVI…")
a = sc.read_h5ad(PROC / "van_galen_phase2_complete.h5ad", backed="r")
Z_raw = a.obsm["X_scVI"][:].astype(np.float32)
Z = (Z_raw - mu) / sd
n, d = Z.shape
del a

N_INIT = 5000
np.random.seed(0)
# half from real cells (random sample), half random noise around the data
n_real = N_INIT // 2
n_noise = N_INIT - n_real
real_idx = np.random.choice(n, n_real, replace=False)
real_init = Z[real_idx]
noise_init = Z[np.random.choice(n, n_noise, replace=False)] \
    + 0.4 * np.random.randn(n_noise, d).astype(np.float32)
init = np.concatenate([real_init, noise_init], axis=0)
print(f"[p3_02] {N_INIT} initial conditions")

# -----------------------------------------------------------------------------
# 3. Deterministic flow: RK2 integration
# -----------------------------------------------------------------------------
MAX_STEPS = 400
TOL = 0.5     # ‖s_θ‖ < TOL → converged
DT = 0.05

x = torch.tensor(init, device=device, dtype=torch.float32)

def step_rk2(model, x, dt):
    s1 = model.score(x)
    x_mid = x + 0.5 * dt * s1
    s2 = model.score(x_mid)
    return x + dt * s2

print(f"[p3_02] integrating {N_INIT} trajectories for up to {MAX_STEPS} steps "
      f"(dt={DT}, tol={TOL})…")
t0 = time.time()
converged_mask = torch.zeros(N_INIT, dtype=torch.bool, device=device)
final_x = x.clone()
trajectory_history = []  # for saddle search later

with torch.enable_grad():
    for step in range(MAX_STEPS):
        s = model.score(x)
        norm = s.norm(dim=1)
        newly_converged = (norm < TOL) & (~converged_mask)
        if newly_converged.any():
            final_x[newly_converged] = x[newly_converged]
            converged_mask |= newly_converged
        if converged_mask.all():
            break
        x = step_rk2(model, x, DT)
        if step % 50 == 0:
            print(f"  step {step:>4d}  active {(~converged_mask).sum().item()}  "
                  f"‖s‖_med {norm.median().item():.2f}")

# Anything still moving at the end — accept its current position
final_x[~converged_mask] = x[~converged_mask]
print(f"[p3_02] integration done in {time.time() - t0:.0f}s  converged "
      f"{converged_mask.sum().item()}/{N_INIT}")

final_x_np = final_x.cpu().numpy()
np.save(OUTL / "trajectory_termini.npy", final_x_np)

# -----------------------------------------------------------------------------
# 4. Cluster terminal points → attractors
# -----------------------------------------------------------------------------
print("[p3_02] DBSCAN clustering of terminal points…")
db = DBSCAN(eps=0.2, min_samples=10).fit(final_x_np)
labels = db.labels_
n_clusters = labels.max() + 1
print(f"[p3_02] DBSCAN found {n_clusters} clusters (excluding noise)")

attractors = []
for k in range(n_clusters):
    members = final_x_np[labels == k]
    centroid = members.mean(axis=0)
    attractors.append(centroid)
attractors = np.array(attractors, dtype=np.float32)
np.save(OUTL / "attractors.npy", attractors)
print(f"[p3_02] {len(attractors)} attractors (centroids of DBSCAN clusters)")

# -----------------------------------------------------------------------------
# 5. Verify attractors are real minima of φ (eigenvalues of Hessian > 0)
# -----------------------------------------------------------------------------
print("[p3_02] checking Hessian eigenvalues at each attractor…")
attr_rows = []
for i, c in enumerate(attractors):
    xt = torch.tensor(c, device=device, dtype=torch.float32)
    eigs, _ = model.hessian_eig(xt)
    eigs_np = eigs.cpu().numpy()
    phi_val = model.phi(xt.unsqueeze(0)).item()
    score_norm = model.score(xt.unsqueeze(0)).norm().item()
    n_neg = int((eigs_np < 0).sum())
    attr_rows.append({
        "id": f"A{i}", "kind": "attractor" if n_neg == 0 else "saddle",
        "phi": phi_val, "score_norm": score_norm,
        "min_eig": float(eigs_np.min()), "max_eig": float(eigs_np.max()),
        "n_negative_eigs": n_neg,
        **{f"x{j}": float(c[j]) for j in range(d)}
    })
df_attr = pd.DataFrame(attr_rows)
df_attr.to_csv(OUTL / "critical_points.csv", index=False)
print(f"[p3_02] wrote {OUTL / 'critical_points.csv'}")
print(df_attr[["id", "kind", "phi", "score_norm", "min_eig", "max_eig",
                "n_negative_eigs"]].to_string())

# -----------------------------------------------------------------------------
# 6. Saddle search: between every adjacent attractor pair, search the
#    midpoint region for points with 1 negative Hessian eigenvalue.
# -----------------------------------------------------------------------------
print("[p3_02] searching for saddles between adjacent attractors…")
# Adjacency via nearest neighbors in attractor space
nn_attr = NearestNeighbors(n_neighbors=min(4, len(attractors))).fit(attractors)
_, neigh_idx = nn_attr.kneighbors(attractors)
adj_pairs = set()
for i, row in enumerate(neigh_idx):
    for j in row[1:]:
        adj_pairs.add(tuple(sorted([i, int(j)])))

saddles = []
saddle_rows = []
for (i, j) in adj_pairs:
    mid = 0.5 * (attractors[i] + attractors[j])
    # Try a few perturbed initial positions around midpoint to find a true saddle
    best = None
    best_score = np.inf
    for trial in range(5):
        x0 = mid + 0.1 * np.random.randn(d).astype(np.float32)
        xt = torch.tensor(x0, device=device, dtype=torch.float32, requires_grad=True)
        # Gradient ascent on |s_θ|² then check Hessian
        local_opt = torch.optim.LBFGS([xt], lr=0.1, max_iter=80, line_search_fn='strong_wolfe')
        def closure():
            local_opt.zero_grad()
            s = model.score(xt.unsqueeze(0))
            loss = (s ** 2).sum()
            loss.backward()
            return loss
        try:
            local_opt.step(closure)
        except Exception:
            continue
        x_final = xt.detach().clone()
        s = model.score(x_final.unsqueeze(0)).squeeze(0)
        sn = s.norm().item()
        if sn < best_score:
            best_score = sn
            best = x_final.cpu().numpy()
    if best is None or best_score > 5.0:
        continue
    # Verify Hessian: a true saddle has exactly one negative eigenvalue
    xt = torch.tensor(best, device=device, dtype=torch.float32)
    eigs, _ = model.hessian_eig(xt)
    eigs_np = eigs.cpu().numpy()
    phi_val = model.phi(xt.unsqueeze(0)).item()
    n_neg = int((eigs_np < -0.01).sum())  # small tolerance
    if n_neg >= 1:  # true saddle or higher-order critical point
        saddles.append(best)
        saddle_rows.append({
            "id": f"S{len(saddles)-1}", "between": f"A{i}-A{j}",
            "phi": phi_val, "score_norm": best_score,
            "min_eig": float(eigs_np.min()), "max_eig": float(eigs_np.max()),
            "n_negative_eigs": n_neg,
            **{f"x{k}": float(best[k]) for k in range(d)}
        })

if saddles:
    np.save(OUTL / "saddles.npy", np.array(saddles))
    df_saddle = pd.DataFrame(saddle_rows)
    df_saddle.to_csv(OUTL / "saddles.csv", index=False)
    print(f"[p3_02] {len(saddles)} saddles found")
    print(df_saddle[["id", "between", "phi", "score_norm",
                     "n_negative_eigs"]].to_string())
else:
    np.save(OUTL / "saddles.npy", np.zeros((0, d), dtype=np.float32))
    print("[p3_02] no saddles found at this resolution (try lower DBSCAN eps)")

print(f"[p3_02] DONE — {len(attractors)} attractors, {len(saddles)} saddles")
