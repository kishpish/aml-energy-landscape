"""Phase 3.4 — barrier heights between adjacent attractor pairs via the string
method.

For each pair of attractors (A_i, A_j):
  1. Parameterize a path of K=20 nodes from A_i to A_j (initial: straight line).
  2. Iterate: each interior node moves in the direction of -∇φ projected
     perpendicular to the local path tangent; then re-parameterize the path
     to equal arc length.
  3. The barrier height ΔU(A_i → A_j) = max_t φ(path[t]) − φ(A_i).
  4. The path's max φ is the saddle estimate.

This is the canonical chain-of-states method (Henkelman & Jónsson 2000)
adapted for a learned potential. It is the physically principled way to
estimate Kramers transition rates k_{i→j} ∝ exp(−ΔU/kT).

Outputs:
  outputs/landscape/barriers.csv             pairwise barriers
  outputs/landscape/paths.npz                converged paths
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

ROOT = Path(".")
MODELS = ROOT / "models/score_based"
OUTL = ROOT / "outputs/landscape"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p3_04] device: {device}")

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

# Attractor positions (in normalized Z-space)
attr = np.load(OUTL / "attractors.npy")
cp = pd.read_csv(OUTL / "critical_points.csv")
true_attr_mask = (cp["kind"] == "attractor").values
attr_true = attr[true_attr_mask]
n_attr = len(attr_true)
print(f"[p3_04] {n_attr} true attractors")

K = 20             # path nodes
ITERS = 200        # string-method iterations
DT = 0.05          # node update step

def reparameterize(path):
    """Re-distribute K nodes to equal arc length along the path."""
    d_seg = (path[1:] - path[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=path.device), d_seg.cumsum(0)])
    total = cum[-1]
    target = torch.linspace(0, 1, K, device=path.device) * total
    new_path = torch.zeros_like(path)
    j = 0
    for i in range(K):
        while j < len(cum) - 2 and cum[j + 1] < target[i]:
            j += 1
        if j == len(cum) - 1:
            new_path[i] = path[-1]
        else:
            t = (target[i] - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
            new_path[i] = (1 - t) * path[j] + t * path[j + 1]
    return new_path

rows = []
paths = {}
t0 = time.time()
for i in range(n_attr):
    for j in range(i + 1, n_attr):
        Ai = torch.tensor(attr_true[i], device=device, dtype=torch.float32)
        Aj = torch.tensor(attr_true[j], device=device, dtype=torch.float32)
        # Initialize straight-line path
        path = torch.stack([Ai + (Aj - Ai) * t for t in torch.linspace(0, 1, K, device=device)])
        for it in range(ITERS):
            # Compute score at interior nodes
            interior = path[1:-1].detach().requires_grad_(True)
            s = model.score(interior)
            # Path tangent at each interior node (centered diff)
            tangent = (path[2:] - path[:-2]) * 0.5
            tnorm = tangent.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            tangent = tangent / tnorm
            # Project s perpendicular to tangent
            s_para = (s * tangent).sum(dim=-1, keepdim=True) * tangent
            s_perp = s - s_para
            # Update interior nodes
            new_interior = interior.detach() + DT * s_perp.detach()
            path = torch.cat([Ai.unsqueeze(0), new_interior, Aj.unsqueeze(0)], dim=0)
            # Reparameterize every 5 iters
            if it % 5 == 0:
                path = reparameterize(path)

        # Evaluate φ along final path
        phi_path = model.phi(path).cpu().numpy()
        barrier_ij = float(phi_path.max() - phi_path[0])
        barrier_ji = float(phi_path.max() - phi_path[-1])
        argmax = int(phi_path.argmax())
        rows.append({
            "from": f"A{i}", "to": f"A{j}",
            "barrier_i_to_j": barrier_ij,
            "barrier_j_to_i": barrier_ji,
            "phi_start": float(phi_path[0]),
            "phi_end": float(phi_path[-1]),
            "phi_max": float(phi_path.max()),
            "saddle_node": argmax,
            "path_length": float((path[1:] - path[:-1]).norm(dim=-1).sum().item()),
        })
        paths[f"A{i}_to_A{j}"] = path.detach().cpu().numpy()
        print(f"  A{i}-A{j}: dU_i_to_j={barrier_ij:.2f}  dU_j_to_i={barrier_ji:.2f}  "
              f"saddle@node{argmax}/{K-1}")

print(f"[p3_04] done in {time.time()-t0:.0f}s")
df = pd.DataFrame(rows)
df.to_csv(OUTL / "barriers.csv", index=False)
np.savez(OUTL / "paths.npz", **paths)
print(f"[p3_04] wrote {OUTL / 'barriers.csv'}")
print(df.to_string())
