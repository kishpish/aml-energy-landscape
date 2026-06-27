"""Phase 3.1 — train a score-based model s_θ(x) ≈ ∇log ρ(x) on the 30-D scVI
latent, then use it as the deterministic force in the Langevin SDE for Phase 4.

PARAMETERIZATION: conservative.  We learn a *scalar potential* φ_θ(x) and
define s_θ(x) := -∇_x φ_θ(x).  This guarantees the learned field is a true
gradient, so the energy landscape U(x) = φ_θ(x) is well-defined and
attractors are honest minima of U (not just sinks of a non-gradient field).

OBJECTIVE: multi-noise denoising score matching.  For σ ∈ {0.01, 0.05, 0.1, 0.3},
  L(θ) = E_{x ∼ ρ, x̃ ∼ N(x, σ²I)} ‖ s_θ(x̃) - (x - x̃)/σ² ‖²
Trained simultaneously by sampling σ uniformly at each batch.

Outputs:
  models/score_based/score_model.pt          PyTorch state dict
  models/score_based/training_curve.csv      loss curve per step
  outputs/landscape/score_field_diag.csv     diagnostics: ‖s_θ‖ at every cell
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/score_based"
OUTL = ROOT / "outputs/landscape"
MODELS.mkdir(exist_ok=True, parents=True)
OUTL.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p3_01] device: {device}")

# -----------------------------------------------------------------------------
# 1. Load X_scVI
# -----------------------------------------------------------------------------
print("[p3_01] loading van_galen_phase2_complete.h5ad…")
a = sc.read_h5ad(PROC / "van_galen_phase2_complete.h5ad", backed="r")
Z = a.obsm["X_scVI"][:].astype(np.float32)  # (n, d)
n, d = Z.shape
print(f"[p3_01] X_scVI: ({n}, {d})")
del a

# Z-score per dim so the noise schedule is interpretable
mu = Z.mean(axis=0)
sd = Z.std(axis=0) + 1e-6
Z = (Z - mu) / sd
np.savez(MODELS / "X_scVI_norm.npz", mu=mu, sd=sd)

x = torch.tensor(Z, device=device, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 2. Conservative score model: s_θ(x) = -∇ φ_θ(x)
# -----------------------------------------------------------------------------
class PotentialMLP(nn.Module):
    """Scalar potential φ_θ(x) with SiLU + LayerNorm. Output a single scalar."""
    def __init__(self, d, h=256, depth=4):
        super().__init__()
        layers = [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)

    def score(self, x):
        """s(x) = -∇ φ(x), computed via autograd."""
        x = x.detach().requires_grad_(True)
        phi = self.forward(x).sum()
        g = torch.autograd.grad(phi, x, create_graph=self.training)[0]
        return -g

model = PotentialMLP(d, h=256, depth=4).to(device)
print(f"[p3_01] model: {sum(p.numel() for p in model.parameters()):,} params")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# -----------------------------------------------------------------------------
# 3. Multi-noise denoising score matching
# -----------------------------------------------------------------------------
sigmas = torch.tensor([0.01, 0.05, 0.1, 0.3], device=device)
BATCH = 1024
N_STEPS = 8000
losses = []
t0 = time.time()

print(f"[p3_01] training {N_STEPS} steps, batch={BATCH}, sigmas={sigmas.tolist()}")
for step in range(N_STEPS):
    idx = torch.randint(0, n, (BATCH,), device=device)
    x0 = x[idx]
    s = sigmas[torch.randint(0, len(sigmas), (BATCH,), device=device)].unsqueeze(1)
    eps = torch.randn_like(x0)
    x_noisy = x0 + s * eps

    # Score target = (x0 - x_noisy) / σ²   so s_θ(x_noisy) ≈ ∇ log p_σ(x_noisy)
    target = (x0 - x_noisy) / (s * s)
    pred = model.score(x_noisy)

    # Weight loss by σ² to balance noise levels
    loss = ((pred - target) ** 2).sum(1) * s.squeeze(1) ** 2
    loss = loss.mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())

    if step % 200 == 0:
        elapsed = time.time() - t0
        last100 = float(np.mean(losses[-100:]))
        print(f"  step {step:>5d}/{N_STEPS}  loss(rolling100)={last100:.4f}  [{elapsed:.0f}s]")

print(f"[p3_01] training done in {time.time() - t0:.0f}s")

# -----------------------------------------------------------------------------
# 4. Save model + curve
# -----------------------------------------------------------------------------
torch.save({
    "model": model.state_dict(),
    "d": d, "h": 256, "depth": 4,
    "mu": mu.tolist(), "sd": sd.tolist(),
    "sigmas": sigmas.tolist(),
    "final_loss": float(np.mean(losses[-100:])),
    "n_steps": N_STEPS,
}, MODELS / "score_model.pt")
pd.DataFrame({"step": range(len(losses)), "loss": losses}).to_csv(
    MODELS / "training_curve.csv", index=False)
print(f"[p3_01] saved {MODELS / 'score_model.pt'}")

# -----------------------------------------------------------------------------
# 5. Evaluate ‖s_θ‖ at every real cell (low magnitude = near critical point)
# -----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    # need grad for the autograd score, so re-enable
    pass

print("[p3_01] computing ‖s_θ(x)‖ at every real cell…")
scores_mag = np.zeros(n, dtype=np.float32)
B = 2048
for b0 in range(0, n, B):
    b1 = min(b0 + B, n)
    xb = x[b0:b1].clone()
    s = model.score(xb)
    scores_mag[b0:b1] = s.norm(dim=1).detach().cpu().numpy()

df = pd.DataFrame({"score_norm": scores_mag})
df.to_csv(OUTL / "score_field_diag.csv", index=False)
print(f"[p3_01] score‖s‖ summary: mean={scores_mag.mean():.3f}  "
      f"median={np.median(scores_mag):.3f}  "
      f"p10={np.percentile(scores_mag, 10):.3f}  "
      f"p90={np.percentile(scores_mag, 90):.3f}")
print(f"[p3_01] wrote {OUTL / 'score_field_diag.csv'}")
