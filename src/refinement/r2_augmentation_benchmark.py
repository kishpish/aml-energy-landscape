"""R2 — does Langevin augmentation actually help? A falsifiable downsampling
recovery benchmark (the paper's central methods claim).

DESIGN (ground-truth recovery):
  * Pick well-populated states (≥ 400 real cells) as ground truth.
  * "Ground-truth markers" = top-30 up-genes from Wilcoxon DE on ALL real
    cells of the state vs the rest.
  * For each small sample size n ∈ {10, 20, 40}:
      For R replicates:
        - draw n real cells of the state (the "observed rare state")
        - REAL-ONLY arm: DE using just those n cells vs rest → top-30 markers
        - AUGMENTED arm: generate 5n synthetic cells via Langevin from those n
          seeds, decode, DE using n+5n cells vs rest → top-30 markers
      Score each arm by Jaccard overlap of its top-30 with the ground truth.
  * If augmentation helps, AUGMENTED Jaccard > REAL-ONLY Jaccard at small n,
    with a paired test across replicates × states.

This is honest: it can show augmentation helps, hurts, or does nothing, and we
report whichever it is.

Output: outputs/phase4/augmentation_benchmark.csv + summary json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.stats import rankdata, wilcoxon
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
OUTL = ROOT / "outputs/landscape"
OUTP4 = ROOT / "outputs/phase4"
OUTP4.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ---- score model (for Langevin) ----
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
mu = np.array(ckpt["mu"], np.float32); sd = np.array(ckpt["sd"], np.float32)
d = ckpt["d"]

# ---- scVI model + HVG AnnData (for decoding) ----
print("[r2] loading scVI + HVG AnnData…")
a = sc.read_h5ad(PROC / "van_galen_qc.h5ad")
a.X = a.layers["counts"]; sc.pp.filter_genes(a, min_cells=10)
try:
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat_v3",
                                 batch_key="patient_id", subset=False, layer="counts", span=0.5)
except Exception:
    tmp=a.copy(); sc.pp.normalize_total(tmp,target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp,n_top_genes=3000,flavor="seurat",batch_key="patient_id",subset=False)
    a.var["highly_variable"]=tmp.var["highly_variable"].values; del tmp
a_hvg = a[:, a.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(a_hvg, layer="counts", batch_key="patient_id")
model = scvi.model.SCVI.load(str(MODELS/"scvi/van_galen_d30"), adata=a_hvg, accelerator="gpu")
hvg = a_hvg.var_names.tolist()

# ---- log-normalized HVG expression for DE ----
Xc = a_hvg.layers["counts"]
Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
lib = Xc.sum(1)
Xln = np.log1p(Xc / (lib[:,None]+1e-9) * 1e4).astype(np.float32)   # cells × HVG, log-norm
Z = (a_hvg.obsm["X_scVI"].astype(np.float32) - mu) / sd if "X_scVI" in a_hvg.obsm else None
# X_scVI not on a_hvg; recompute latent
if Z is None:
    lat = model.get_latent_representation(a_hvg)
    Z = (lat.astype(np.float32) - mu) / sd

# ---- state labels: bring from phase5 file (aligned by barcode) ----
p5 = sc.read_h5ad(PROC / "van_galen_phase5_complete.h5ad", backed="r")
is_syn = p5.obs["is_synthetic"].astype(str).isin(["True","1","1.0"]).values
state_of = pd.Series(p5.obs["state_id"].astype(str).values[~is_syn],
                     index=p5.obs_names[~is_syn].astype(str))
# align to a_hvg barcodes
common_bc = [b for b in a_hvg.obs_names.astype(str) if b in state_of.index]
bc_to_row = {b:i for i,b in enumerate(a_hvg.obs_names.astype(str))}
state_arr = np.array(["?"]*a_hvg.n_obs, dtype=object)
for b in common_bc:
    state_arr[bc_to_row[b]] = state_of[b]

# well-populated states (≥400 real)
state_def = pd.read_csv(ROOT/"outputs/phase5/state_definitions.csv")
big_states = state_def[(state_def["n_real"]>=400) & (~state_def["is_rare_state"].astype(str).isin(["True","1"]))]["state_id"].tolist()[:8]
print(f"[r2] benchmark states (≥400 real): {big_states}")

def de_topk(mask_state, k=30):
    """Wilcoxon-ish: rank by AUC of state vs rest using mean-rank difference per gene."""
    g_in = Xln[mask_state]; g_out = Xln[~mask_state]
    # use difference of means on log-norm as a fast effect size, restrict to up
    eff = g_in.mean(0) - g_out.mean(0)
    top = np.argsort(eff)[::-1][:k]
    return set(top.tolist())

def langevin_decode(seed_Z, seed_rows, factor=5, tau=20, dt=0.02):
    n = len(seed_Z)
    x = torch.tensor(np.repeat(seed_Z, factor, 0), device=device, dtype=torch.float32)
    for _ in range(tau):
        x = x + pot.score(x)*dt + torch.randn_like(x)*np.sqrt(2*0.05*dt)
    synth_Z = x.detach().cpu().numpy()*sd + mu   # back to X_scVI scale
    # decode via scVI
    src_rows = np.repeat(seed_rows, factor)
    ad = a_hvg[src_rows].copy()
    lat_t = torch.tensor(synth_Z, device=device, dtype=torch.float32)
    libt = torch.tensor(np.log(Xc[src_rows].sum(1)+1).astype(np.float32), device=device)
    bt = torch.tensor(a_hvg.obs["_scvi_batch"].values[src_rows].astype(np.int64), device=device)
    with torch.no_grad():
        out = model.module.decoder(model.module.dispersion, lat_t,
                                    libt.unsqueeze(1), bt.unsqueeze(1))
        # scvi-tools returns (px_scale, px_r, px_rate, px_dropout) or similar;
        # px_rate is the expected counts (largest-magnitude positive tensor).
        out = list(out)
    # px_scale sums to 1 per cell; px_rate ~ library-scaled. Identify px_rate as
    # the entry whose row-sums are ~ library size.
    libnp = Xc[src_rows].sum(1)
    px_rate = None
    for t in out:
        if t is None or not torch.is_tensor(t): continue
        if t.shape == lat_t.shape[:1] + (a_hvg.n_vars,) or (t.dim()==2 and t.shape[1]==a_hvg.n_vars):
            rs = t.sum(1).detach().cpu().numpy()
            if np.median(rs) > 5:   # library-scaled, not the simplex px_scale
                px_rate = t
    if px_rate is None:
        # fallback: use px_scale * library
        px_scale = out[0]
        px_rate = px_scale * torch.tensor(libnp, device=device, dtype=torch.float32).unsqueeze(1)
    mu_g = px_rate.cpu().numpy()
    # dispersion (px_r): the per-gene tensor of shape (n_vars,) or (n, n_vars)
    px_r = model.module.px_r.detach() if hasattr(model.module, "px_r") else None
    if px_r is not None:
        r_np = px_r.cpu().numpy()
        r_g = np.broadcast_to(np.exp(r_np) if r_np.ndim==1 else r_np, mu_g.shape)
    else:
        r_g = np.full_like(mu_g, 10.0)
    r_g = np.clip(r_g, 1e-2, 1e3)
    lam = rng.gamma(r_g, np.clip(mu_g,1e-6,None)/r_g)
    cnt = rng.poisson(lam).astype(np.float32)
    ln = np.log1p(cnt/(cnt.sum(1,keepdims=True)+1e-9)*1e4)
    return ln

rows = []
SIZES = [10, 20, 40]; REPS = 12
for sid in big_states:
    smask_full = (state_arr == sid)
    if smask_full.sum() < 400: continue
    gt = de_topk(smask_full, 30)              # ground-truth markers (all real)
    state_rows = np.where(smask_full)[0]
    rest_ln = Xln[~smask_full]
    for n in SIZES:
        for rep in range(REPS):
            pick = rng.choice(state_rows, n, replace=False)
            # REAL-ONLY arm
            mask_real = np.zeros(a_hvg.n_obs, bool); mask_real[pick] = True
            eff = Xln[pick].mean(0) - Xln[~mask_real].mean(0)
            real_top = set(np.argsort(eff)[::-1][:30].tolist())
            j_real = len(real_top & gt)/len(real_top | gt)
            # AUGMENTED arm
            synth_ln = langevin_decode(Z[pick], pick, factor=5)
            aug_in = np.vstack([Xln[pick], synth_ln])
            # rest = all non-picked real cells
            eff_a = aug_in.mean(0) - Xln[~mask_real].mean(0)
            aug_top = set(np.argsort(eff_a)[::-1][:30].tolist())
            j_aug = len(aug_top & gt)/len(aug_top | gt)
            rows.append({"state":sid, "n":n, "rep":rep,
                         "jaccard_real":j_real, "jaccard_aug":j_aug,
                         "delta":j_aug-j_real})
    print(f"[r2] {sid} done")

df = pd.DataFrame(rows)
df.to_csv(OUTP4/"augmentation_benchmark.csv", index=False)

# paired test per size
summary = {"n_states": len(big_states), "reps_per_cell": REPS}
for n in SIZES:
    sub = df[df["n"]==n]
    jr, ja = sub["jaccard_real"].mean(), sub["jaccard_aug"].mean()
    try:
        stat, p = wilcoxon(sub["jaccard_aug"], sub["jaccard_real"])
    except Exception:
        p = np.nan
    summary[f"n{n}"] = {"jaccard_real_mean": round(float(jr),3),
                        "jaccard_aug_mean": round(float(ja),3),
                        "mean_delta": round(float(sub["delta"].mean()),3),
                        "wilcoxon_p": float(p),
                        "augmentation_helps": bool(ja > jr and p < 0.05)}
with open(OUTP4/"augmentation_benchmark_summary.json","w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n[r2]   AUGMENTATION BENCHMARK  ")
print(json.dumps(summary, indent=2, default=str))
