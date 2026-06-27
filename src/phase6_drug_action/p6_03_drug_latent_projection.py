"""Phase 6.3 — map drug perturbations into the scVI latent space and project
onto each state's landscape soft-mode.

Two objects per drug:
  Δlatent_drug : the direction the drug pushes cells in the 30-D scVI latent.
                 Computed by perturbing a fixed reference cell population's
                 HVG expression by the drug's z-profile, encoding both
                 baseline and perturbed through scVI, and taking the mean
                 latent shift.

Per state, one object:
  v_soft       : eigenvector of the Hessian of φ at the state attractor with
                 the SMALLEST eigenvalue (the "softest" / easiest-to-destabilize
                 direction of the basin).

Drug score per (state, drug):
  hessian_projection = <Δlatent_drug_normalized, v_soft>
                       large |projection| → drug acts along the soft mode →
                       maximally destabilizing.

We aggregate the drug's many LINCS signatures (cell line × dose × time) into
one mean z-profile per compound before computing Δlatent.

Outputs:
  data/processed/drug_latent_directions.npy        (n_drug, d) Δlatent per drug
  data/processed/drug_latent_names.csv             drug order
  outputs/phase6/hessian_projection_per_state_drug.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
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
OUTP6.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[p6_03] device: {device}")

# ---- load LINCS AML matrix, collapse to per-compound mean z-profile ----
print("[p6_03] loading LINCS AML matrix…")
mat = pd.read_parquet(PROC / "lincs_aml_zscores.parquet")   # genes × signatures
sig = pd.read_csv(PROC / "lincs_aml_sig_info.csv").set_index("sig_id")
# mean z per compound
pert_of_sig = sig.loc[mat.columns, "pert_iname"].values
comp_df = mat.T.copy()
comp_df["drug"] = pert_of_sig
drug_mean_z = comp_df.groupby("drug").mean().T   # genes × drugs
print(f"[p6_03] {drug_mean_z.shape[1]} unique compounds, {drug_mean_z.shape[0]} genes")

# ---- load scVI model + HVG AnnData ----
print("[p6_03] loading scVI model…")
a = sc.read_h5ad(PROC / "van_galen_qc.h5ad")
a.X = a.layers["counts"]
sc.pp.filter_genes(a, min_cells=10)
try:
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat_v3",
                                 batch_key="patient_id", subset=False,
                                 layer="counts", span=0.5)
except Exception:
    tmp = a.copy(); sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=3000, flavor="seurat",
                                 batch_key="patient_id", subset=False)
    a.var["highly_variable"] = tmp.var["highly_variable"].values; del tmp
a_hvg = a[:, a.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(a_hvg, layer="counts", batch_key="patient_id")
model = scvi.model.SCVI.load(str(MODELS / "scvi/van_galen_d30"), adata=a_hvg, accelerator="gpu")
hvg_genes = a_hvg.var_names.tolist()
print(f"[p6_03] HVG genes: {len(hvg_genes)}")

# ---- map drug z-profiles onto HVG gene axis ----
lincs_genes = set(drug_mean_z.index)
hvg_in_lincs = [g for g in hvg_genes if g in lincs_genes]
print(f"[p6_03] HVG ∩ LINCS: {len(hvg_in_lincs)}")
hvg_idx_in_lincs = [hvg_genes.index(g) for g in hvg_in_lincs]
# z-profile matrix on HVG axis: (n_drug, n_hvg), zeros for genes not in LINCS
drug_names = drug_mean_z.columns.tolist()
z_hvg = np.zeros((len(drug_names), len(hvg_genes)), dtype=np.float32)
lincs_sub = drug_mean_z.loc[hvg_in_lincs].values.T  # (n_drug, n_overlap)
for j, gi in enumerate(hvg_idx_in_lincs):
    z_hvg[:, gi] = lincs_sub[:, j]

# ---- reference cell population (fixed sample of 500 cells) ----
np.random.seed(0)
ref_idx = np.random.choice(a_hvg.n_obs, 500, replace=False)
ref_counts = a_hvg.layers["counts"][ref_idx]
import scipy.sparse as sp
ref_counts = ref_counts.toarray() if sp.issparse(ref_counts) else np.asarray(ref_counts)
ref_counts = ref_counts.astype(np.float32)

# baseline latent for the reference population
ref_ad = a_hvg[ref_idx].copy()
base_latent = model.get_latent_representation(ref_ad).mean(axis=0)  # (d,)

# Drug perturbation in count space: counts_pert = counts * exp(alpha * z)
ALPHA = 0.3   # perturbation strength (z-scores ~ standardized; 0.3 keeps it gentle)
print(f"[p6_03] computing Δlatent for {len(drug_names)} drugs (alpha={ALPHA})…")

drug_latent_dirs = np.zeros((len(drug_names), base_latent.shape[0]), dtype=np.float32)
B = 50  # drugs per batch of encoder calls
for d0 in range(0, len(drug_names), B):
    d1 = min(d0 + B, len(drug_names))
    for di in range(d0, d1):
        z = z_hvg[di]
        pert = ref_counts * np.exp(ALPHA * z)[None, :]
        pert_ad = ref_ad.copy()
        pert_ad.layers["counts"] = sp.csr_matrix(pert.astype(np.float32))
        pert_ad.X = pert_ad.layers["counts"]
        pert_latent = model.get_latent_representation(pert_ad).mean(axis=0)
        drug_latent_dirs[di] = pert_latent - base_latent
    if d0 % 100 == 0:
        print(f"  {d1}/{len(drug_names)}")

np.save(PROC / "drug_latent_directions.npy", drug_latent_dirs)
pd.Series(drug_names).to_csv(PROC / "drug_latent_names.csv", index=False, header=["drug"])
print(f"[p6_03] saved drug latent directions ({drug_latent_dirs.shape})")

# ---- Hessian soft-mode per state attractor, project drug directions ----
print("[p6_03] computing Hessian soft-modes + projections…")
class PotentialMLP(nn.Module):
    def __init__(self, d, h=256, depth=4):
        super().__init__()
        layers = [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)
    def hessian(self, x):
        x = x.detach().requires_grad_(True)
        g = torch.autograd.grad(self.forward(x.unsqueeze(0)).sum(), x, create_graph=True)[0]
        d = x.shape[0]; H = torch.zeros(d, d, device=x.device)
        for i in range(d):
            H[i] = torch.autograd.grad(g[i], x, retain_graph=(i < d-1))[0]
        return 0.5 * (H + H.T)

ckpt = torch.load(MODELS / "score_based/score_model.pt", weights_only=False, map_location=device)
pot = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
pot.load_state_dict(ckpt["model"]); pot.eval()
mu = np.array(ckpt["mu"], dtype=np.float32); sd = np.array(ckpt["sd"], dtype=np.float32)

# Drug latent directions are in raw X_scVI space; convert to Z-space (divide by sd)
drug_dirs_Z = drug_latent_dirs / sd[None, :]
drug_dirs_Z_norm = drug_dirs_Z / (np.linalg.norm(drug_dirs_Z, axis=1, keepdims=True) + 1e-8)

attractors = np.load(OUTL / "attractors.npy")
cp = pd.read_csv(OUTL / "critical_points.csv")
attr_true = attractors[(cp["kind"] == "attractor").values]

# Soft mode per basin attractor
soft_modes = {}
for i, attr in enumerate(attr_true):
    xt = torch.tensor(attr, device=device, dtype=torch.float32)
    H = pot.hessian(xt).cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(H)
    soft_modes[f"A{i}"] = eigvecs[:, 0]  # smallest eigenvalue eigenvector
    print(f"  basin A{i}: soft-mode eigenvalue = {eigvals[0]:.3f}")

# Project each drug direction onto each basin soft mode
rows = []
state_def = pd.read_csv(OUTP5 / "state_definitions.csv")
state_basin = dict(zip(state_def["state_id"], state_def["basin"]))
for sid, basin in state_basin.items():
    if basin not in soft_modes:
        continue
    v = soft_modes[basin]
    proj = drug_dirs_Z_norm @ v   # (n_drug,)
    for di, drug in enumerate(drug_names):
        rows.append({"state_id": sid, "basin": basin, "drug": drug,
                     "hessian_projection": round(float(proj[di]), 4)})
proj_df = pd.DataFrame(rows)
proj_df.to_csv(OUTP6 / "hessian_projection_per_state_drug.csv", index=False)
print(f"[p6_03] wrote hessian_projection_per_state_drug.csv ({len(proj_df)} rows)")
