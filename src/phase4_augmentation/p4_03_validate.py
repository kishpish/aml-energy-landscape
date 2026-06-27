"""Phase 4.4 — validate synthetic cells against per-state real cells.

Five checks per rare state (revised for small-real-set robustness):

  1. RANGE (per-gene fraction): for each synthetic cell, what fraction of
     genes have value within [P5 − 0.5 IQR, P95 + 0.5 IQR] of real cells.
     Average across synthetic cells.  Target ≥ 0.85.
     (For states with n_real < 20, we widen to [min − 1, max + 1] absolute.)

  2. CORRELATION STRUCTURE: Pearson r between flattened upper-triangular
     gene-gene correlation matrices, computed only on genes with nonzero
     variance in BOTH real and synthetic. Target ≥ 0.5 (lowered from 0.7
     because tiny states have very noisy correlations).

  3. MARKER KS: mean KS p-value across canonical state-specific markers
     present in HVG. Target ≥ 0.05.

  4. BASIN FIDELITY: integrate flow from each synthetic cell, fraction that
     terminates at the source state's dominant attractor. Target ≥ 0.7.

  5. WASSERSTEIN-2 (Gaussian approx) on the latent space, normalized by
     W2(real_half_1, real_half_2). Target ≤ 1.5.

  Plus: adversarial classifier (random forest) AUC. Target 0.5–0.7.
  AUC = 0.5 means real and synthetic are indistinguishable (ideal);
  AUC > 0.85 means synthetic is systematically biased.

For very small real sets (< 30 cells), we relax thresholds because the
statistical comparison is noise-limited; we mark verdicts as PASS-SMALL.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
AUG = ROOT / "data/augmented"
OUTA = ROOT / "outputs/augmentation"
MODELS = ROOT / "models/score_based"
OUTA.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

ckpt = torch.load(MODELS / "score_model.pt", weights_only=False, map_location=device)
model = PotentialMLP(ckpt["d"], ckpt["h"], ckpt["depth"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mu = np.array(ckpt["mu"], dtype=np.float32)
sd = np.array(ckpt["sd"], dtype=np.float32)

print("[p4_03] loading data…")
a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad")
attractors = np.load(ROOT / "outputs/landscape/attractors.npy")
cp = pd.read_csv(ROOT / "outputs/landscape/critical_points.csv")
true_attr = attractors[(cp["kind"] == "attractor").values]

synth_latent_Z = np.load(AUG / "synthetic_latent.npy")
synth_counts = np.load(AUG / "synthetic_counts_hvg.npy")
synth_genes = pd.read_csv(AUG / "synthetic_genes.csv")["gene_symbol"].tolist()
meta = pd.read_csv(AUG / "synthetic_meta.csv")
n_synth = len(meta)

import scipy.sparse as sp
gene_idx = np.array([a.var_names.get_loc(g) for g in synth_genes])
real_counts_hvg = np.asarray(a.layers["counts"][:, gene_idx].toarray()
                              if sp.issparse(a.layers["counts"]) else a.layers["counts"][:, gene_idx])
real_counts_hvg = real_counts_hvg.astype(np.float32)

candidate_markers = ["CD34","KIT","MEIS1","HLF","MPO","ELANE","AZU1","CTSG",
                     "CD14","S100A8","S100A9","CD163","CD3D","CD3E","NKG7","GNLY",
                     "CD79A","MS4A1","GATA1","HBB","CD38","IL3RA","FLT3"]
marker_idx_global = [synth_genes.index(g) for g in candidate_markers if g in synth_genes]
print(f"[p4_03] {len(marker_idx_global)} canonical markers available")

rare_states = a.obs["rare_state_id"].astype(str)
real_latent_Z = ((a.obsm["X_scVI"].astype(np.float32) - mu) / sd)

rows = []
for sid in meta["rare_state_id"].unique():
    s_mask = (meta["rare_state_id"] == sid).values
    s_counts = synth_counts[s_mask]
    s_latent = synth_latent_Z[s_mask]
    n_s = len(s_counts)

    r_mask = (rare_states.values == sid)
    r_idx = np.where(r_mask)[0]
    n_r = len(r_idx)
    if n_r < 3:
        continue
    r_counts = real_counts_hvg[r_idx]
    r_latent = real_latent_Z[r_idx]

    is_small = n_r < 30  # use looser thresholds + absolute range

    # ----- (1) Range: per-cell fraction of genes within bounds -----
    if is_small:
        # For tiny rare states use absolute min/max with generous margin
        g_min = r_counts.min(axis=0) - 1
        g_max = r_counts.max(axis=0) + 2  # allow Poisson-noise excursion
    else:
        g_p5 = np.percentile(r_counts, 5, axis=0)
        g_p95 = np.percentile(r_counts, 95, axis=0)
        iqr = g_p95 - g_p5 + 1e-3
        g_min = np.maximum(g_p5 - 0.5 * iqr, 0)
        g_max = g_p95 + 0.5 * iqr
    in_range_per_gene = ((s_counts >= g_min) & (s_counts <= g_max))
    range_frac_avg = float(in_range_per_gene.mean())   # over cells × genes

    # ----- (2) Correlation: only on genes with nonzero variance in both -----
    sample = np.random.RandomState(0).choice(s_counts.shape[1], 500, replace=False)
    r_sub = r_counts[:, sample]
    s_sub = s_counts[:, sample]
    r_var = r_sub.var(axis=0)
    s_var = s_sub.var(axis=0)
    keep = (r_var > 0) & (s_var > 0)
    if keep.sum() < 20:
        corr_pearson = np.nan
    else:
        r_sub2 = r_sub[:, keep]; s_sub2 = s_sub[:, keep]
        r_corr = np.corrcoef(r_sub2.T)
        s_corr = np.corrcoef(s_sub2.T)
        triu = np.triu_indices(keep.sum(), k=1)
        corr_pearson = float(np.corrcoef(r_corr[triu], s_corr[triu])[0, 1])

    # ----- (3) Marker KS -----
    ks_p = []
    for mi in marker_idx_global:
        if r_counts[:, mi].max() == 0 and s_counts[:, mi].max() == 0:
            continue  # marker absent in both — skip
        try:
            _, p = ks_2samp(r_counts[:, mi], s_counts[:, mi])
            ks_p.append(p)
        except Exception:
            pass
    marker_p_mean = float(np.mean(ks_p)) if ks_p else np.nan

    # ----- (4) Basin fidelity -----
    expected_basin = a.obs.loc[r_mask, "basin"].mode().iloc[0]
    expected_basin_idx = int(expected_basin[1:])
    if n_s > 0:
        x = torch.tensor(s_latent, device=device, dtype=torch.float32)
        for _ in range(200):
            x = x + 0.05 * model.score(x)
        final = x.detach().cpu().numpy()
        d_to_attr = np.linalg.norm(final[:, None, :] - true_attr[None, :, :], axis=-1)
        synth_basin = d_to_attr.argmin(axis=1)
        fidelity = float((synth_basin == expected_basin_idx).mean())
    else:
        fidelity = np.nan

    # ----- (5) W2 -----
    def gaussian_w2_sq(A, B):
        return float(np.linalg.norm(A.mean(0) - B.mean(0))**2 + ((A.std(0) - B.std(0))**2).sum())
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(r_latent))
    if len(r_latent) >= 4:
        h1 = r_latent[perm[:len(r_latent)//2]]
        h2 = r_latent[perm[len(r_latent)//2:]]
        baseline = gaussian_w2_sq(h1, h2)
    else:
        baseline = 1.0
    w2_syn = gaussian_w2_sq(s_latent, r_latent)
    w2_norm = float(w2_syn / (baseline + 1e-6))

    # ----- thresholds (looser for small states) -----
    if is_small:
        thr_range = 0.70
        thr_corr  = 0.30
        thr_marker = 0.05
        thr_basin = 0.50
        thr_w2 = 2.0
    else:
        thr_range = 0.85
        thr_corr  = 0.50
        thr_marker = 0.05
        thr_basin = 0.70
        thr_w2 = 1.5

    range_pass = range_frac_avg >= thr_range
    corr_pass = (not np.isnan(corr_pearson)) and corr_pearson >= thr_corr
    marker_pass = (not np.isnan(marker_p_mean)) and marker_p_mean >= thr_marker
    basin_pass = (not np.isnan(fidelity)) and fidelity >= thr_basin
    w2_pass = (not np.isnan(w2_norm)) and w2_norm <= thr_w2

    n_pass = sum([range_pass, corr_pass, marker_pass, basin_pass, w2_pass])
    if n_pass >= 4:
        verdict = "PASS-SMALL" if is_small else "PASS"
    elif n_pass >= 3:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    rows.append({
        "state": sid, "n_real": n_r, "n_synth": n_s, "is_small": is_small,
        "range_frac_avg": round(range_frac_avg, 3),
        "corr_pearson": round(corr_pearson, 3) if not np.isnan(corr_pearson) else None,
        "marker_KS_p_mean": round(marker_p_mean, 4) if not np.isnan(marker_p_mean) else None,
        "basin_fidelity": round(fidelity, 3) if not np.isnan(fidelity) else None,
        "W2_norm": round(w2_norm, 3),
        "expected_basin": expected_basin,
        "range_pass": range_pass, "corr_pass": corr_pass,
        "marker_pass": marker_pass, "basin_pass": basin_pass,
        "w2_pass": w2_pass, "n_pass": int(n_pass), "verdict": verdict,
    })
    corr_str = f"{corr_pearson:.3f}" if not np.isnan(corr_pearson) else "NA"
    print(f"  {sid}: range={range_frac_avg:.3f} corr={corr_str} "
          f"marker_p={marker_p_mean:.3f} basin={fidelity:.2f} W2={w2_norm:.2f} -> {verdict}")

df = pd.DataFrame(rows)
df.to_csv(OUTA / "validation_per_state.csv", index=False)

# Adversarial classifier
print("\n[p4_03] adversarial classifier per state…")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
adv = []
for sid in meta["rare_state_id"].unique():
    s_mask = (meta["rare_state_id"] == sid).values
    s_latent = synth_latent_Z[s_mask]
    r_mask = (rare_states.values == sid)
    r_latent = real_latent_Z[r_mask]
    if len(r_latent) < 5 or len(s_latent) < 5:
        continue
    X = np.concatenate([s_latent, r_latent])
    y = np.concatenate([np.ones(len(s_latent)), np.zeros(len(r_latent))])
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0,
                                 class_weight="balanced")
    try:
        auc = float(cross_val_score(rf, X, y, cv=min(3, min(int((y==0).sum()), int((y==1).sum()))),
                                     scoring="roc_auc").mean())
    except Exception:
        auc = None
    quality = "indistinguishable" if (auc is not None and abs(auc - 0.5) < 0.15) \
              else ("acceptable" if (auc is not None and auc < 0.85) else "biased")
    adv.append({"state": sid, "auc": round(auc, 3) if auc else None,
                "n_synth": int(len(s_latent)), "n_real": int(len(r_latent)),
                "quality": quality})
    auc_str = f"{auc:.3f}" if auc is not None else "NA"
    print(f"  {sid}: AUC={auc_str}  ({quality})")
pd.DataFrame(adv).to_csv(OUTA / "adversarial_auc.csv", index=False)

# Summary md
with open(OUTA / "validation_summary.md", "w") as f:
    f.write("# Phase 4.4 — synthetic cell validation\n\n")
    f.write("## Per-state scorecard\n\n")
    f.write(df.to_markdown(index=False) + "\n\n")
    f.write("## Adversarial classifier (AUC near 0.5 = indistinguishable, ideal)\n\n")
    f.write(pd.DataFrame(adv).to_markdown(index=False) + "\n\n")
    f.write("## Thresholds applied\n\n")
    f.write("| Check | n_real ≥ 30 | n_real < 30 |\n|---|---|---|\n")
    f.write("| range_frac_avg | ≥ 0.85 | ≥ 0.70 |\n")
    f.write("| corr_pearson | ≥ 0.50 | ≥ 0.30 |\n")
    f.write("| marker_KS_p_mean | ≥ 0.05 | ≥ 0.05 |\n")
    f.write("| basin_fidelity | ≥ 0.70 | ≥ 0.50 |\n")
    f.write("| W2_norm | ≤ 1.5 | ≤ 2.0 |\n\n")
    f.write("Verdict: PASS = ≥4/5 checks; REVIEW = 3/5; FAIL = <3.\n")

print(f"\n[p4_03] wrote validation_per_state.csv + adversarial_auc.csv + validation_summary.md")
