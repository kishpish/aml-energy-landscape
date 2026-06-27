"""Phase 5.5 — diagnosis vs post-treatment persistence ratios.

For each AML patient with paired pre-treatment and post-treatment samples,
compute per-state composition fractions:

  f_pre(s)  = n_cells(state=s, patient=p, timepoint=D0) / n_cells(patient=p, D0)
  f_post(s) = n_cells(state=s, patient=p, timepoint=post) / n_cells(patient=p, post)

Persistence ratio per (patient, state) = f_post(s) / f_pre(s)
  > 1   state is *enriched* after treatment   → resistant / persister
  ≈ 1   state is *preserved* through treatment
  < 1   state is *depleted*                  → sensitive

Then aggregate per state across patients:
  median persistence ratio
  fraction of patients with PR > 1

The biophysics prediction we test:
  PR is positively correlated with basin_depth (deep basins persist).

Outputs:
  outputs/phase5/persistence_per_patient_state.csv
  outputs/phase5/persistence_per_state.csv
  outputs/phase5/persistence_vs_depth.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
AUG = ROOT / "data/augmented"
OUTP5 = ROOT / "outputs/phase5"

print("[p5_05] loading LSC-tagged AnnData…")
a = sc.read_h5ad(AUG / "van_galen_phase5_lsc.h5ad", backed="r")
print(f"[p5_05] {a.shape}")

# Use REAL cells only (synthetic cells inherit timepoint from source so they
# would double-count; the persistence question is fundamentally about *real*
# longitudinal sample composition).
obs = a.obs.copy()
is_syn = obs["is_synthetic"].astype(str).isin(["True", "1", "1.0"]).values
obs_real = obs[~is_syn].copy()
print(f"[p5_05] real cells: {len(obs_real)}")

# Use AML patients only (drop healthy donors)
obs_real = obs_real[obs_real["disease_state"].astype(str).isin(
    ["AML_diagnosis", "AML_residual_or_relapse"])].copy()
print(f"[p5_05] AML cells: {len(obs_real)}")

# Identify patients with both diagnosis (pre) AND post-treatment samples
patient_ts = (obs_real.groupby(["patient_id", "treatment_status"], observed=False)
              .size().unstack(fill_value=0))
print(f"[p5_05] patient × treatment_status:\n{patient_ts}")
longitudinal = patient_ts[(patient_ts.get("pre_treatment", 0) > 0) &
                          (patient_ts.get("post_chemo", 0) > 0)].index.tolist()
print(f"[p5_05] {len(longitudinal)} longitudinal patients: {longitudinal}")

if not longitudinal:
    print("[p5_05] No longitudinal patients found — skipping.")
    raise SystemExit(0)

# Per-(patient, state) ratio
rows = []
for pat in longitudinal:
    pre = obs_real[(obs_real["patient_id"] == pat) &
                   (obs_real["treatment_status"] == "pre_treatment")]
    post = obs_real[(obs_real["patient_id"] == pat) &
                    (obs_real["treatment_status"] == "post_chemo")]
    n_pre = len(pre); n_post = len(post)
    pre_counts = pre["state_id"].astype(str).value_counts()
    post_counts = post["state_id"].astype(str).value_counts()
    all_states = set(pre_counts.index) | set(post_counts.index)
    for s in all_states:
        f_pre = pre_counts.get(s, 0) / max(n_pre, 1)
        f_post = post_counts.get(s, 0) / max(n_post, 1)
        if f_pre > 0 or f_post > 0:
            # Use pseudocount (1 cell) to handle zeros robustly:
            #   f_pre_eps = (n_pre + 1) / (n_total_pre + n_states)
            #   f_post_eps = (n_post + 1) / (n_total_post + n_states)
            # log2_PR := log2(f_post_eps / f_pre_eps)
            eps_pre = (pre_counts.get(s, 0) + 1) / (n_pre + len(all_states))
            eps_post = (post_counts.get(s, 0) + 1) / (n_post + len(all_states))
            log2_pr = float(np.log2(eps_post / eps_pre))
            rows.append({
                "patient_id": pat,
                "state_id": s,
                "n_pre": int(pre_counts.get(s, 0)),
                "n_post": int(post_counts.get(s, 0)),
                "f_pre": round(f_pre, 4),
                "f_post": round(f_post, 4),
                "log2_PR_pseudo": round(log2_pr, 4),
            })
df_pp = pd.DataFrame(rows)
df_pp.to_csv(OUTP5 / "persistence_per_patient_state.csv", index=False)
print(f"[p5_05] wrote per-(patient, state) ratios ({len(df_pp)} rows)")

# Aggregate per state across patients (use log2_PR_pseudo for robustness)
print("\n[p5_05] aggregating per state across patients (log2 PR with pseudocounts)…")
agg = (df_pp.groupby("state_id")
         .agg(n_patients=("patient_id", "nunique"),
              median_log2_PR=("log2_PR_pseudo", "median"),
              mean_log2_PR=("log2_PR_pseudo", "mean"),
              n_pre_total=("n_pre", "sum"),
              n_post_total=("n_post", "sum"))
         .reset_index())
agg["frac_patients_enriched"] = df_pp.assign(enr=lambda x: x["log2_PR_pseudo"] > 0) \
                                    .groupby("state_id")["enr"].mean().reindex(agg["state_id"]).values
agg["frac_patients_enriched"] = agg["frac_patients_enriched"].round(3)
agg = agg.sort_values("median_log2_PR", ascending=False)
agg.to_csv(OUTP5 / "persistence_per_state.csv", index=False)
print(agg.head(15).to_string(index=False))

# Test the biophysics prediction: persistence vs basin depth
print("\n[p5_05] testing prediction: PR ∝ basin_depth …")
fp = pd.read_csv(OUTP5 / "biophysical_fingerprints.csv")
joined = agg.merge(fp[["state_id", "basin", "basin_depth", "D_trace_mean"]],
                    on="state_id", how="inner")
joined = joined[joined["n_patients"] >= 3]  # at least 3 patients seeing the state
print(f"[p5_05] {len(joined)} states with ≥3-patient ratios + depth")
if len(joined) >= 5:
    valid = joined[~joined["basin_depth"].isna()]
    if len(valid) >= 5:
        rho, p = spearmanr(valid["basin_depth"], valid["median_log2_PR"])
        print(f"  Spearman(basin_depth, median_log2_PR): rho={rho:.3f}, p={p:.4g}")
        joined["depth_vs_PR_spearman_rho"] = rho
        joined["depth_vs_PR_spearman_p"] = p
joined.to_csv(OUTP5 / "persistence_vs_depth.csv", index=False)
print(f"[p5_05] wrote {OUTP5 / 'persistence_vs_depth.csv'}")

# Also annotate LSC subtypes
print("\n[p5_05] LSC subtype persistence (real cells only)…")
lsc = pd.read_csv(OUTP5 / "lsc_subtypes.csv")
# Per-patient LSC enrichment
lsc_obs = obs_real.copy()
lsc_obs["is_lsc"] = lsc_obs["LSC_high_confidence"].astype(str).isin(["1", "1.0", "True"])
rows = []
for pat in longitudinal:
    pre = lsc_obs[(lsc_obs["patient_id"] == pat) &
                  (lsc_obs["treatment_status"] == "pre_treatment")]
    post = lsc_obs[(lsc_obs["patient_id"] == pat) &
                   (lsc_obs["treatment_status"] == "post_chemo")]
    if len(pre) == 0 or len(post) == 0: continue
    f_pre = lsc_obs.is_lsc.values[pre.index.get_indexer(pre.index)].mean() if hasattr(pre, "index") else 0
    f_pre = pre["is_lsc"].mean()
    f_post = post["is_lsc"].mean()
    rows.append({"patient_id": pat, "n_pre": len(pre), "n_post": len(post),
                 "lsc_frac_pre": round(f_pre, 4), "lsc_frac_post": round(f_post, 4),
                 "lsc_PR": round(f_post / max(f_pre, 1e-4), 4)})
df_lsc_pr = pd.DataFrame(rows)
df_lsc_pr.to_csv(OUTP5 / "lsc_persistence_per_patient.csv", index=False)
print(df_lsc_pr.to_string(index=False))
print(f"\n  median LSC persistence ratio: {df_lsc_pr['lsc_PR'].median():.3f}")
print(f"  patients with LSC enrichment (PR>1): {(df_lsc_pr['lsc_PR'] > 1).sum()}/{len(df_lsc_pr)}")
