"""Phase 8.2 — survival validation in TCGA-LAML.

For each state ssGSEA score:
  A. Univariate Kaplan-Meier: split patients at median score, log-rank test.
  B. Multivariate Cox: state_score + age + FLT3_mut + NPM1_mut.
     (ELN risk / cytogenetics are not in the Xena GDC export; FLT3 and NPM1
      mutation status are derived from the somatic-mutation file.)
  C. C-index improvement: covariates-only Cox vs covariates + state_score.

Outputs:
  outputs/phase8/survival_univariate.csv      KM log-rank per state
  outputs/phase8/survival_multivariate.csv    Cox HR/CI/p per state
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
TCGA = ROOT / "data/raw/tcga_laml"
OUTP8 = ROOT / "outputs/phase8"

# --- scores ---
scores = pd.read_csv(PROC / "tcga_state_ssgsea.csv", index_col=0)
scores.index = [s[:12] for s in scores.index]   # TCGA-AB-2987-03A → TCGA-AB-2987 patient
scores = scores[~scores.index.duplicated(keep="first")]
print(f"[p8_02] scores: {scores.shape}")

# --- survival ---
surv = pd.read_csv(TCGA / "TCGA-LAML.survival.tsv.gz", sep="\t")
surv["patient"] = surv["_PATIENT"]
surv = surv.drop_duplicates("patient").set_index("patient")
print(f"[p8_02] survival: {surv.shape}")

# --- covariates: age + FLT3/NPM1 mutation ---
clin = pd.read_csv(TCGA / "TCGA-LAML.clinical.tsv.gz", sep="\t")
# find a patient id col + age col
pid_col = [c for c in clin.columns if "submitter_id" in c.lower() and "sample" not in c.lower()]
pid_col = pid_col[0] if pid_col else clin.columns[0]
age_col = "age_at_index.demographic" if "age_at_index.demographic" in clin.columns else None
clin["patient"] = clin[pid_col].astype(str).str[:12]
clin_small = clin[["patient"]].copy()
clin_small["age"] = pd.to_numeric(clin[age_col], errors="coerce") if age_col else np.nan
clin_small = clin_small.drop_duplicates("patient").set_index("patient")

# FLT3 / NPM1 mutation status
mut = pd.read_csv(TCGA / "TCGA-LAML.somaticmutation_wxs.tsv.gz", sep="\t")
gene_col = [c for c in mut.columns if c.lower() in ("gene", "gene_symbol", "hugo_symbol")]
samp_col = [c for c in mut.columns if "sample" in c.lower()]
print(f"[p8_02] mutation cols: gene={gene_col}, sample={samp_col}")
if gene_col and samp_col:
    gcol, scol = gene_col[0], samp_col[0]
    mut["patient"] = mut[scol].astype(str).str[:12]
    flt3_pts = set(mut[mut[gcol] == "FLT3"]["patient"])
    npm1_pts = set(mut[mut[gcol] == "NPM1"]["patient"])
else:
    flt3_pts, npm1_pts = set(), set()
print(f"[p8_02] FLT3-mut patients: {len(flt3_pts)}, NPM1-mut: {len(npm1_pts)}")

# --- assemble analysis frame ---
common = scores.index.intersection(surv.index)
print(f"[p8_02] patients with scores + survival: {len(common)}")
base = pd.DataFrame(index=common)
base["OS_time"] = surv.loc[common, "OS.time"]
base["OS_event"] = surv.loc[common, "OS"]
base["age"] = clin_small["age"].reindex(common)
base["FLT3"] = [1 if p in flt3_pts else 0 for p in common]
base["NPM1"] = [1 if p in npm1_pts else 0 for p in common]
base = base.dropna(subset=["OS_time", "OS_event"])
base["age"] = base["age"].fillna(base["age"].median())
print(f"[p8_02] analysis frame: {base.shape}")

# Covariate-only Cox C-index
cov_cols = ["age", "FLT3", "NPM1"]
cph_cov = CoxPHFitter(penalizer=0.1)
df_cov = base[["OS_time","OS_event"] + cov_cols].dropna()
cph_cov.fit(df_cov, "OS_time", "OS_event")
cidx_cov = cph_cov.concordance_index_
print(f"[p8_02] covariate-only C-index: {cidx_cov:.3f}")

# --- per-state KM + Cox ---
uni_rows, multi_rows = [], []
for sid in scores.columns:
    sc_vals = scores.loc[common, sid].reindex(base.index)
    df = base.copy()
    df["score"] = sc_vals.values
    df = df.dropna(subset=["score"])
    if len(df) < 30:
        continue
    # KM: split at median
    med = df["score"].median()
    hi = df[df["score"] > med]; lo = df[df["score"] <= med]
    lr = logrank_test(hi["OS_time"], lo["OS_time"], hi["OS_event"], lo["OS_event"])
    uni_rows.append({"state_id": sid, "logrank_p": lr.p_value,
                     "n_high": len(hi), "n_low": len(lo),
                     "median_OS_high": hi["OS_time"].median(),
                     "median_OS_low": lo["OS_time"].median()})
    # Cox multivariate
    try:
        dfc = df[["OS_time","OS_event","score"] + cov_cols].dropna()
        # z-score the state score for interpretable HR
        dfc["score"] = (dfc["score"] - dfc["score"].mean()) / (dfc["score"].std() + 1e-9)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(dfc, "OS_time", "OS_event")
        hr = float(np.exp(cph.params_["score"]))
        ci_low = float(np.exp(cph.confidence_intervals_.loc["score"].iloc[0]))
        ci_high = float(np.exp(cph.confidence_intervals_.loc["score"].iloc[1]))
        p = float(cph.summary.loc["score", "p"])
        cidx_full = cph.concordance_index_
        multi_rows.append({"state_id": sid, "HR_per_SD": round(hr,3),
                           "HR_CI_low": round(ci_low,3), "HR_CI_high": round(ci_high,3),
                           "cox_p": p, "C_index_full": round(cidx_full,3),
                           "C_index_cov_only": round(cidx_cov,3),
                           "delta_C_index": round(cidx_full - cidx_cov, 4),
                           "independent": (ci_low > 1 or ci_high < 1)})
    except Exception as e:
        pass

uni = pd.DataFrame(uni_rows)
uni["logrank_q"] = multipletests(uni["logrank_p"], method="fdr_bh")[1]
uni = uni.sort_values("logrank_p")
uni.to_csv(OUTP8 / "survival_univariate.csv", index=False)

multi = pd.DataFrame(multi_rows)
multi["cox_q"] = multipletests(multi["cox_p"], method="fdr_bh")[1]
multi = multi.sort_values("cox_p")
multi.to_csv(OUTP8 / "survival_multivariate.csv", index=False)

print(f"\n[p8_02] univariate: {(uni['logrank_q']<0.05).sum()} states with q<0.05 (log-rank)")
print(uni.head(10).to_string(index=False))
print(f"\n[p8_02] multivariate: {(multi['cox_q']<0.05).sum()} states independently "
      f"prognostic (Cox q<0.05); {(multi['independent']).sum()} with CI excluding 1")
print(multi.head(10).to_string(index=False))
print(f"\n[p8_02] max ΔC-index: {multi['delta_C_index'].max():.4f}")
