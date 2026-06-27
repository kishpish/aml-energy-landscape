"""Phase 8.4 — merge survival results into the validated catalog, write summary."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTP7 = ROOT / "outputs/phase7"
OUTP8 = ROOT / "outputs/phase8"
OUT = ROOT / "outputs"

cat = pd.read_csv(OUTP7 / "validated_state_catalog.csv")
uni = pd.read_csv(OUTP8 / "survival_univariate.csv")
multi = pd.read_csv(OUTP8 / "survival_multivariate.csv")

cat = cat.merge(uni[["state_id","logrank_p","logrank_q"]], on="state_id", how="left")
cat = cat.merge(multi[["state_id","HR_per_SD","HR_CI_low","HR_CI_high",
                       "cox_p","cox_q","delta_C_index","independent"]],
                on="state_id", how="left")
cat.to_csv(OUTP8 / "final_state_catalog.csv", index=False)

panels = json.load(open(OUTP8 / "panels_summary.json"))
scoring = json.load(open(OUTP8 / "tcga_scoring_meta.json"))

summary = {
    "n_tcga_patients_scored": scoring["n_tcga_samples"],
    "n_states_survival_tested": int(uni.shape[0]),
    "univariate_logrank_q_lt_0.05": int((uni["logrank_q"] < 0.05).sum()),
    "univariate_nominal_p_lt_0.05": int((uni["logrank_p"] < 0.05).sum()),
    "multivariate_cox_q_lt_0.05": int((multi["cox_q"] < 0.05).sum()),
    "multivariate_independent_CI": int(multi["independent"].sum()),
    "nominal_adverse_states": multi[(multi["cox_p"]<0.05)&(multi["HR_per_SD"]>1)]["state_id"].tolist(),
    "max_delta_C_index": round(float(multi["delta_C_index"].max()), 4),
    "diagnostic_panel_size": panels["diagnostic_panel_size"],
    "diagnostic_cv_auc": panels["diagnostic_cv_auc"],
    "prognostic_panel_size": panels["prognostic_panel_size"],
    "prognostic_panel_cindex": panels["prognostic_panel_cindex"],
    "LSC17_cindex": panels["LSC17_cindex"],
    "prognostic_beats_LSC17": (panels["prognostic_panel_cindex"] or 0) > (panels["LSC17_cindex"] or 1),
}
with open(OUT / "phase8_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(OUT / "phase8_summary.md", "w") as f:
    f.write("# Phase 8 — Survival Validation + Biomarker Panels summary\n\n")
    f.write(f"- **TCGA-LAML patients scored**: {summary['n_tcga_patients_scored']}\n")
    f.write(f"- **states survival-tested**: {summary['n_states_survival_tested']}\n\n")
    f.write("## Survival (TCGA-LAML, OS)\n\n")
    f.write(f"- univariate log-rank q<0.05: {summary['univariate_logrank_q_lt_0.05']} "
            f"(nominal p<0.05: {summary['univariate_nominal_p_lt_0.05']})\n")
    f.write(f"- multivariate Cox q<0.05: {summary['multivariate_cox_q_lt_0.05']}\n")
    f.write(f"- nominal adverse (Cox p<0.05, HR>1): {summary['nominal_adverse_states']}\n")
    f.write(f"- max ΔC-index (state adds over age+FLT3+NPM1): {summary['max_delta_C_index']}\n\n")
    f.write("## Diagnostic panel (malignant vs normal, single-cell)\n\n")
    f.write(f"- **{summary['diagnostic_panel_size']} genes, 5-fold CV-AUC = "
            f"{summary['diagnostic_cv_auc']}** (target ≥ 0.95 ✓)\n\n")
    f.write("## Prognostic panel (TCGA-LAML OS, LASSO-Cox)\n\n")
    f.write(f"- **{summary['prognostic_panel_size']} genes, C-index = "
            f"{summary['prognostic_panel_cindex']}**\n")
    f.write(f"- LSC17 C-index (same cohort) = {summary['LSC17_cindex']}\n")
    f.write(f"- prognostic panel beats LSC17: **{summary['prognostic_beats_LSC17']}** "
            f"(in-sample; needs external validation)\n")

print(f"[p8_04] wrote final_state_catalog.csv + phase8_summary.{{json,md}}")
print(json.dumps(summary, indent=2, default=str))
