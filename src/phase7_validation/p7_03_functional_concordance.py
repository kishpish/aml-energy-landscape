"""Phase 7.3 — functional concordance: do our connectivity drug predictions
agree with Beat AML ex vivo sensitivity?

This is the headline validation number. For each state, we ask:
  Of the drugs our connectivity scoring predicts as therapeutic (top reversal,
  conn_score in bottom decile), what fraction are CONFIRMED by Beat AML as
  sensitivity-associated (negative Spearman rho, q < 0.1)?

Restricted to the 21 drugs present in BOTH LINCS and Beat AML (by normalized
name), since concordance can only be tested where both measure the same drug.

Outputs:
  outputs/phase7/functional_concordance.csv
  outputs/phase7/functional_concordance_summary.json
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTP6 = ROOT / "outputs/phase6"
OUTP7 = ROOT / "outputs/phase7"
OUTP7.mkdir(exist_ok=True, parents=True)

def norm(s):
    s = str(s).lower(); s = re.sub(r"\(.*?\)", "", s); return re.sub(r"[^a-z0-9]", "", s)

conn = pd.read_csv(OUTP6 / "connectivity_per_state_drug.csv")
beat = pd.read_csv(OUTP6 / "beataml_drug_state_correlation.csv")
conn["drug_key"] = conn["drug"].map(norm)
beat["drug_key"] = beat["drug"].map(norm)

# drugs in both
both = set(conn["drug_key"]) & set(beat["drug_key"])
print(f"[p7_03] drugs in both LINCS and Beat AML: {len(both)}")

conn_b = conn[conn["drug_key"].isin(both)].copy()
beat_b = beat[beat["drug_key"].isin(both)].copy()

# For each state, define "predicted therapeutic" = conn_score in bottom 25% within state
rows = []
for sid in conn_b["state_id"].unique():
    cs = conn_b[conn_b["state_id"] == sid]
    if len(cs) < 4:
        continue
    thr = cs["conn_score"].quantile(0.25)
    predicted = set(cs[cs["conn_score"] <= thr]["drug_key"])
    # Beat AML confirmation: negative rho with q<0.25 for this state
    bs = beat_b[beat_b["state_id"] == sid]
    confirmed = set(bs[(bs["spearman_rho"] < 0) & (bs["q_value"] < 0.25)]["drug_key"])
    if not predicted:
        continue
    overlap = predicted & confirmed
    # also compute the background confirmation rate (all 'both' drugs for this state)
    all_drugs = set(cs["drug_key"])
    bg_confirmed = set(bs[(bs["spearman_rho"] < 0) & (bs["q_value"] < 0.25)]["drug_key"]) & all_drugs
    rows.append({
        "state_id": sid,
        "n_predicted": len(predicted),
        "n_confirmed_of_predicted": len(overlap),
        "concordance_rate": round(len(overlap) / len(predicted), 3) if predicted else 0,
        "n_drugs_tested": len(all_drugs),
        "background_confirm_rate": round(len(bg_confirmed) / len(all_drugs), 3) if all_drugs else 0,
        "predicted_confirmed_drugs": ",".join(sorted(overlap)),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTP7 / "functional_concordance.csv", index=False)

overall_conc = df["concordance_rate"].mean() if len(df) else 0
overall_bg = df["background_confirm_rate"].mean() if len(df) else 0
summary = {
    "n_drugs_in_both": len(both),
    "drugs_in_both": sorted(both),
    "n_states_tested": int(len(df)),
    "mean_concordance_rate": round(float(overall_conc), 3),
    "mean_background_confirm_rate": round(float(overall_bg), 3),
    "enrichment_over_background": round(float(overall_conc / overall_bg), 2) if overall_bg > 0 else None,
    "interpretation": (
        "concordance_rate = fraction of connectivity-predicted therapeutics "
        "confirmed by Beat AML negative AUC correlation (q<0.25). "
        "Compare to background_confirm_rate (all tested drugs)."
    ),
}
with open(OUTP7 / "functional_concordance_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"[p7_03] {len(df)} states tested")
print(f"[p7_03] mean concordance rate: {overall_conc:.3f}")
print(f"[p7_03] mean background confirm rate: {overall_bg:.3f}")
print(f"[p7_03] enrichment over background: {summary['enrichment_over_background']}")
print(df.sort_values("concordance_rate", ascending=False).head(15).to_string(index=False))
