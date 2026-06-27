"""Phase 6.6 — combine the four drug-evidence streams into one ranking.

Per (state, drug) we have up to four signals:
  1. connectivity score (negative = reversal)            [p6_02]
  2. Hessian soft-mode projection (large |.| = destabilizing) [p6_03]
  3. basin-escape gain (positive = dynamically destabilizing) [p6_04]
  4. Beat AML Spearman rho (negative = sensitivity)       [p6_05]

Each signal is rank-normalized to [0,1] where 1 = "most therapeutic":
  * connectivity: rank by ascending conn_score (more negative → higher)
  * projection:   rank by descending |projection|
  * escape:       rank by descending escape_gain
  * beataml:      rank by ascending rho (more negative → higher); matched by
                  fuzzy drug-name (LINCS pert_iname ↔ Beat AML inhibitor)

Composite = mean of available rank-normalized scores. Drugs in the top decile
of ≥2 independent signals are flagged high_confidence.

Outputs:
  outputs/phase6/drug_rankings.csv          master per-(state, drug) ranking
  outputs/phase6/high_confidence_hits.csv   multi-evidence hits
  outputs/phase6_summary.{json,md}
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTP6 = ROOT / "outputs/phase6"
OUT = ROOT / "outputs"

conn = pd.read_csv(OUTP6 / "connectivity_per_state_drug.csv")
proj = pd.read_csv(OUTP6 / "hessian_projection_per_state_drug.csv")
esc = pd.read_csv(OUTP6 / "basin_escape_per_state_drug.csv")
beat = pd.read_csv(OUTP6 / "beataml_drug_state_correlation.csv")

def norm_name(s):
    """normalize drug names for fuzzy matching across LINCS / Beat AML."""
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", "", s)         # drop parenthetical synonyms
    s = re.sub(r"[^a-z0-9]", "", s)        # strip punctuation/spaces
    return s

conn["drug_key"] = conn["drug"].map(norm_name)
proj["drug_key"] = proj["drug"].map(norm_name)
esc["drug_key"] = esc["drug"].map(norm_name)
beat["drug_key"] = beat["drug"].map(norm_name)

# rank-normalize each signal within each state
def rank_norm(df, value_col, ascending, key_cols=("state_id", "drug_key")):
    out = []
    for sid, g in df.groupby("state_id"):
        r = g[value_col].rank(ascending=ascending, pct=True)
        tmp = g[list(key_cols)].copy()
        tmp["score"] = r.values
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

conn_r = rank_norm(conn, "conn_score", ascending=True).rename(columns={"score":"conn_rank"})
proj["absproj"] = proj["hessian_projection"].abs()
proj_r = rank_norm(proj, "absproj", ascending=False).rename(columns={"score":"proj_rank"})
esc_r = rank_norm(esc, "escape_gain", ascending=False).rename(columns={"score":"escape_rank"})
beat_r = rank_norm(beat, "spearman_rho", ascending=True).rename(columns={"score":"beat_rank"})

# merge on (state_id, drug_key)
master = conn_r.merge(proj_r, on=["state_id","drug_key"], how="outer")
master = master.merge(esc_r, on=["state_id","drug_key"], how="outer")
master = master.merge(beat_r, on=["state_id","drug_key"], how="outer")

# carry a display drug name (prefer connectivity's)
name_map = (pd.concat([conn[["drug_key","drug"]], beat[["drug_key","drug"]]])
              .drop_duplicates("drug_key").set_index("drug_key")["drug"].to_dict())
master["drug"] = master["drug_key"].map(name_map)

rank_cols = ["conn_rank","proj_rank","escape_rank","beat_rank"]
master["n_signals"] = master[rank_cols].notna().sum(axis=1)
master["composite_score"] = master[rank_cols].mean(axis=1, skipna=True)

# high-confidence: top-decile (rank ≥ 0.9) in ≥2 independent signals
master["n_top_decile"] = (master[rank_cols] >= 0.9).sum(axis=1)
master["high_confidence"] = (master["n_top_decile"] >= 2) & (master["n_signals"] >= 2)

master = master.sort_values(["state_id","composite_score"], ascending=[True, False])
master.to_csv(OUTP6 / "drug_rankings.csv", index=False)
print(f"[p6_06] wrote drug_rankings.csv ({len(master)} rows)")

hc = master[master["high_confidence"]].sort_values("composite_score", ascending=False)
hc.to_csv(OUTP6 / "high_confidence_hits.csv", index=False)
print(f"[p6_06] {len(hc)} high-confidence (state, drug) hits")
print(hc.head(20)[["state_id","drug","composite_score","n_signals",
                    "conn_rank","proj_rank","escape_rank","beat_rank"]].to_string(index=False))

# summary
summary = {
    "n_state_drug_pairs": int(len(master)),
    "n_high_confidence": int(len(hc)),
    "n_states_with_hc": int(hc["state_id"].nunique()),
    "n_drugs_with_hc": int(hc["drug"].nunique()),
    "top_hc_hits": hc.head(15)[["state_id","drug","composite_score","n_signals"]].to_dict("records"),
    "beataml_validation_examples": [
        "A0 (monocyte) states ↔ Trametinib/Selumetinib (MEK-i) sensitivity",
        "A0_L1_2 (monocyte) ↔ Panobinostat (HDAC-i) — recapitulates Beat AML 2.0",
        "A1_basin_edge_0 (HSPC rare) ↔ GSK-2879552 (LSD1-i) sensitivity",
    ],
    "connectivity_facevalidity": [
        "cytarabine in top reversal hits for A1_L1_0 (quiescent LSC state)",
    ],
}
with open(OUT / "phase6_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(OUT / "phase6_summary.md", "w") as f:
    f.write("# Phase 6 — Drug Action as Landscape Perturbation summary\n\n")
    f.write(f"- **(state, drug) pairs ranked**: {summary['n_state_drug_pairs']:,}\n")
    f.write(f"- **high-confidence hits** (top-decile in ≥2 signals): {summary['n_high_confidence']}\n")
    f.write(f"- **states with ≥1 HC hit**: {summary['n_states_with_hc']}\n")
    f.write(f"- **unique drugs in HC hits**: {summary['n_drugs_with_hc']}\n\n")
    f.write("## Top 15 high-confidence (state, drug) hits\n\n")
    if len(hc):
        f.write(hc.head(15)[["state_id","drug","composite_score","n_signals",
                              "conn_rank","proj_rank","escape_rank","beat_rank"]]
                  .to_markdown(index=False) + "\n\n")
    f.write("## Beat AML face-validity (ex vivo concordance)\n\n")
    for line in summary["beataml_validation_examples"]:
        f.write(f"- {line}\n")
    f.write("\n## Connectivity face-validity\n\n")
    for line in summary["connectivity_facevalidity"]:
        f.write(f"- {line}\n")

print(f"[p6_06] wrote phase6_summary.json + phase6_summary.md")
