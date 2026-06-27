"""Phase 7.4 — assign a confidence tier to every cell state and assemble the
final validated catalog.

Tier rubric (per state):
  HIGH        reproduces in scAtlas (exact OR broad lineage) AND has a
              significant pathway enrichment AND ≥50 real cells
  MODERATE    reproduces at broad-lineage level OR significant enrichment,
              but not both; ≥30 real cells
  EXPLORATORY everything else (small states, no clear reproduction)

We merge the Phase 5 state_fingerprints with Phase 7 reproduction + add the
tier. Output is the final cross-validated state catalog.

Outputs:
  outputs/phase7/validated_state_catalog.csv
  outputs/phase7_summary.{json,md}
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTP5 = ROOT / "outputs/phase5"
OUTP7 = ROOT / "outputs/phase7"
OUT = ROOT / "outputs"

fp = pd.read_csv(OUTP5 / "state_fingerprints.csv")
repro = pd.read_csv(OUTP7 / "scatlas_reproduction.csv")

def broad(ct):
    ct = str(ct)
    if ct in ["T","NK","B","CTL","ProB","Plasma"]: return "lymphoid"
    if ct in ["CD14+ Mono","CD16+ Mono","ProMono","cDC","pDC","GMP","CMP"]: return "myeloid"
    if ct in ["Erythroid","MEP","earlyEry","lateEry"]: return "erythroid"
    if ct in ["HSPC","HSC"]: return "stem"
    return "other"

repro["exp_broad"] = repro["expected_atlas_ct"].map(broad)
repro["top_broad"] = repro["top_scoring_atlas_ct"].map(broad)
repro["broad_reproduces"] = repro["exp_broad"] == repro["top_broad"]
repro["enrichment_sig"] = repro["enrichment_p"] < 0.05

cat = fp.merge(
    repro[["state_id","reproduces","broad_reproduces","enrichment_sig",
           "top_scoring_atlas_ct","enrichment_p"]],
    on="state_id", how="left")

def tier(row):
    n_real = row.get("n_real", 0)
    exact = bool(row.get("reproduces", False))
    broad_ok = bool(row.get("broad_reproduces", False))
    enr = bool(row.get("enrichment_sig", False))
    has_pathway = isinstance(row.get("best_pathway_term", None), str) and row.get("best_pathway_term","") not in ("", "nan")
    if (exact or broad_ok) and (enr or has_pathway) and n_real >= 50:
        return "HIGH"
    if (broad_ok or enr or has_pathway) and n_real >= 30:
        return "MODERATE"
    return "EXPLORATORY"

cat["confidence_tier"] = cat.apply(tier, axis=1)
cat.to_csv(OUTP7 / "validated_state_catalog.csv", index=False)

tier_counts = cat["confidence_tier"].value_counts().to_dict()
print(f"[p7_04] confidence tiers: {tier_counts}")

# load other Phase 7 results for the summary
func = json.load(open(OUTP7 / "functional_concordance_summary.json"))
sens = json.load(open(OUTP7 / "sensitivity_summary.json"))

summary = {
    "n_states": int(len(cat)),
    "confidence_tiers": tier_counts,
    "scatlas_exact_reproduce": int(repro["reproduces"].sum()),
    "scatlas_broad_reproduce": int(repro["broad_reproduces"].sum()),
    "scatlas_enrichment_sig": int(repro["enrichment_sig"].sum()),
    "scatlas_total_tested": int(len(repro)),
    "functional_concordance_rate": func["mean_concordance_rate"],
    "functional_background_rate": func["mean_background_confirm_rate"],
    "functional_enrichment": func["enrichment_over_background"],
    "n_drugs_overlap_lincs_beataml": func["n_drugs_in_both"],
    "sensitivity_marker_overlap": sens["A_marker_concordance_mean_top10_overlap"],
    "sensitivity_basin_agreement": sens["B_basin_agreement_phase3_vs_phase5"],
    "sensitivity_lsc_real_only": sens["C_all_lsc_subtypes_real_only"],
    "high_confidence_states": cat[cat["confidence_tier"]=="HIGH"]["state_id"].tolist(),
}
with open(OUT / "phase7_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(OUT / "phase7_summary.md", "w") as f:
    f.write("# Phase 7 — Multi-Level Validation summary\n\n")
    f.write(f"- **states**: {summary['n_states']}\n")
    f.write(f"- **confidence tiers**: {tier_counts}\n\n")
    f.write("## Cross-dataset reproduction (AML scAtlas, 748k cells)\n\n")
    f.write(f"- exact cell-type match: {summary['scatlas_exact_reproduce']}/{summary['scatlas_total_tested']}\n")
    f.write(f"- broad-lineage match: {summary['scatlas_broad_reproduce']}/{summary['scatlas_total_tested']}\n")
    f.write(f"- significant enrichment (p<0.05): {summary['scatlas_enrichment_sig']}/{summary['scatlas_total_tested']}\n\n")
    f.write("## Functional concordance (LINCS predictions vs Beat AML ex vivo)\n\n")
    f.write(f"- drugs in both resources: {summary['n_drugs_overlap_lincs_beataml']}\n")
    f.write(f"- mean concordance rate: {summary['functional_concordance_rate']}\n")
    f.write(f"- background rate: {summary['functional_background_rate']}\n")
    f.write(f"- enrichment over background: {summary['functional_enrichment']}× "
            f"(near 1.0 = connectivity does NOT strongly predict ex vivo cytotoxicity "
            f"in the 21 overlapping drugs — honest null)\n\n")
    f.write("## Robustness to synthetic cells\n\n")
    f.write(f"- real-vs-augmented marker overlap: {summary['sensitivity_marker_overlap']}/10\n")
    f.write(f"- basin agreement Phase3↔Phase5: {summary['sensitivity_basin_agreement']*100:.0f}%\n")
    f.write(f"- all LSC subtypes real-only: {summary['sensitivity_lsc_real_only']}\n\n")
    f.write("## HIGH-confidence states\n\n")
    hc = cat[cat["confidence_tier"]=="HIGH"][
        ["state_id","basin","top_vangalen_type","top_scoring_atlas_ct",
         "best_pathway_term","n_real"]]
    f.write(hc.to_markdown(index=False) + "\n")

print(f"[p7_04] wrote validated_state_catalog.csv + phase7_summary.{{json,md}}")
print(json.dumps(summary, indent=2, default=str)[:2000])
