"""Phase 9.1 — assemble the final MHDA-URCS catalog.

The Phase 8 final_state_catalog.csv already merges Phases 5–8 per state. Here
we (a) attach the top-3 drug hits per state from Phase 6, (b) write the
catalog as both parquet (typed) and a human-readable markdown of the
HIGH-confidence states, and (c) emit a compact JSON of the headline numbers.

Outputs:
  outputs/catalog.parquet                   typed full catalog
  outputs/catalog_high_confidence.md         human-readable HIGH states
  outputs/catalog_headline.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTP6 = ROOT / "outputs/phase6"
OUTP8 = ROOT / "outputs/phase8"
OUT = ROOT / "outputs"

cat = pd.read_csv(OUTP8 / "final_state_catalog.csv")

# top-3 drugs per state by composite Phase 6 ranking
rankings = pd.read_csv(OUTP6 / "drug_rankings.csv")
top3 = (rankings.sort_values(["state_id","composite_score"], ascending=[True,False])
                 .groupby("state_id").head(3)
                 .groupby("state_id")["drug"]
                 .apply(lambda x: ",".join(map(str, x)))
                 .reset_index().rename(columns={"drug":"top3_drugs"}))
cat = cat.merge(top3, on="state_id", how="left")

# Beat AML best sensitivity hit per state
beat = pd.read_csv(OUTP6 / "beataml_drug_state_correlation.csv")
beat_best = (beat[beat["spearman_rho"] < 0]
              .sort_values("spearman_rho")
              .groupby("state_id").head(1)[["state_id","drug","spearman_rho","q_value"]]
              .rename(columns={"drug":"beataml_top_sensitizer",
                               "spearman_rho":"beataml_rho",
                               "q_value":"beataml_q"}))
cat = cat.merge(beat_best, on="state_id", how="left")

cat.to_parquet(OUT / "catalog.parquet")
cat.to_csv(OUT / "catalog.csv", index=False)
print(f"[p9_01] wrote catalog.parquet + catalog.csv ({cat.shape})")

# human-readable HIGH-confidence table
high = cat[cat["confidence_tier"] == "HIGH"].copy()
cols = ["state_id","basin","top_vangalen_type","top_scoring_atlas_ct",
        "n_real","malignant_frac","basin_depth","committor_to_mature_monocyte",
        "lsc_subtype_phase_tag","best_pathway_term","top3_drugs",
        "beataml_top_sensitizer","HR_per_SD","persistence_log2_PR_median"]
cols = [c for c in cols if c in high.columns]
with open(OUT / "catalog_high_confidence.md", "w") as f:
    f.write("# MHDA-URCS Catalog — HIGH-confidence cell states\n\n")
    f.write(f"{len(high)} HIGH-confidence states (cross-dataset reproducible + "
            f"pathway-enriched + ≥50 real cells).\n\n")
    f.write(high[cols].sort_values("n_real", ascending=False).to_markdown(index=False))
    f.write("\n")
print(f"[p9_01] wrote catalog_high_confidence.md ({len(high)} states)")

headline = {
    "n_states_total": int(len(cat)),
    "n_high_confidence": int((cat["confidence_tier"]=="HIGH").sum()),
    "n_moderate": int((cat["confidence_tier"]=="MODERATE").sum()),
    "n_exploratory": int((cat["confidence_tier"]=="EXPLORATORY").sum()),
    "n_rare_states_augmented": int(cat["is_rare_state"].astype(str).isin(["True","1"]).sum()),
    "lsc_subtypes": ["quiescent (ANGPT1/MALAT1/NEAT1)", "cycling (HMGB1/CENPF)"],
    "catalog_columns": list(cat.columns),
}
with open(OUT / "catalog_headline.json", "w") as f:
    json.dump(headline, f, indent=2, default=str)
print(json.dumps({k:v for k,v in headline.items() if k!='catalog_columns'}, indent=2))
