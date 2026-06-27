"""Phase 5.6 — assemble the master state_fingerprints.csv and save the final
Phase 5 deliverable AnnData.

Joins together (per state_id):
  * state_definitions.csv          (n_cells, top cell type, basin)
  * biophysical_fingerprints.csv   (depth, MFPT, committor, D_trace)
  * persistence_per_state.csv      (median_log2_PR, frac_enriched)
  * marker_concordance.csv         (top-5 real markers, real-vs-aug overlap)
  * top3_enrichment_per_state.csv  (best pathway per state)
  * lsc_subtypes.csv               (LSC subtype tag per dominant state)

Output:
  outputs/phase5/state_fingerprints.csv   — the master catalog
  outputs/phase5_summary.{json,md}
  data/processed/van_galen_phase5_complete.h5ad
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(".")
AUG = ROOT / "data/augmented"
PROC = ROOT / "data/processed"
OUTP5 = ROOT / "outputs/phase5"
OUT = ROOT / "outputs"

print("[p5_06] loading source tables…")
def safe_read(p, **kw):
    if not Path(p).exists():
        print(f"  MISSING: {p}")
        return pd.DataFrame()
    return pd.read_csv(p, **kw)

state_def = safe_read(OUTP5 / "state_definitions.csv")
bio = safe_read(OUTP5 / "biophysical_fingerprints.csv")
persist = safe_read(OUTP5 / "persistence_per_state.csv")
concord = safe_read(OUTP5 / "marker_concordance.csv")
enrich = safe_read(OUTP5 / "top3_enrichment_per_state.csv")
lsc = safe_read(OUTP5 / "lsc_subtypes.csv")

# Pull "best enrichment per state" (lowest adj_p_value)
if not enrich.empty:
    best_enr = (enrich.sort_values("adj_p_value")
                       .drop_duplicates("state_id"))[
        ["state_id", "library", "term", "adj_p_value", "overlap"]
    ].rename(columns={
        "library": "best_pathway_library",
        "term": "best_pathway_term",
        "adj_p_value": "best_pathway_adj_p",
        "overlap": "best_pathway_overlap",
    })
else:
    best_enr = pd.DataFrame()

# Marker concordance
if not concord.empty:
    concord = concord[["state_id", "top10_overlap", "spearman_top10",
                        "real_top5"]].rename(columns={"real_top5": "top5_markers"})

# Persistence
if not persist.empty:
    persist = persist[["state_id", "n_patients", "median_log2_PR",
                        "frac_patients_enriched"]].rename(columns={
        "n_patients": "n_patients_longitudinal",
        "median_log2_PR": "persistence_log2_PR_median",
        "frac_patients_enriched": "persistence_frac_enriched",
    })

# Bio fingerprint
keep_bio = ["state_id", "centroid_phi", "centroid_score_norm", "basin_depth",
             "width_logdet", "D_trace_mean",
             "MFPT_to_mature_monocyte", "MFPT_to_primitive_blast",
             "committor_to_mature_monocyte", "committor_to_primitive_blast",
             "censored_frac", "inherited_from_basin_mean"]
keep_bio = [c for c in keep_bio if c in bio.columns]
bio = bio[keep_bio]

# LSC overlay: tag states that contain LSC subtype as dominant_state_id
if not lsc.empty:
    lsc_map = lsc.set_index("dominant_state_id").to_dict("index")
else:
    lsc_map = {}

# Merge
master = state_def.merge(bio, on="state_id", how="left")
master = master.merge(persist, on="state_id", how="left")
master = master.merge(concord, on="state_id", how="left")
master = master.merge(best_enr, on="state_id", how="left")
master["lsc_subtype_dominant"] = master["state_id"].map(
    lambda s: lsc_map.get(s, {}).get("lsc_subtype_id", ""))
master["lsc_subtype_phase_tag"] = master["state_id"].map(
    lambda s: lsc_map.get(s, {}).get("phase_tag", ""))

# Reorder columns
front = ["state_id", "basin", "n_cells_total", "n_real", "n_synth",
          "augmentation_factor", "top_vangalen_type", "top_vangalen_frac",
          "top_atlas_type", "malignant_frac", "n_patients", "is_rare_state"]
back = [c for c in master.columns if c not in front]
master = master[front + back]
master = master.sort_values("n_cells_total", ascending=False)

master.to_csv(OUTP5 / "state_fingerprints.csv", index=False)
print(f"[p5_06] wrote {OUTP5 / 'state_fingerprints.csv'}  ({len(master)} states × {len(master.columns)} cols)")

# Summary
summary = {
    "n_states": int(len(master)),
    "n_states_computed_bio": int((master.get("inherited_from_basin_mean", pd.Series([])) == False).sum()),
    "n_rare_states": int(master["is_rare_state"].astype(str).isin(["True", "1"]).sum()),
    "n_states_longitudinal_observed": int(master["n_patients_longitudinal"].notna().sum())
        if "n_patients_longitudinal" in master.columns else 0,
    "lsc_subtypes": lsc.to_dict("records") if not lsc.empty else [],
    "top_5_by_size": master[["state_id","basin","n_cells_total","top_vangalen_type","malignant_frac"]] \
                       .head(5).to_dict("records"),
    "top_5_persistent": (
        master.dropna(subset=["persistence_log2_PR_median"])
              .nlargest(5, "persistence_log2_PR_median")
              [["state_id","basin","persistence_log2_PR_median","persistence_frac_enriched"]]
              .to_dict("records")
    ),
    "deepest_basin_states": (
        master.dropna(subset=["basin_depth"])
              .nlargest(5, "basin_depth")
              [["state_id","basin","basin_depth","D_trace_mean","committor_to_mature_monocyte"]]
              .to_dict("records")
    ),
}

with open(OUT / "phase5_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(OUT / "phase5_summary.md", "w") as f:
    f.write("# Phase 5 — Dynamical Characterization summary\n\n")
    f.write(f"- **n_states_total**: {summary['n_states']}\n")
    f.write(f"- **states with computed biophysical fingerprints**: {summary['n_states_computed_bio']}\n")
    f.write(f"- **rare states (from Phase 3)**: {summary['n_rare_states']}\n")
    f.write(f"- **states observed longitudinally**: {summary['n_states_longitudinal_observed']}\n")
    f.write(f"- **LSC subtypes discovered**: {len(summary['lsc_subtypes'])}\n\n")

    f.write("## LSC subtypes\n\n")
    if summary["lsc_subtypes"]:
        df_lsc = pd.DataFrame(summary["lsc_subtypes"])
        cols = ["lsc_subtype_id","n_cells","n_real","n_patients","dominant_basin",
                "dominant_state_id","G1_frac","S_frac","G2M_frac","cycling_frac",
                "phase_tag","LSC17_mean","LSC6_mean","D_trace_mean"]
        cols = [c for c in cols if c in df_lsc.columns]
        f.write(df_lsc[cols].to_markdown(index=False) + "\n\n")

    f.write("## Top 5 most-persistent states (median log2 PR, post-treatment)\n\n")
    if summary["top_5_persistent"]:
        f.write(pd.DataFrame(summary["top_5_persistent"]).to_markdown(index=False) + "\n\n")
    f.write("## Top 5 deepest-basin states\n\n")
    if summary["deepest_basin_states"]:
        f.write(pd.DataFrame(summary["deepest_basin_states"]).to_markdown(index=False) + "\n\n")
    f.write("## Top 5 largest states\n\n")
    f.write(pd.DataFrame(summary["top_5_by_size"]).to_markdown(index=False) + "\n")

print(f"[p5_06] wrote {OUT / 'phase5_summary.json'} and phase5_summary.md")

# Final deliverable AnnData
print("[p5_06] saving final Phase 5 AnnData…")
a = sc.read_h5ad(AUG / "van_galen_phase5_lsc.h5ad")
a.write_h5ad(PROC / "van_galen_phase5_complete.h5ad", compression="gzip")
print(f"[p5_06] wrote {PROC / 'van_galen_phase5_complete.h5ad'}")

print(json.dumps(summary, indent=2, default=str)[:3000])
