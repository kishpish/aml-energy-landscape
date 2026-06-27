"""Phase 3 finalize — sanity-check the landscape AnnData, write summary."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUT = ROOT / "outputs"

a = sc.read_h5ad(PROC / "van_galen_phase3_landscape.h5ad")
print(f"cells: {a.n_obs}, genes: {a.n_vars}")
print(f"obs columns added by Phase 3: "
      f"{[c for c in a.obs.columns if c in ['basin','basin_2nd','basin_dist_to_attractor','basin_margin','phi','score_norm','high_U','basin_edge','near_saddle','rare_category','rare_state_id']]}")

barriers = pd.read_csv(OUT / "landscape/barriers.csv")
critical = pd.read_csv(OUT / "landscape/critical_points.csv")
rare = pd.read_csv(OUT / "landscape/rare_states_catalog.csv")
basin_cross = pd.read_csv(OUT / "landscape/basin_vs_vangalen_celltype.csv", index_col=0)

# Summary
summary = {
    "n_cells": int(a.n_obs),
    "n_attractors": int((critical["kind"] == "attractor").sum()),
    "n_saddles_from_hessian": int((critical["kind"] == "saddle").sum()),
    "n_saddles_via_midpoint_LBFGS": 0,
    "basin_counts": a.obs["basin"].value_counts().to_dict(),
    "rare_category_counts": a.obs["rare_category"].value_counts().to_dict(),
    "n_rare_states": int((rare["n_cells"] > 0).sum()),
    "barriers_min": float(barriers["barrier_i_to_j"].min()),
    "barriers_max": float(barriers["barrier_i_to_j"].max()),
    "barriers_median": float(barriers["barrier_i_to_j"].median()),
    "score_norm_mean_real_cells": float(a.obs["score_norm"].mean()),
    "phi_min": float(a.obs["phi"].min()),
    "phi_max": float(a.obs["phi"].max()),
}

with open(OUT / "phase3_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# Markdown summary
with open(OUT / "phase3_summary.md", "w") as f:
    f.write("# Phase 3 — Empirical Landscape Reconstruction summary\n\n")
    f.write(f"- **n_cells**: {summary['n_cells']:,}\n")
    f.write(f"- **attractors**: {summary['n_attractors']}\n")
    f.write(f"- **saddles (from Hessian inspection)**: {summary['n_saddles_from_hessian']}\n")
    f.write(f"- **basin counts**: {summary['basin_counts']}\n")
    f.write(f"- **rare-category counts**: {summary['rare_category_counts']}\n")
    f.write(f"- **rare states discovered**: {summary['n_rare_states']}\n")
    f.write(f"- **barrier heights**: median {summary['barriers_median']:.2f}, "
            f"range [{summary['barriers_min']:.2f}, {summary['barriers_max']:.2f}]\n\n")
    f.write("## Critical points\n\n")
    f.write(critical[["id", "kind", "phi", "score_norm", "min_eig", "max_eig",
                      "n_negative_eigs"]].to_markdown(index=False) + "\n\n")
    f.write("## Barriers (string method)\n\n")
    f.write(barriers[["from", "to", "barrier_i_to_j", "barrier_j_to_i",
                       "phi_max", "saddle_node"]].to_markdown(index=False) + "\n\n")
    f.write("## Rare states catalog\n\n")
    f.write(rare.to_markdown(index=False) + "\n\n")
    f.write("## Basin × Van Galen cell type crosstab\n\n")
    f.write(basin_cross.to_markdown() + "\n")

print(f"[p3_06] wrote {OUT / 'phase3_summary.json'} and phase3_summary.md")
print(json.dumps(summary, indent=2, default=str))
