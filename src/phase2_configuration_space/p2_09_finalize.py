"""Phase 2 finalizer — runs after p2_08 and produces:

  data/processed/van_galen_phase2_complete.h5ad  (final deliverable for Phase 3)
  outputs/phase2_summary.json                    (machine-readable summary)
  outputs/phase2_summary.md                      (human-readable summary)

Verifies the AnnData has all expected columns/keys; if a step was skipped (e.g.
BoneMarrowMap projection was not run because R wasn't available), records the
gap in the summary instead of failing.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUT = ROOT / "outputs"

print("[final] loading van_galen_phase2_complete.h5ad…")
a = sc.read_h5ad(PROC / "van_galen_phase2_complete.h5ad")

required_obs = [
    "sample_id", "patient_id", "timepoint", "treatment_status", "disease_state",
    "n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo",
    "S_score", "G2M_score", "phase",
    "celltypist_label", "celltypist_majority",
    "cnv_score", "cnv_proxy_call", "primary_malignant_call",
    "D_local_trace", "D_local_anisotropy",
    "LSC17_score", "LSC6_score", "LSC_surface_score",
]
optional_obs = ["Author Cell Type", "CellType", "PredictionRefined",
                "bmm_predicted_state", "bmm_confidence", "bmm_pseudotime"]
required_obsm = ["X_scVI", "X_scVI_d30", "X_scVI_d20", "X_scVI_d40",
                 "X_umap"]
present_obs = [c for c in required_obs if c in a.obs.columns]
missing_obs = [c for c in required_obs if c not in a.obs.columns]
present_optional = [c for c in optional_obs if c in a.obs.columns]
present_obsm = [c for c in required_obsm if c in a.obsm]
missing_obsm = [c for c in required_obsm if c not in a.obsm]

# Top markers per cluster (Leiden)
print("[final] running Leiden + rank_genes_groups for sanity…")
import warnings
warnings.filterwarnings("ignore")
sc.tl.leiden(a, resolution=1.0, key_added="leiden_r1")
n_clu = a.obs["leiden_r1"].nunique()
sc.tl.rank_genes_groups(a, "leiden_r1", method="wilcoxon", n_genes=20, use_raw=False)

# Save top-5 markers per cluster
top5 = pd.DataFrame(a.uns["rank_genes_groups"]["names"]).head(5).T
top5.columns = [f"gene_{i+1}" for i in range(5)]
top5.to_csv(OUT / "phase2_cluster_top5_markers.csv")

summary = {
    "n_cells": int(a.n_obs),
    "n_genes": int(a.n_vars),
    "n_patients": int(a.obs["patient_id"].nunique()),
    "n_samples": int(a.obs["sample_id"].nunique()),
    "obsm_keys": list(a.obsm.keys()),
    "obs_columns_n": len(a.obs.columns),
    "obs_present_required": present_obs,
    "obs_missing_required": missing_obs,
    "obs_present_optional": present_optional,
    "obsm_present": present_obsm,
    "obsm_missing": missing_obsm,
    "leiden_n_clusters": int(n_clu),
    "disease_state": a.obs["disease_state"].value_counts().to_dict(),
    "primary_malignant_call": (
        a.obs["primary_malignant_call"].value_counts().to_dict()
        if "primary_malignant_call" in a.obs else {}),
    "cell_cycle_phase": (
        a.obs["phase"].value_counts().to_dict()
        if "phase" in a.obs else {}),
    "patients_with_longitudinal_samples": sorted([
        p for p, g in a.obs.groupby("patient_id", observed=False)
        if g["timepoint"].nunique() > 1 and p != "cell_line"
    ]),
    "n_total_obs_columns": len(a.obs.columns),
}
with open(OUT / "phase2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(OUT / "phase2_summary.md", "w") as f:
    f.write("# Phase 2 — final deliverable summary\n\n")
    f.write(f"- **n_cells**: {summary['n_cells']:,}\n")
    f.write(f"- **n_genes**: {summary['n_genes']:,}\n")
    f.write(f"- **n_patients (primary cohort)**: {summary['n_patients']}\n")
    f.write(f"- **n_samples (primary cohort)**: {summary['n_samples']}\n")
    f.write(f"- **disease_state**: {summary['disease_state']}\n")
    f.write(f"- **primary_malignant_call**: {summary['primary_malignant_call']}\n")
    f.write(f"- **cell_cycle_phase**: {summary['cell_cycle_phase']}\n")
    f.write(f"- **Leiden r=1.0 clusters**: {summary['leiden_n_clusters']}\n")
    f.write(f"- **patients_with_longitudinal_samples**: "
            f"{summary['patients_with_longitudinal_samples']}\n\n")
    f.write(f"## obsm keys present\n")
    for k in summary["obsm_keys"]:
        f.write(f"- `{k}`: shape {a.obsm[k].shape}\n")
    f.write(f"\n## obs columns present (required)\n")
    for c in summary["obs_present_required"]:
        f.write(f"- `{c}`\n")
    if summary["obs_missing_required"]:
        f.write(f"\n## obs columns MISSING (required)\n")
        for c in summary["obs_missing_required"]:
            f.write(f"- `{c}` ← gap\n")
    f.write(f"\n## obs columns present (optional)\n")
    for c in summary["obs_present_optional"]:
        f.write(f"- `{c}`\n")

print(f"[final] wrote {OUT / 'phase2_summary.json'}")
print(f"[final] wrote {OUT / 'phase2_summary.md'}")
print(f"\nMissing required obs cols: {missing_obs}")
print(f"Missing required obsm keys: {missing_obsm}")
print(f"\n  Final summary  ")
print(json.dumps(summary, indent=2, default=str))
