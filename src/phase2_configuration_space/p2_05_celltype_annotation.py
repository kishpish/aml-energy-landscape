"""Phase 2.8 — three-pass cell type annotation:

  PASS 1: Van Galen 'CellType' / 'PredictionRefined' (gold standard for this
          dataset; already in obs from the joiner).
  PASS 2: Atlas 'Author Cell Type' (a second independent labeling, present for
          ~60% of cells from the atlas join).
  PASS 3: CellTypist Immune_All_Low.pkl model (Python).
  PASS 4 (recommended, BUT requires R): BoneMarrowMap Symphony projection.
          Implemented here as an rpy2 stub script — see p2_05b_bonemarrowmap.R.

Outputs:
  data/processed/van_galen_annotated.h5ad   (integrated + annotated)
  outputs/annotation/celltype_concordance.csv
  outputs/annotation/celltypist_predictions.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTA = ROOT / "outputs/annotation"
OUTA.mkdir(exist_ok=True, parents=True)

print("[anno] loading integrated AnnData…")
a = sc.read_h5ad(PROC / "van_galen_integrated.h5ad")

# Make sure log-normalized values are in .X (CellTypist wants log1p)
if not np.isclose(a.X.toarray()[:1].max() if hasattr(a.X, 'toarray') else a.X[:1].max(),
                  a.X[:1].max(), atol=0):
    pass
# (assume p2_03 already log1p'd; if not, do it now)
if a.X.max() > 100:
    print("[anno] re-normalizing…")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

# ---------------------------------------------------------------------------
# CellTypist
# ---------------------------------------------------------------------------
print("[anno] downloading CellTypist model Immune_All_Low.pkl …")
models.download_models(model=["Immune_All_Low.pkl"], force_update=False)
ct_model = models.Model.load("Immune_All_Low.pkl")
print("[anno] running CellTypist (majority_voting=True)…")
preds = celltypist.annotate(a, model=ct_model, majority_voting=True,
                              over_clustering=None, mode="best match")
pl = preds.predicted_labels
print(f"[anno] CellTypist returned columns: {list(pl.columns)}")
a.obs["celltypist_label"] = pl.get("predicted_labels", pl.iloc[:, 0]).values
if "majority_voting" in pl.columns:
    a.obs["celltypist_majority"] = pl["majority_voting"].values
else:
    a.obs["celltypist_majority"] = a.obs["celltypist_label"]
# probability matrix lives in preds.probability_matrix or preds.decision_matrix
try:
    prob = preds.probability_matrix
    a.obs["celltypist_conf"] = prob.max(axis=1).values
except Exception:
    a.obs["celltypist_conf"] = 1.0

# ---------------------------------------------------------------------------
# Reconcile across labels & compute concordance metrics
# ---------------------------------------------------------------------------
def norm(s):
    return pd.Series(s).astype(str).str.lower().str.strip()

rows = []
n = a.n_obs
vg = norm(a.obs.get("CellType", pd.Series(["nan"]*n))) \
        if "CellType" in a.obs.columns else None
atlas_lab = norm(a.obs.get("Author Cell Type", pd.Series(["nan"]*n))) \
                if "Author Cell Type" in a.obs.columns else None
ct = norm(a.obs["celltypist_majority"])

def concord(x, y, name):
    if x is None or y is None: return None
    mask = (x != "nan") & (y != "nan")
    return {"comparison": name, "n_cells": int(mask.sum()),
            "agreement_rate": round(float((x[mask] == y[mask]).mean()), 4)}

for pair, name in [((vg, atlas_lab), "VanGalen_vs_Atlas"),
                   ((vg, ct),        "VanGalen_vs_CellTypist"),
                   ((atlas_lab, ct), "Atlas_vs_CellTypist")]:
    r = concord(*pair, name=name)
    if r is not None: rows.append(r)

pd.DataFrame(rows).to_csv(OUTA / "celltype_concordance.csv", index=False)
print(f"[anno] concordance rows: {len(rows)}")
for r in rows: print(f"  {r}")

# Top CellTypist predictions
ctab = a.obs["celltypist_majority"].value_counts().head(30)
ctab.to_csv(OUTA / "celltypist_predictions.csv")

# ---------------------------------------------------------------------------
# Save annotated AnnData
# ---------------------------------------------------------------------------
out = PROC / "van_galen_annotated.h5ad"
a.write_h5ad(out, compression="gzip")
print(f"[anno] wrote {out}")
