"""Phase 2.11 — pre-compute per-cell signature scores that Phases 3+ will read
without re-deriving:

  obs['LSC17_score']         Ng et al. 2016 leukemic stem-cell 17-gene score
  obs['LSC6_score']          Elsayed et al. 2020 pediatric LSC6
  obs['LSC_surface_score']   CD34 + KIT + IL3RA - CD38 (immunophenotypic proxy)
  obs['Tirosh_S_score']      already in obs from p2_03 (cell cycle)
  obs['Tirosh_G2M_score']    already in obs from p2_03

  obs['Hallmark_*']          decoupler aucell scores for 50 Hallmark gene sets
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import json
import decoupler as dc
import gseapy as gp

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTS = ROOT / "outputs/signatures"
OUTS.mkdir(exist_ok=True, parents=True)

print("[sig] loading AnnData…")
a = sc.read_h5ad(PROC / "van_galen_diff.h5ad")

# Ensure log-normalized values
if a.X.max() > 100:
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

# ---------------------------------------------------------------------------
# 1. LSC scores via scanpy.score_genes
# ---------------------------------------------------------------------------
sigs = json.load(open(ROOT / "data/raw/msigdb/AML_signatures.json"))

def safe_score(adata, gene_list, key):
    genes = [g for g in gene_list if g in adata.var_names]
    if len(genes) < 3:
        print(f"  [sig] skipping {key}: only {len(genes)} genes present")
        adata.obs[key] = np.nan
        return
    sc.tl.score_genes(adata, gene_list=genes, score_name=key, ctrl_size=max(50, len(genes)*5),
                      use_raw=False)
    print(f"  {key}: {len(genes)}/{len(gene_list)} genes  mean={adata.obs[key].mean():.3f}")

safe_score(a, sigs["LSC17_Ng2016"], "LSC17_score")
safe_score(a, sigs["LSC6_Elsayed2020"], "LSC6_score")

# LSC surface marker proxy: CD34 + KIT + IL3RA (CD123) - CD38
def lsc_surface(adata):
    sc.tl.score_genes(adata, gene_list=["CD34", "KIT", "IL3RA"],
                      score_name="LSC_surface_pos", ctrl_size=50, use_raw=False)
    sc.tl.score_genes(adata, gene_list=["CD38"],
                      score_name="LSC_surface_neg", ctrl_size=50, use_raw=False)
    adata.obs["LSC_surface_score"] = adata.obs["LSC_surface_pos"] - adata.obs["LSC_surface_neg"]
    print(f"  LSC_surface_score range: [{adata.obs['LSC_surface_score'].min():.3f}, "
          f"{adata.obs['LSC_surface_score'].max():.3f}]")
lsc_surface(a)

# ---------------------------------------------------------------------------
# 2. Hallmark pathway activity via decoupler aucell (replaces ssGSEA — much
#    faster for single-cell + works on log-norm sparse expr)
# ---------------------------------------------------------------------------
# Parse Hallmark .gmt into decoupler's long-format DataFrame
print("[sig] reading Hallmark GMT…")
hallmark_gmt = ROOT / "data/raw/msigdb/MSigDB_Hallmark_2020.gmt"
hallmark = []
with open(hallmark_gmt) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        term = parts[0]
        # GMT typically: term, description, gene1, gene2, ...
        genes = [g for g in parts[2:] if g]
        for g in genes:
            hallmark.append({"source": term, "target": g, "weight": 1.0})
hallmark = pd.DataFrame(hallmark)
print(f"[sig] Hallmark: {hallmark['source'].nunique()} terms, {len(hallmark)} edges")

print("[sig] running decoupler AUCell on Hallmark…")
# decoupler 2.1.x has dc.mt.aucell with positional (data, net) and the net DataFrame
# columns must be ['source','target','weight'] (already true for `hallmark`).
try:
    if hasattr(dc, "mt") and hasattr(dc.mt, "aucell"):
        dc.mt.aucell(data=a, net=hallmark, verbose=False)
    else:
        dc.run_aucell(mat=a, net=hallmark, source="source", target="target",
                      use_raw=False, min_n=5, verbose=False)
    new_keys = [k for k in a.obsm.keys() if "aucell" in k.lower()]
    print(f"[sig] decoupler created obsm keys: {new_keys}")
    # Promote a few top Hallmark scores into .obs for convenience
    for k in new_keys:
        m = a.obsm[k]
        if hasattr(m, 'columns'):
            for term in m.columns[:5]:
                a.obs[f"Hallmark_{term.replace(' ', '_')}"] = m[term].values
        break
except Exception as e:
    print(f"[sig] decoupler AUCell failed: {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# 3. Save final integrated, annotated, scored AnnData
# ---------------------------------------------------------------------------
out = PROC / "van_galen_phase2_complete.h5ad"
a.write_h5ad(out, compression="gzip")
print(f"[sig] wrote {out}  size: {out.stat().st_size/1e9:.2f} GB")

# Summary report
summary = {
    "n_cells": int(a.n_obs),
    "n_genes": int(a.n_vars),
    "patients": int(a.obs["patient_id"].nunique()),
    "samples": int(a.obs["sample_id"].nunique()),
    "obs_columns": list(a.obs.columns),
    "obsm_keys": list(a.obsm.keys()),
    "disease_state_counts": a.obs["disease_state"].value_counts().to_dict(),
    "primary_malignant_call_counts": (
        a.obs["primary_malignant_call"].value_counts().to_dict()
        if "primary_malignant_call" in a.obs else {}),
    "LSC17_mean": float(a.obs["LSC17_score"].mean()) if "LSC17_score" in a.obs else None,
    "D_local_trace_mean": float(a.obs["D_local_trace"].mean()) if "D_local_trace" in a.obs else None,
}
with open(OUTS / "phase2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[sig] summary → {OUTS / 'phase2_summary.json'}")
print(json.dumps(summary, indent=2, default=str))
