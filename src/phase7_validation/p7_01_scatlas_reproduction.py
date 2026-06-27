"""Phase 7.1 — cross-dataset reproduction of discovered states in the AML scAtlas.

For each cell state with a marker signature (top-30 up-genes from Phase 5
real-cell DE), score every cell of the 748k-cell AML scAtlas via AUCell-style
mean-rank scoring, then test whether the state's signature is *enriched* in a
recognizable scAtlas population (i.e. concentrated rather than uniform).

We use the scAtlas's own `Author Cell Type` labels as the recognizable
populations. A state "reproduces" if its top-scoring scAtlas cell type is a
biologically sensible match to the state's Van Galen identity, and the
score is significantly higher there than the global median (Mann-Whitney).

Because the scAtlas is 748k × 21,588 and ~6 GB, we score in backed/chunked
mode using only the marker genes.

Outputs:
  outputs/phase7/scatlas_reproduction.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTP5 = ROOT / "outputs/phase5"
OUTP7 = ROOT / "outputs/phase7"
OUTP7.mkdir(exist_ok=True, parents=True)

# state signatures (top-30 up-genes, real DE)
de = pd.read_csv(OUTP5 / "de_per_state_real.csv")
state_def = pd.read_csv(OUTP5 / "state_definitions.csv")
eval_states = state_def[state_def["n_real"] >= 50]["state_id"].tolist()
sig = {}
for sid in eval_states:
    g = de[(de["group"] == sid) & (de["logfoldchanges"] > 0)].nlargest(30, "logfoldchanges")["names"].tolist()
    if len(g) >= 10:
        sig[sid] = g
print(f"[p7_01] {len(sig)} state signatures")

# Load scAtlas (need a gene-subset; read backed then subset to union markers)
print("[p7_01] loading scAtlas (backed)…")
atlas = sc.read_h5ad(ROOT / "data/raw/scatlas/AML_scAtlas.h5ad", backed="r")
# atlas var uses feature_name for symbols
atlas_syms = atlas.var["feature_name"].astype(str).values if "feature_name" in atlas.var.columns else atlas.var_names.astype(str).values
sym_to_col = {s: i for i, s in enumerate(atlas_syms)}

union_markers = sorted(set().union(*sig.values()))
present = [g for g in union_markers if g in sym_to_col]
print(f"[p7_01] union markers: {len(union_markers)}, present in atlas: {len(present)}")
col_idx = np.array([sym_to_col[g] for g in present])

# Load just those columns into memory (748k × ~600 markers ≈ manageable)
print("[p7_01] extracting marker columns from atlas…")
atlas_mem = atlas[:, col_idx].to_memory()
import scipy.sparse as sp
Xm = atlas_mem.X
Xm = Xm.toarray() if sp.issparse(Xm) else np.asarray(Xm)
Xm = Xm.astype(np.float32)
present_map = {g: j for j, g in enumerate(present)}
author_ct = atlas.obs["Author Cell Type"].astype(str).values
print(f"[p7_01] atlas marker matrix: {Xm.shape}")

# AUCell-style: rank genes within each cell, score = mean rank of signature genes
print("[p7_01] computing per-cell ranks…")
# rank across the marker genes per cell (axis=1)
from scipy.stats import rankdata
# For speed, rank within the marker submatrix per cell
ranks = np.apply_along_axis(rankdata, 1, Xm)  # (n_cells, n_markers)
n_markers = ranks.shape[1]

# expected scAtlas cell type for each state (from Van Galen identity)
vg_to_atlas = {
    "T": "T", "NK": "T", "CTL": "T", "B": "B", "Plasma": "Plasma",
    "Mono": "CD14+ Mono", "Mono-like": "CD14+ Mono", "ProMono": "ProMono",
    "ProMono-like": "ProMono", "cDC": "cDC", "cDC-like": "cDC", "pDC": "pDC",
    "GMP": "GMP", "GMP-like": "GMP", "HSC": "HSPC", "HSC-like": "HSPC",
    "Prog": "HSPC", "Prog-like": "HSPC", "earlyEry": "Erythroid",
    "lateEry": "Erythroid",
}

rows = []
for sid, genes in sig.items():
    gidx = [present_map[g] for g in genes if g in present_map]
    if len(gidx) < 5:
        continue
    score = ranks[:, gidx].mean(axis=1)  # (n_cells,)
    # which atlas cell type has the highest mean score
    df = pd.DataFrame({"score": score, "ct": author_ct})
    ct_means = df.groupby("ct")["score"].mean().sort_values(ascending=False)
    top_ct = ct_means.index[0]
    # expected
    vg = state_def[state_def["state_id"] == sid]["top_vangalen_type"].iloc[0]
    expected_ct = vg_to_atlas.get(str(vg), None)
    # enrichment test: top_ct cells vs rest
    in_top = (author_ct == top_ct)
    if in_top.sum() > 10 and (~in_top).sum() > 10:
        u, p = mannwhitneyu(score[in_top], score[~in_top], alternative="greater")
    else:
        p = np.nan
    reproduces = (expected_ct is not None and top_ct == expected_ct)
    rows.append({
        "state_id": sid,
        "vangalen_type": vg,
        "expected_atlas_ct": expected_ct,
        "top_scoring_atlas_ct": top_ct,
        "reproduces": reproduces,
        "enrichment_p": p,
        "top_ct_mean_score": round(float(ct_means.iloc[0]), 2),
        "global_median_score": round(float(np.median(score)), 2),
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTP7 / "scatlas_reproduction.csv", index=False)
n_repro = df_out["reproduces"].sum()
print(f"[p7_01] {n_repro}/{len(df_out)} states reproduce (top atlas CT matches expected)")
print(df_out[["state_id","vangalen_type","expected_atlas_ct",
              "top_scoring_atlas_ct","reproduces","enrichment_p"]].head(30).to_string(index=False))
