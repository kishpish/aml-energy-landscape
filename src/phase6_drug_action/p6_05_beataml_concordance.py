"""Phase 6.5 — Beat AML deconvolution + drug-AUC concordance.

1. Build a per-state signature matrix (mean log-expression on the union of
   top-30 markers per state, from Phase 5 real-cell DE).
2. Deconvolve each Beat AML bulk RNA-seq sample into per-state fractions via
   non-negative least squares (NNLS) — the lightweight, server-free analogue
   of CIBERSORTx / BayesPrism.
3. For each (state, drug) pair, Spearman-correlate the per-sample state
   fraction with the ex vivo drug AUC across patients. Negative correlation
   = higher state fraction → lower AUC → more drug-sensitive.
4. BH-correct across all (state, drug) pairs.

Outputs:
  outputs/phase6/beataml_state_fractions.csv         samples × states
  outputs/phase6/beataml_drug_state_correlation.csv  (state, drug, rho, p, q)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
BEAT = ROOT / "data/raw/beataml2_repo"
OUTP5 = ROOT / "outputs/phase5"
OUTP6 = ROOT / "outputs/phase6"

# ---- 1. state signature matrix ----
print("[p6_05] building per-state signature matrix…")
a = sc.read_h5ad(PROC / "van_galen_phase5_complete.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True","1","1.0"]).values
a_real = a[~is_syn].copy()

de = pd.read_csv(OUTP5 / "de_per_state_real.csv")
state_def = pd.read_csv(OUTP5 / "state_definitions.csv")
eval_states = state_def[state_def["n_real"] >= 50]["state_id"].tolist()

# marker union (top 30 up per state)
marker_union = set()
state_markers = {}
for sid in eval_states:
    sub = de[(de["group"] == sid) & (de["logfoldchanges"] > 0)].nlargest(30, "logfoldchanges")
    genes = sub["names"].tolist()
    state_markers[sid] = genes
    marker_union |= set(genes)
marker_union = sorted(marker_union)
print(f"[p6_05] {len(eval_states)} states, {len(marker_union)} union markers")

# mean log-expression per state on marker_union
gene_idx = {g: i for i, g in enumerate(a_real.var_names)}
mk_idx = [gene_idx[g] for g in marker_union if g in gene_idx]
mk_genes = [g for g in marker_union if g in gene_idx]
import scipy.sparse as sp
X = a_real[:, mk_genes].X
X = X.toarray() if sp.issparse(X) else np.asarray(X)
sig_mat = pd.DataFrame(index=mk_genes)
for sid in eval_states:
    m = (a_real.obs["state_id"].astype(str).values == sid)
    if m.sum() == 0: continue
    sig_mat[sid] = X[m].mean(axis=0)
print(f"[p6_05] signature matrix: {sig_mat.shape}")

# ---- 2. Beat AML bulk expression ----
print("[p6_05] loading Beat AML expression…")
expr = pd.read_csv(BEAT / "beataml_waves1to4_norm_exp_dbgap.txt", sep="\t")
# first column is gene symbol (display_label or stable_id)
gcol = expr.columns[0]
# Some Beat AML files have 'display_label' and 'stable_id' as first two columns
id_cols = [c for c in expr.columns if expr[c].dtype == object][:2]
print(f"[p6_05] expr id columns: {id_cols}")
# Beat AML: 'stable_id' = Ensembl, 'display_label' = gene SYMBOL. Use symbol.
sym_col = "display_label" if "display_label" in expr.columns else id_cols[-1]
print(f"[p6_05] using symbol column: {sym_col}")
expr = expr.set_index(sym_col)
sample_cols = [c for c in expr.columns if c not in id_cols]
expr_num = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
print(f"[p6_05] Beat AML expr: {expr_num.shape} (genes × samples)")

# align signature genes with Beat AML genes
common = [g for g in sig_mat.index if g in expr_num.index]
print(f"[p6_05] signature genes in Beat AML: {len(common)}/{len(sig_mat)}")
S = sig_mat.loc[common].values            # (n_markers, n_states)
B = expr_num.loc[common].values            # (n_markers, n_samples)
# z-score each gene across the references to balance scales
S_z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)
B_z = (B - np.nanmean(B, 1, keepdims=True)) / (np.nanstd(B, 1, keepdims=True) + 1e-6)
B_z = np.nan_to_num(B_z)

# ---- 3. NNLS deconvolution per sample ----
print("[p6_05] NNLS deconvolution…")
fractions = np.zeros((B.shape[1], S.shape[1]), dtype=np.float32)
for j in range(B.shape[1]):
    coef, _ = nnls(S_z, B_z[:, j])
    fractions[j] = coef
# normalize to sum to 1 per sample
fractions = fractions / (fractions.sum(1, keepdims=True) + 1e-9)
frac_df = pd.DataFrame(fractions, index=sample_cols, columns=eval_states)
frac_df.to_csv(OUTP6 / "beataml_state_fractions.csv")
print(f"[p6_05] wrote beataml_state_fractions.csv ({frac_df.shape})")

# ---- 4. drug AUC table ----
print("[p6_05] loading Beat AML drug AUC…")
drug = pd.read_csv(BEAT / "beataml_probit_curve_fits_v4_dbgap.txt", sep="\t")
# columns include dbgap_rnaseq_sample, inhibitor, auc
auc_col = "auc" if "auc" in drug.columns else [c for c in drug.columns if "auc" in c.lower()][0]
samp_col = "dbgap_rnaseq_sample"
drug = drug.dropna(subset=[samp_col, auc_col, "inhibitor"])
print(f"[p6_05] drug AUC rows: {len(drug)}; samples with expr+AUC: "
      f"{len(set(drug[samp_col]) & set(sample_cols))}")

# pivot to samples × drugs
auc_pivot = drug.pivot_table(index=samp_col, columns="inhibitor",
                              values=auc_col, aggfunc="median")
common_samples = [s for s in sample_cols if s in auc_pivot.index]
print(f"[p6_05] common samples (expr ∩ AUC): {len(common_samples)}")

# ---- 5. correlate state fraction vs drug AUC ----
print("[p6_05] correlating state fractions with drug AUC…")
rows = []
frac_common = frac_df.loc[common_samples]
auc_common = auc_pivot.loc[common_samples]
# limit drugs to those with ≥ 30 measured samples
drug_counts = auc_common.notna().sum()
valid_drugs = drug_counts[drug_counts >= 30].index.tolist()
print(f"[p6_05] {len(valid_drugs)} drugs with ≥30 samples")

for sid in eval_states:
    fr = frac_common[sid].values
    for dr in valid_drugs:
        au = auc_common[dr].values
        ok = ~np.isnan(au) & ~np.isnan(fr)
        if ok.sum() < 30: continue
        rho, p = spearmanr(fr[ok], au[ok])
        rows.append({"state_id": sid, "drug": dr, "n_samples": int(ok.sum()),
                     "spearman_rho": round(float(rho), 4),
                     "p_value": float(p)})
corr = pd.DataFrame(rows)
# BH correction
corr["q_value"] = multipletests(corr["p_value"], method="fdr_bh")[1]
corr = corr.sort_values("spearman_rho")  # most negative = sensitivity
corr.to_csv(OUTP6 / "beataml_drug_state_correlation.csv", index=False)
print(f"[p6_05] wrote beataml_drug_state_correlation.csv ({len(corr)} rows)")
print("\n[p6_05] strongest negative correlations (state fraction → drug sensitivity):")
print(corr[corr["q_value"] < 0.1].head(15)[
    ["state_id","drug","spearman_rho","q_value","n_samples"]].to_string(index=False))
