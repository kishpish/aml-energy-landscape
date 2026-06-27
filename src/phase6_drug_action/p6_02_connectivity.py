"""Phase 6.2 — classical connectivity (reversal) scoring.

For each cell state we build a query signature from the Phase 5 DE results:
  up-genes   = top 150 by logFC (positive, real-cell DE)
  down-genes = top 150 by logFC (negative)

For each AML-line drug signature (z-score profile over ~12k genes), we
compute a *reversal score* using a Kolmogorov-Smirnov-style weighted
connectivity (the cmap "tau"-analogue):

  score = mean( rank(drug_z)[up] )  −  mean( rank(drug_z)[down] )    [normalized]

A drug that *reverses* the disease signature pushes up-genes DOWN and
down-genes UP, giving a strongly negative score. We report a normalized
connectivity in [-100, +100] (negative = reversal = candidate therapeutic).

Per (state, drug) we also aggregate across the multiple signatures of the
same compound (different cell lines / doses / times) by taking the median.

Outputs:
  outputs/phase6/connectivity_per_state_drug.csv   (state, drug, conn_score, n_sigs)
  outputs/phase6/connectivity_top_hits.csv         top-20 reversal drugs per state
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTP5 = ROOT / "outputs/phase5"
OUTP6 = ROOT / "outputs/phase6"
OUTP6.mkdir(exist_ok=True, parents=True)

print("[p6_02] loading LINCS AML matrix + sig info…")
mat = pd.read_parquet(PROC / "lincs_aml_zscores.parquet")   # genes × signatures
sig = pd.read_csv(PROC / "lincs_aml_sig_info.csv")
sig = sig.set_index("sig_id")
print(f"[p6_02] LINCS matrix: {mat.shape}")

# Pre-rank each drug signature column once (rank across genes)
print("[p6_02] pre-ranking drug signatures…")
# rank within each column: high z-score → high rank
ranks = mat.rank(axis=0)  # genes × signatures, rank 1..n_genes per column
n_genes = ranks.shape[0]
gene_to_row = {g: i for i, g in enumerate(ranks.index)}
ranks_np = ranks.values  # (n_genes, n_sigs)
sig_ids = ranks.columns.tolist()
sig_pert = sig.loc[sig_ids, "pert_iname"].values

# Compute full two-tailed DE (the saved Phase 5 DE only has top-50 up-genes).
print("[p6_02] computing full two-tailed DE per state…")
import scanpy as sc
import warnings; warnings.filterwarnings("ignore")
a = sc.read_h5ad(ROOT / "data/processed/van_galen_phase5_complete.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True","1","1.0"]).values
a_real = a[~is_syn].copy()
sdef = pd.read_csv(OUTP5 / "state_definitions.csv")
eval_states = sdef[sdef["n_real"] >= 50]["state_id"].tolist()
a_real = a_real[a_real.obs["state_id"].astype(str).isin(eval_states)].copy()
a_real.obs["state_id"] = a_real.obs["state_id"].astype("category")
sc.tl.rank_genes_groups(a_real, "state_id", method="wilcoxon",
                        n_genes=a_real.n_vars, use_raw=False)
de = sc.get.rank_genes_groups_df(a_real, group=None)
states = de["group"].unique().tolist()
print(f"[p6_02] {len(states)} states (full DE, {de.shape[0]} rows) "
      f"to score against {len(sig_ids)} drug sigs")

def connectivity(up_idx, down_idx):
    """Weighted connectivity over all drug signatures.
    Returns array of per-signature scores in [-100, 100]."""
    if len(up_idx) == 0 or len(down_idx) == 0:
        return None
    up_mean = ranks_np[up_idx].mean(axis=0)      # (n_sigs,)
    down_mean = ranks_np[down_idx].mean(axis=0)
    # normalize ranks to [0,1]
    raw = (up_mean - down_mean) / n_genes        # in [-1, 1]
    return raw * 100.0                            # [-100, 100]

rows = []
for sid in states:
    sub = de[de["group"] == sid]
    up = sub[sub["logfoldchanges"] > 0].nlargest(150, "logfoldchanges")["names"].tolist()
    down = sub[sub["logfoldchanges"] < 0].nsmallest(150, "logfoldchanges")["names"].tolist()
    up_idx = np.array([gene_to_row[g] for g in up if g in gene_to_row])
    down_idx = np.array([gene_to_row[g] for g in down if g in gene_to_row])
    if len(up_idx) < 10 or len(down_idx) < 10:
        continue
    scores = connectivity(up_idx, down_idx)        # (n_sigs,)
    if scores is None:
        continue
    df_s = pd.DataFrame({"sig_id": sig_ids, "drug": sig_pert,
                         "conn_score": scores})
    # aggregate per drug (median across signatures)
    agg = (df_s.groupby("drug")
              .agg(conn_score=("conn_score", "median"),
                   n_sigs=("conn_score", "size"))
              .reset_index())
    agg["state_id"] = sid
    rows.append(agg)

conn = pd.concat(rows, ignore_index=True)
conn.to_csv(OUTP6 / "connectivity_per_state_drug.csv", index=False)
print(f"[p6_02] wrote connectivity_per_state_drug.csv ({len(conn)} rows)")

# Top reversal hits per state (most negative connectivity)
top_hits = (conn.sort_values("conn_score")
                .groupby("state_id")
                .head(20)
                .reset_index(drop=True))
top_hits.to_csv(OUTP6 / "connectivity_top_hits.csv", index=False)
print(f"[p6_02] wrote connectivity_top_hits.csv")

# Show top hits for a few key states
for sid in ["A1_basin_edge_0", "A1_L1_0", "A0_L1_2"]:
    if sid in conn["state_id"].values:
        t = conn[conn["state_id"] == sid].nsmallest(8, "conn_score")
        print(f"\n[p6_02] top reversal drugs for {sid}:")
        print(t[["drug", "conn_score", "n_sigs"]].to_string(index=False))
