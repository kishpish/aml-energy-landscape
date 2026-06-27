"""Phase 5.3 — differential expression + pathway enrichment per state.

For each state with n_real ≥ 30:

  A. Wilcoxon rank-sum DE vs all other cells (per-state markers).
     We compute this TWICE:
       i.  using REAL cells only (the "gold standard" comparison)
       ii. using ALL cells incl. synthetic (the "augmented" comparison)
     A high Spearman rank correlation between the two top-10 marker lists
     means the synthetic cells are amplifying signal, not introducing bias.

  B. Pathway enrichment via gseapy.enrichr on the top-50 markers (real-only),
     against MSigDB_Hallmark_2020, KEGG_2021_Human, and Reactome_2022.

Outputs:
  outputs/phase5/de_per_state.csv               n_state × top-10 markers (real / aug)
  outputs/phase5/marker_concordance.csv         per-state Spearman (real top vs aug top)
  outputs/phase5/enrichment/<state_id>.csv      one enrichment table per state
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
AUG = ROOT / "data/augmented"
OUTP5 = ROOT / "outputs/phase5"
ENR = OUTP5 / "enrichment"
ENR.mkdir(exist_ok=True, parents=True)

print("[p5_03] loading augmented + state-id AnnData…")
a = sc.read_h5ad(AUG / "van_galen_phase5_states.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True", "1", "1.0"]).values
print(f"[p5_03] {a.shape}, real={(~is_syn).sum()}, synth={is_syn.sum()}")

# Use log-normalized .X (already log-normed by p4_04)
state_defs = pd.read_csv(OUTP5 / "state_definitions.csv")
eval_states = state_defs[state_defs["n_real"] >= 30]["state_id"].tolist()
print(f"[p5_03] DE on {len(eval_states)} states (n_real ≥ 30)")

# ----- A. Wilcoxon DE -----
def run_de(adata, key="state_id", n_genes=50):
    sc.tl.rank_genes_groups(adata, groupby=key, method="wilcoxon", n_genes=n_genes,
                            use_raw=False, pts=True)
    res = sc.get.rank_genes_groups_df(adata, group=None)
    return res

# DE on real cells only
print("[p5_03] DE on REAL cells only…")
a_real = a[~is_syn].copy()
# subset to evaluation states (drop tiny states from groupby)
a_real = a_real[a_real.obs["state_id"].astype(str).isin(eval_states)].copy()
a_real.obs["state_id"] = a_real.obs["state_id"].astype("category")
de_real = run_de(a_real, "state_id")
de_real.to_csv(OUTP5 / "de_per_state_real.csv", index=False)
print(f"[p5_03] DE real → {len(de_real)} rows")

# DE on augmented (all cells)
print("[p5_03] DE on AUGMENTED cells…")
a_aug = a[a.obs["state_id"].astype(str).isin(eval_states)].copy()
a_aug.obs["state_id"] = a_aug.obs["state_id"].astype("category")
de_aug = run_de(a_aug, "state_id")
de_aug.to_csv(OUTP5 / "de_per_state_augmented.csv", index=False)
print(f"[p5_03] DE augmented → {len(de_aug)} rows")

# ----- A.ii. Marker concordance: rank correlation of top-10 between real & aug
print("[p5_03] computing per-state concordance…")
concord_rows = []
for sid in eval_states:
    real_genes = de_real[de_real["group"] == sid].head(10)["names"].tolist()
    aug_genes = de_aug[de_aug["group"] == sid].head(10)["names"].tolist()
    # Spearman rank correlation on intersection
    overlap = set(real_genes) & set(aug_genes)
    if len(overlap) >= 3:
        real_rank = {g: real_genes.index(g) for g in overlap}
        aug_rank = {g: aug_genes.index(g) for g in overlap}
        # Spearman of paired ranks
        gs = list(overlap)
        rho = pd.Series([real_rank[g] for g in gs]).corr(
            pd.Series([aug_rank[g] for g in gs]), method="spearman")
    else:
        rho = np.nan
    concord_rows.append({
        "state_id": sid,
        "top10_overlap": len(overlap),
        "spearman_top10": round(float(rho), 3) if not np.isnan(rho) else None,
        "real_top5": ",".join(real_genes[:5]),
        "aug_top5":  ",".join(aug_genes[:5]),
    })
df_concord = pd.DataFrame(concord_rows)
df_concord.to_csv(OUTP5 / "marker_concordance.csv", index=False)
print(f"[p5_03] marker concordance → {df_concord['top10_overlap'].mean():.1f} mean overlap")

# ----- B. Pathway enrichment via gseapy.enrichr -----
print("[p5_03] pathway enrichment per state…")
import gseapy as gp

LIBRARIES = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"]

# Pre-loaded GMTs from /data/raw/msigdb (avoids any network call)
msigdb_dir = ROOT / "data/raw/msigdb"

# Build a {term: [genes]} dict per library
def load_gmt(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            term = parts[0]
            genes = [g for g in parts[2:] if g]
            d[term] = genes
    return d

libs = {lib: load_gmt(msigdb_dir / f"{lib}.gmt") for lib in LIBRARIES}
print(f"[p5_03] loaded {sum(len(v) for v in libs.values())} terms across {len(libs)} libraries")

# Use simple hypergeometric enrichment instead (gp.enrich has API drift)
from scipy.stats import hypergeom

BACKGROUND = 20000  # total human protein-coding genes

def enrich_hypergeom(gene_list, lib_dict, background=BACKGROUND, top_k=10):
    """For each term in lib_dict, compute one-sided hypergeometric P-value
    for overlap between gene_list and term. Returns top_k by p-value."""
    gset = set(gene_list)
    N = background
    K = len(gset)
    rows = []
    for term, term_genes in lib_dict.items():
        term_set = set(term_genes)
        overlap = gset & term_set
        k = len(overlap)
        if k < 2: continue
        M = len(term_set)
        # P(X >= k) where X ~ Hypergeometric(N, M, K)
        pval = hypergeom.sf(k - 1, N, M, K)
        rows.append({
            "term": term, "overlap_n": k, "term_size": M, "query_size": K,
            "p_value": pval,
            "overlap_genes": ",".join(sorted(overlap))[:200],
        })
    if not rows: return None
    df = pd.DataFrame(rows).sort_values("p_value").head(top_k)
    # BH adjustment within library
    df["adj_p"] = (df["p_value"] * len(df) / np.arange(1, len(df) + 1)).clip(upper=1.0)
    return df

enrich_summary = []
for sid in eval_states:
    top50 = de_real[(de_real["group"] == sid) & (de_real["logfoldchanges"] > 0)] \
                .head(50)["names"].tolist()
    if len(top50) < 5:
        continue
    state_enr = []
    for lib_name, lib_dict in libs.items():
        df_lib = enrich_hypergeom(top50, lib_dict, top_k=10)
        if df_lib is None or df_lib.empty: continue
        df_lib["library"] = lib_name
        df_lib["state_id"] = sid
        state_enr.append(df_lib)
    if state_enr:
        enr_df = pd.concat(state_enr, ignore_index=True)
        enr_df.to_csv(ENR / f"{sid}.csv", index=False)
        # Pull top 3 across all libraries by p-value
        for _, r in enr_df.sort_values("p_value").head(3).iterrows():
            enrich_summary.append({
                "state_id": sid,
                "library": r["library"],
                "term": r["term"],
                "p_value": float(r["p_value"]),
                "adj_p_value": float(r["adj_p"]),
                "overlap": f"{r['overlap_n']}/{r['term_size']}",
                "overlap_genes": r["overlap_genes"],
            })

# Top-3 enrichments per state, consolidated
df_enrich = pd.DataFrame(enrich_summary)
df_enrich.to_csv(OUTP5 / "top3_enrichment_per_state.csv", index=False)
print(f"[p5_03] enrichment for {df_enrich['state_id'].nunique()} states; "
      f"{len(df_enrich)} top-3 records")
print(df_enrich.head(20).to_string(index=False))
