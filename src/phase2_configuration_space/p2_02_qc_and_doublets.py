"""Phase 2.1–2.3 — per-sample QC metrics, threshold inspection, Scrublet doublet
detection. Writes:
  data/processed/van_galen_qc.h5ad                  (cells after QC+doublet removal)
  outputs/qc/qc_per_sample.csv                      (retention stats)
  outputs/qc/scrublet_thresholds.csv                (doublet thresholds chosen)

Implements the Seq-Well-specific decisions from the methodology:
  * No CellBender (no empty-droplet info from Seq-Well dem matrices).
  * Skip SoupX too — Seq-Well has nano-well isolation and downstream scVI
    handles residual ambient effects. We document this in PHASE2_RESEARCH.md
    and keep raw integer counts as the only input layer.
  * Per-sample thresholds chosen *adaptively*: lower-mode of gene-count, 95th
    percentile of mitochondrial fraction (capped at 25%).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scrublet as scr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(".")
PROC = ROOT / "data/processed"
QC = ROOT / "outputs/qc"
QC.mkdir(exist_ok=True, parents=True)

print("[qc] loading raw…")
adata = sc.read_h5ad(PROC / "van_galen_raw.h5ad")
adata.X = adata.layers["counts"]   # ensure X is raw counts for QC metrics
print(f"[qc] starting cells: {adata.n_obs:,}")

# ---------------------------------------------------------------------------
# 1. Annotate gene metadata: mt, ribo, hemoglobin
# ---------------------------------------------------------------------------
adata.var["mt"]    = adata.var_names.str.startswith("MT-")
adata.var["ribo"]  = adata.var_names.str.match(r"^(RPS|RPL)")
adata.var["hb"]    = adata.var_names.str.match(r"^HB[^P]")
print(f"[qc] gene flags: mt={adata.var['mt'].sum()}, "
      f"ribo={adata.var['ribo'].sum()}, hb={adata.var['hb'].sum()}")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"],
                            inplace=True, log1p=False, percent_top=None)
# columns produced: n_genes_by_counts, total_counts, pct_counts_mt, pct_counts_ribo, pct_counts_hb

# ---------------------------------------------------------------------------
# 2. Per-sample adaptive thresholds + Scrublet
# ---------------------------------------------------------------------------
samples = sorted(adata.obs["sample_id"].unique())
print(f"[qc] {len(samples)} samples to process")

retain_rows = []
doublet_rows = []
keep_mask = np.zeros(adata.n_obs, dtype=bool)

for sid in samples:
    sub = adata.obs["sample_id"] == sid
    idx = np.where(sub)[0]
    if len(idx) < 30:
        # too few cells to threshold sensibly; keep all
        keep_mask[idx] = True
        retain_rows.append({"sample_id": sid, "n_before": len(idx),
                            "n_after_qc": len(idx), "n_after_doublet": len(idx),
                            "thr_min_genes": None, "thr_max_genes": None,
                            "thr_max_mt": None, "doublet_thr": None,
                            "doublet_rate": 0.0,
                            "note": "tiny sample — no filtering"})
        continue

    sub_obs = adata.obs.loc[adata.obs_names[idx]].copy()
    ng = sub_obs["n_genes_by_counts"].values
    mt = sub_obs["pct_counts_mt"].values

    # Adaptive thresholds:
    # min_genes = max(150, 5th percentile - 0.5*IQR), capped at 200
    p05, p25, p75, p95 = np.percentile(ng, [5, 25, 75, 95])
    iqr = p75 - p25
    thr_min_genes = max(150.0, p05 - 0.25 * iqr)
    thr_min_genes = min(thr_min_genes, 200.0)
    thr_max_genes = 5000.0
    thr_max_mt = min(25.0, np.percentile(mt, 95))  # 95th percentile or 25%, whichever is tighter
    thr_max_mt = max(thr_max_mt, 15.0)  # never go below 15%

    qc_ok = ((ng >= thr_min_genes) & (ng <= thr_max_genes) & (mt <= thr_max_mt))
    n_after_qc = qc_ok.sum()

    # Scrublet on QC-passing cells of this sample
    sub_idx = idx[qc_ok]
    if n_after_qc < 30 or sp.issparse(adata.X) is False:
        doublet_thr = None
        doublet_rate = 0.0
        doublet_keep = np.ones(n_after_qc, dtype=bool)
    else:
        X_sub = adata.layers["counts"][sub_idx]
        if not sp.issparse(X_sub):
            X_sub = sp.csr_matrix(X_sub)
        try:
            sc_obj = scr.Scrublet(X_sub.toarray() if X_sub.shape[0] < 5000 else X_sub,
                                  expected_doublet_rate=min(0.08, 0.008 * n_after_qc / 1000),
                                  random_state=0)
            scores, called = sc_obj.scrub_doublets(min_counts=2, min_cells=3,
                                                    min_gene_variability_pctl=85,
                                                    n_prin_comps=min(30, max(5, n_after_qc // 10)),
                                                    verbose=False)
            if scores is None:
                doublet_thr = None
                doublet_rate = 0.0
                doublet_keep = np.ones(n_after_qc, dtype=bool)
            else:
                # If Scrublet didn't bimodally separate, use 0.25 as conservative default
                thr = sc_obj.threshold_ if sc_obj.threshold_ is not None else 0.25
                doublet_keep = ~(scores >= thr)
                doublet_thr = float(thr)
                doublet_rate = float((~doublet_keep).mean())
        except Exception as e:
            print(f"  Scrublet failed for {sid}: {e}")
            doublet_thr = None
            doublet_rate = 0.0
            doublet_keep = np.ones(n_after_qc, dtype=bool)

    final_keep_idx = sub_idx[doublet_keep]
    keep_mask[final_keep_idx] = True

    retain_rows.append({"sample_id": sid, "n_before": len(idx),
                        "n_after_qc": int(n_after_qc),
                        "n_after_doublet": int(doublet_keep.sum()),
                        "thr_min_genes": round(thr_min_genes, 1),
                        "thr_max_genes": round(thr_max_genes, 1),
                        "thr_max_mt": round(thr_max_mt, 1),
                        "doublet_thr": doublet_thr,
                        "doublet_rate": round(doublet_rate, 3),
                        "note": ""})

    print(f"  {sid:20s}  {len(idx):>5d} → QC {int(n_after_qc):>5d} "
          f"(min_genes={thr_min_genes:.0f}, max_mt={thr_max_mt:.1f}%) "
          f"→ post-doublet {int(doublet_keep.sum()):>5d} "
          f"(rate={doublet_rate*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Subset, save QC report, save AnnData
# ---------------------------------------------------------------------------
report = pd.DataFrame(retain_rows)
report.to_csv(QC / "qc_per_sample.csv", index=False)
print(f"\n[qc] cumulative retention: {keep_mask.sum():,}/{adata.n_obs:,} "
      f"({100*keep_mask.mean():.1f}%)")

adata_qc = adata[keep_mask].copy()
adata_qc.write_h5ad(PROC / "van_galen_qc.h5ad", compression="gzip")
print(f"[qc] saved {PROC / 'van_galen_qc.h5ad'}  ({adata_qc.n_obs:,} cells × "
      f"{adata_qc.n_vars:,} genes)")
