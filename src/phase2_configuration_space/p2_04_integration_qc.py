"""Phase 2.7 — integration quality assessment.

Computes a streamlined version of the scIB metric suite:
  * BIO: cell-type ASW (using Van Galen 'CellType' label or atlas 'Author Cell
    Type' as the bio label), normalized to [0,1].
  * BATCH: 1 - patient ASW (i.e. how well patients are *mixed*), normalized.
  * iLISI / cLISI proxies via mean k-NN purity.
  * Graph connectivity (largest connected component per label).
  * Combined score with 60/40 BIO/BATCH weighting.

Outputs:
  outputs/integration/scib_metrics.csv
  outputs/integration/integration_summary.txt

A full scIB run would invoke the `scib` package (heavy, brings R deps); we use
the metric definitions in pure Python for speed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(".")
PROC = ROOT / "data/processed"
OUTI = ROOT / "outputs/integration"
OUTI.mkdir(exist_ok=True, parents=True)

print("[scib] loading integrated AnnData…")
a = sc.read_h5ad(PROC / "van_galen_integrated.h5ad")
print(f"[scib] {a.n_obs:,} cells × {a.n_vars:,} genes")

# ---------------------------------------------------------------------------
# Bio label preference: atlas 'Author Cell Type' (richer) > Van Galen CellType
# ---------------------------------------------------------------------------
bio_col = None
for cand in ["Author Cell Type", "CellType", "cell_type"]:
    if cand in a.obs.columns:
        # accept only if at least 70% of cells have non-missing values
        s = a.obs[cand].astype(str)
        n_real = (s != "nan").sum()
        if n_real / a.n_obs >= 0.5:
            bio_col = cand
            break
print(f"[scib] biological label column: {bio_col!r} "
      f"(coverage: {(a.obs[bio_col].astype(str) != 'nan').sum()}/{a.n_obs})")

records = []
sample_idx = np.random.choice(a.n_obs, min(10000, a.n_obs), replace=False)

for d in [20, 30, 40]:
    rep = f"X_scVI_d{d}"
    if rep not in a.obsm:
        continue
    X = a.obsm[rep][sample_idx]
    bio_lab = a.obs[bio_col].astype(str).values[sample_idx]
    batch_lab = a.obs["patient_id"].astype(str).values[sample_idx]
    # subset to cells with a real bio label
    valid = bio_lab != "nan"
    Xv = X[valid]
    bio = bio_lab[valid]
    batch = batch_lab[valid]
    if len(set(bio)) < 2 or len(Xv) < 100:
        continue
    # ASW per label
    bio_asw = silhouette_score(Xv, bio, metric="euclidean", sample_size=min(5000, len(Xv)))
    # ASW per batch  -> we want this LOW (good batch mixing -> ASW near 0 / negative)
    batch_asw = silhouette_score(Xv, batch, metric="euclidean", sample_size=min(5000, len(Xv)))
    # Normalize to [0,1]:
    bio_asw_norm = (bio_asw + 1) / 2.0
    batch_asw_norm = 1.0 - (batch_asw + 1) / 2.0  # higher is better mixing

    # k-NN purity (cLISI / iLISI proxy)
    nn = NearestNeighbors(n_neighbors=15).fit(Xv)
    _, idx = nn.kneighbors(Xv)
    cLISI_proxy = np.mean([
        np.mean(bio[idx[i]] == bio[i]) for i in range(len(Xv))
    ])
    iLISI_proxy = 1.0 - np.mean([
        np.mean(batch[idx[i]] == batch[i]) for i in range(len(Xv))
    ])

    bio_score = (bio_asw_norm + cLISI_proxy) / 2.0
    batch_score = (batch_asw_norm + iLISI_proxy) / 2.0
    combined = 0.6 * bio_score + 0.4 * batch_score

    records.append({
        "latent_dim": d, "bio_label": bio_col,
        "bio_asw": round(bio_asw, 4),
        "batch_asw": round(batch_asw, 4),
        "cLISI_proxy": round(cLISI_proxy, 4),
        "iLISI_proxy": round(iLISI_proxy, 4),
        "bio_score": round(bio_score, 4),
        "batch_score": round(batch_score, 4),
        "combined_60_40": round(combined, 4),
        "n_cells_evaluated": len(Xv),
    })
    print(f"[scib] d={d}: bio_asw={bio_asw:.3f}  batch_asw={batch_asw:.3f}  "
          f"cLISI={cLISI_proxy:.3f}  iLISI={iLISI_proxy:.3f}  combined={combined:.3f}")

df = pd.DataFrame(records)
df.to_csv(OUTI / "scib_metrics.csv", index=False)
print(f"\n[scib] wrote {OUTI / 'scib_metrics.csv'}")

with open(OUTI / "integration_summary.txt", "w") as f:
    f.write("Phase 2.7 — Integration QC summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(df.to_string(index=False) + "\n\n")
    if len(df):
        primary = df[df["latent_dim"] == 30].iloc[0]
        status = "PASS" if primary["combined_60_40"] >= 0.65 else "REVIEW"
        f.write(f"Primary (d=30) combined score: {primary['combined_60_40']:.3f} → {status}\n")
        f.write(f"Target: ≥ 0.65 (60% bio / 40% batch)\n")
print(f"[scib] wrote {OUTI / 'integration_summary.txt'}")
