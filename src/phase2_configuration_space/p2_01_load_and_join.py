"""Phase 2.1a — load Van Galen raw per-sample matrices, exclude the validation cohort
(cell lines + nanopore), join with the AML scAtlas Van Galen subset (so we inherit
atlas-level QC + harmonized barcodes + Author Cell Type annotations), and emit
data/processed/van_galen_raw.h5ad with .layers['counts'] = raw integer UMI counts.

This script realizes the strategic decision made after Flag 2: the AML scAtlas
contains study `van_galen_2019` with 23,344 cells. Their cell barcodes use the
same format as the raw GEO files (e.g. `AML1012-D0_AAAAAGTTACGT`), so we
inner-join on barcode to:
    * inherit the atlas's QC filtering (no need to redo ambient correction +
      mt/gene-count thresholds — the atlas authors did it already and consistently
      across 20 studies),
    * inherit the atlas's `Author Cell Type` labels and ontology calls,
    * keep the raw integer counts from Van Galen (the atlas X is normalized).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import gzip
import sys
import time

ROOT = Path(".")
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
PROC.mkdir(exist_ok=True, parents=True)

manifest = pd.read_csv(ROOT / "config/sample_manifest.tsv", sep="\t")
primary_samples = set(manifest.loc[manifest["cohort"] == "primary", "sample_id"].tolist())
print(f"[load] primary cohort samples: {len(primary_samples)}")

# ---------------------------------------------------------------------------
# 1. Load atlas Van Galen subset (in-memory, just the obs slice + barcodes)
# ---------------------------------------------------------------------------
print(f"[load] reading atlas (backed)…")
atlas = sc.read_h5ad(RAW / "scatlas/AML_scAtlas.h5ad", backed="r")
vg_mask = (atlas.obs["Study"].astype(str) == "van_galen_2019").values
print(f"[load] atlas Van Galen cells: {vg_mask.sum()}")
vg_obs = atlas.obs.loc[vg_mask].copy()
vg_barcodes = atlas.obs_names[vg_mask].astype(str).tolist()
vg_var = atlas.var.copy()  # gene metadata
del atlas

# Atlas barcode pattern: "AML1012-D0_AAAAAGTTACGT"
# Some samples may have suffix variants — extract sample_id portion
vg_sample_in_atlas = {b.rsplit("_", 1)[0] for b in vg_barcodes}
print(f"[load] sample_ids represented in atlas: {len(vg_sample_in_atlas)}")
print("       ", sorted(vg_sample_in_atlas))

# ---------------------------------------------------------------------------
# 2. Load Van Galen raw matrices for the primary cohort and inner-join on barcode
# ---------------------------------------------------------------------------
def load_one_dem(dem_path: Path, sample_id: str) -> ad.AnnData:
    counts = pd.read_csv(dem_path, sep="\t", index_col=0)  # genes x cells
    # Some GEO files have a "Gene" header on the index column already; pandas handled it.
    counts.columns = [c.strip() for c in counts.columns]
    X = sp.csr_matrix(counts.T.values.astype(np.int32))  # cells x genes
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=counts.columns),
        var=pd.DataFrame(index=counts.index),
    )
    a.obs["sample_id"] = sample_id
    return a

extracted = RAW / "van_galen/extracted"
dem_files = sorted(extracted.glob("*.dem.txt.gz"))
print(f"[load] found {len(dem_files)} .dem files in {extracted}")

per_sample = []
total_loaded = 0
t0 = time.time()
for dem in dem_files:
    # parse sample_id from "GSMxxx_<sample>.dem.txt.gz"
    sid = dem.name.replace(".dem.txt.gz", "")
    sid = sid.split("_", 1)[1]  # drop "GSM…_"
    if sid not in primary_samples:
        # validation cohort (cell line / nanopore) — skipped for primary analysis
        continue
    a = load_one_dem(dem, sid)
    total_loaded += a.n_obs
    per_sample.append(a)
    print(f"  {sid:20s}  {a.n_obs:>5d} cells  (cum {total_loaded:>6d})  "
          f"[{time.time() - t0:.1f}s]")

print(f"[load] concatenating {len(per_sample)} samples…")
raw = ad.concat(per_sample, join="outer", index_unique=None, merge="same",
                label="batch_sample")
# obs_names already include the sample prefix from the .dem column headers
# (e.g. "AML1012-D0_AAAAAGTTACGT") — DO NOT re-prefix.
print(f"[load] raw concatenated: {raw.n_obs} cells × {raw.n_vars} genes")

# ---------------------------------------------------------------------------
# 3. Annotate with atlas labels (left join — atlas only covers diagnosis +
#    healthy samples, so post-treatment cells get NaN labels which is fine)
# ---------------------------------------------------------------------------
atlas_bc_set = set(vg_barcodes)
in_atlas = np.array([b in atlas_bc_set for b in raw.obs_names])
print(f"[load] cells overlapping atlas: {in_atlas.sum()}/{raw.n_obs}  "
      f"(atlas-side: {len(atlas_bc_set)})")
raw.obs["in_atlas"] = in_atlas

# Subset atlas obs to those cells we have, then join
vg_obs_present = vg_obs.loc[vg_obs.index.intersection(raw.obs_names)].copy()
print(f"[load] joining {len(vg_obs_present)} atlas annotations…")

filtered = raw  # we keep ALL primary-cohort cells
filtered.layers["counts"] = filtered.X.copy()  # preserve raw integer counts
# Drop columns that conflict with our manifest-derived ones
to_drop = [c for c in ["sample_id", "patient_id", "donor_id"] if c in vg_obs_present.columns]
vg_obs_present = vg_obs_present.drop(columns=to_drop)
filtered.obs = filtered.obs.join(vg_obs_present, how="left", rsuffix="_atlas")

# ---------------------------------------------------------------------------
# 4. Annotate timepoint / patient / disease_state from the manifest
# ---------------------------------------------------------------------------
m = manifest.set_index("sample_id")
for col in ["patient_id", "timepoint", "day_since_dx",
            "treatment_status", "disease_state", "cohort"]:
    filtered.obs[col] = filtered.obs["sample_id"].map(m[col])

# ---------------------------------------------------------------------------
# 5. Coerce mixed-type object columns to strings (h5ad needs uniform dtypes)
# ---------------------------------------------------------------------------
for col in filtered.obs.columns:
    s = filtered.obs[col]
    if s.dtype == object or str(s.dtype) == "category":
        try:
            filtered.obs[col] = s.astype(str)
        except Exception:
            pass
    elif s.dtype == bool:
        filtered.obs[col] = s.astype("uint8")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out = PROC / "van_galen_raw.h5ad"
print(f"[load] writing {out}")
filtered.write_h5ad(out, compression="gzip")
print(f"[load] done.  n_obs={filtered.n_obs}  n_vars={filtered.n_vars}")
print(f"       patients: {filtered.obs['patient_id'].nunique()}")
print(f"       disease_state: {filtered.obs['disease_state'].value_counts().to_dict()}")
print(f"       has 'Author Cell Type' col: {'Author Cell Type' in filtered.obs.columns}")
