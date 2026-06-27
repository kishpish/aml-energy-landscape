"""Attach Van Galen's per-cell .anno columns (CellType, PredictionRefined,
Score_HSC..Score_NK, CyclingScore, MutTranscripts) to the integrated AnnData.

These columns were missed by the initial loader (which only read .dem files);
this script reads each .anno.txt.gz, builds a barcode→row map, and merges.

Reads:   data/processed/van_galen_annotated.h5ad
Writes:  data/processed/van_galen_annotated.h5ad  (in place)
"""
from pathlib import Path
import pandas as pd
import scanpy as sc

ROOT = Path(".")
PROC = ROOT / "data/processed"
EXTRACTED = ROOT / "data/raw/van_galen/extracted"

print("[anno] loading current AnnData…")
a = sc.read_h5ad(PROC / "van_galen_annotated.h5ad")

manifest = pd.read_csv(ROOT / "config/sample_manifest.tsv", sep="\t")
primary = set(manifest.loc[manifest["cohort"] == "primary", "sample_id"])

# Build a per-barcode anno dataframe
parts = []
for anno in sorted(EXTRACTED.glob("*.anno.txt.gz")):
    sid = anno.name.replace(".anno.txt.gz", "").split("_", 1)[1]
    if sid not in primary:
        continue
    df = pd.read_csv(anno, sep="\t", index_col=0)
    parts.append(df)

anno_df = pd.concat(parts, axis=0)
print(f"[anno] anno rows: {len(anno_df)}")
print(f"[anno] anno columns: {list(anno_df.columns)}")

# Align by barcode
overlap = anno_df.index.intersection(a.obs_names)
print(f"[anno] barcode overlap: {len(overlap)}/{a.n_obs}")
anno_df = anno_df.loc[overlap]

# Reindex to AnnData order, fill NaN for cells not in anno
anno_df = anno_df.reindex(a.obs_names)

# Subset to columns we want
keep = ["CellType", "PredictionRefined", "PredictionRF2",
        "CyclingScore", "CyclingBinary", "MutTranscripts", "WtTranscripts",
        "Score_HSC", "Score_Prog", "Score_GMP", "Score_ProMono", "Score_Mono",
        "Score_cDC", "Score_pDC", "Score_earlyEry", "Score_lateEry",
        "Score_ProB", "Score_B", "Score_Plasma", "Score_T", "Score_CTL", "Score_NK"]
keep = [c for c in keep if c in anno_df.columns]
for c in keep:
    s = anno_df[c]
    if s.dtype == object:
        a.obs[f"vg_{c}"] = s.fillna("nan").astype(str).values
    else:
        a.obs[f"vg_{c}"] = pd.to_numeric(s, errors="coerce").astype("float32").values

# Friendly aliases
if "vg_CellType" in a.obs:
    a.obs["VanGalen_CellType"] = a.obs["vg_CellType"]
if "vg_PredictionRefined" in a.obs:
    a.obs["VanGalen_malignant_call"] = a.obs["vg_PredictionRefined"]

print(f"[anno] attached {len(keep)} columns")
print(f"[anno] VanGalen_CellType values: "
      f"{a.obs['VanGalen_CellType'].value_counts().head(15).to_dict()}")
print(f"[anno] VanGalen_malignant_call: "
      f"{a.obs['VanGalen_malignant_call'].value_counts().to_dict()}")

# Re-save
out = PROC / "van_galen_annotated.h5ad"
a.write_h5ad(out, compression="gzip")
print(f"[anno] re-wrote {out}")
