"""One-time builder: maps the 19,441 HVG-filtered Van Galen gene symbols to
chromosomes via mygene.info, caches in config/gene_chromosomes.tsv.
Idempotent; safe to re-run."""
from pathlib import Path
import json
import pandas as pd
import scanpy as sc
import requests
import sys

ROOT = Path(".")
OUT = ROOT / "config/gene_chromosomes.tsv"

if OUT.exists() and OUT.stat().st_size > 100_000:
    print(f"[gene_chr] already exists: {OUT} ({OUT.stat().st_size} bytes)")
    sys.exit(0)

# Use the QC'd AnnData as the gene universe
in_h5 = ROOT / "data/processed/van_galen_qc.h5ad"
if not in_h5.exists():
    # fall back to raw
    in_h5 = ROOT / "data/processed/van_galen_raw.h5ad"

a = sc.read_h5ad(in_h5, backed="r")
symbols = list(a.var_names.astype(str))
print(f"[gene_chr] looking up {len(symbols)} symbols")

results = {}
for i in range(0, len(symbols), 1000):
    chunk = symbols[i:i+1000]
    r = requests.post("https://mygene.info/v3/query",
                      data={"q": ",".join(chunk),
                            "scopes": "symbol",
                            "fields": "genomic_pos.chr",
                            "species": "human"},
                      timeout=60)
    for hit in r.json():
        chrom = None
        gp = hit.get("genomic_pos")
        if isinstance(gp, dict):
            chrom = gp.get("chr")
        elif isinstance(gp, list) and gp:
            chrom = gp[0].get("chr")
        if chrom:
            results[hit["query"]] = chrom
    print(f"  {i+len(chunk):>5d} / {len(symbols)}  (mapped so far: {len(results)})")

df = pd.DataFrame([{"gene_symbol": k, "chr": v} for k, v in results.items()])
df.to_csv(OUT, sep="\t", index=False)
print(f"[gene_chr] wrote {OUT}  ({len(df)} mapped rows)")
