"""Phase 6.1 — prepare the LINCS L1000 drug-perturbation matrix.

Reads the decompressed Level-5 GCTX, filters to AML-cell-line compound
signatures (trt_cp), and caches the filtered matrix + metadata so the
connectivity scoring (6.2) and latent-mapping (6.3) steps don't have to touch
the 35 GB file again.

AML cell lines present in GSE92742 Phase 1: HL60, THP1, PL21, SKM1, U937
(MOLM13/MV411/OCIAML3/KG1/KASUMI1/NB4 are Phase-2-only — GSE70138 — and are
documented as a deferred extension.)

Outputs:
  data/processed/lincs_aml_zscores.parquet   genes × signatures (z-scores)
  data/processed/lincs_aml_sig_info.csv      signature metadata
  data/processed/lincs_gene_info.csv         gene metadata (with landmark flag)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse

ROOT = Path(".")
LINCS = ROOT / "data/raw/lincs"
PROC = ROOT / "data/processed"

GCTX = LINCS / "GSE92742_Level5.gctx"

AML_LINES = ["HL60", "THP1", "PL21", "SKM1", "U937",
             # phase-2 lines (absent here, kept for forward-compat):
             "MOLM13", "MV411", "OCIAML3", "KG1", "KASUMI1", "NB4"]

print("[p6_01] reading sig_info…")
sig = pd.read_csv(LINCS / "GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t",
                  low_memory=False)
gene = pd.read_csv(LINCS / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")

mask = sig["cell_id"].isin(AML_LINES) & (sig["pert_type"] == "trt_cp")
sig_aml = sig[mask].copy()
print(f"[p6_01] AML-line compound signatures: {len(sig_aml)} "
      f"({sig_aml['pert_iname'].nunique()} unique compounds, "
      f"{sig_aml['cell_id'].nunique()} cell lines)")

print(f"[p6_01] parsing {len(sig_aml)} columns from GCTX (this reads only the "
      f"selected signatures, fast)…")
gx = parse(str(GCTX), cid=sig_aml["sig_id"].tolist())
mat = gx.data_df  # rows = gene ids (str), cols = sig_ids
print(f"[p6_01] matrix: {mat.shape}  (genes × signatures)")

# Map gene ids (integers as strings) to symbols
gene["pr_gene_id"] = gene["pr_gene_id"].astype(str)
id_to_sym = dict(zip(gene["pr_gene_id"], gene["pr_gene_symbol"]))
mat.index = [id_to_sym.get(str(g), str(g)) for g in mat.index]
# drop duplicate symbols (keep first)
mat = mat[~mat.index.duplicated(keep="first")]
print(f"[p6_01] matrix after symbol mapping: {mat.shape}")

# Save
mat.to_parquet(PROC / "lincs_aml_zscores.parquet")
sig_aml.to_csv(PROC / "lincs_aml_sig_info.csv", index=False)
gene.to_csv(PROC / "lincs_gene_info.csv", index=False)
print(f"[p6_01] wrote lincs_aml_zscores.parquet ({mat.shape}), "
      f"sig_info ({len(sig_aml)}), gene_info ({len(gene)})")
print(f"[p6_01] per-line counts: {sig_aml['cell_id'].value_counts().to_dict()}")
