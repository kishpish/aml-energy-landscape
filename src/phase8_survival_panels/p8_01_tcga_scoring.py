"""Phase 8.1 — score validated-state signatures in TCGA-LAML (bulk RNA-seq).

1. Load TCGA-LAML STAR TPM (Ensembl IDs), map to gene symbols via the scAtlas
   var table (feature_id → feature_name).
2. For each HIGH/MODERATE-confidence state, take its top-50 up-gene signature
   (Phase 5 real DE) and compute a per-patient ssGSEA score via gseapy.ssgsea.
3. Build a patient × state score matrix.

Outputs:
  data/processed/tcga_state_ssgsea.csv      patients × states
  outputs/phase8/tcga_scoring_meta.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
TCGA = ROOT / "data/raw/tcga_laml"
OUTP5 = ROOT / "outputs/phase5"
OUTP7 = ROOT / "outputs/phase7"
OUTP8 = ROOT / "outputs/phase8"
OUTP8.mkdir(exist_ok=True, parents=True)

# --- Ensembl → symbol map from scAtlas var ---
print("[p8_01] building Ensembl→symbol map from scAtlas…")
atlas = sc.read_h5ad(ROOT / "data/raw/scatlas/AML_scAtlas.h5ad", backed="r")
ens_to_sym = {}
if "feature_id" in atlas.var.columns:
    for eid, sym in zip(atlas.var["feature_id"].astype(str), atlas.var["feature_name"].astype(str)):
        ens_to_sym[eid.split(".")[0]] = sym
else:
    # atlas var index may be Ensembl; feature_name is symbol
    for eid, sym in zip(atlas.var_names.astype(str), atlas.var["feature_name"].astype(str)):
        ens_to_sym[eid.split(".")[0]] = sym
print(f"[p8_01] Ensembl→symbol entries: {len(ens_to_sym)}")
del atlas

# --- TCGA TPM ---
print("[p8_01] loading TCGA-LAML TPM…")
tpm = pd.read_csv(TCGA / "TCGA-LAML.star_tpm.tsv.gz", sep="\t", index_col=0)
print(f"[p8_01] TPM: {tpm.shape} (genes × samples)")
# map index Ensembl (with version) → symbol
tpm.index = [ens_to_sym.get(str(e).split(".")[0], None) for e in tpm.index]
tpm = tpm[~tpm.index.isna()]
tpm = tpm[~tpm.index.duplicated(keep="first")]
print(f"[p8_01] TPM after symbol mapping: {tpm.shape}")

# --- state signatures (HIGH + MODERATE) ---
cat = pd.read_csv(OUTP7 / "validated_state_catalog.csv")
keep_states = cat[cat["confidence_tier"].isin(["HIGH", "MODERATE"])]["state_id"].tolist()
print(f"[p8_01] scoring {len(keep_states)} HIGH/MODERATE states")

de = pd.read_csv(OUTP5 / "de_per_state_real.csv")
gene_sets = {}
for sid in keep_states:
    g = de[(de["group"] == sid) & (de["logfoldchanges"] > 0)].nlargest(50, "logfoldchanges")["names"].tolist()
    g = [x for x in g if x in tpm.index]
    if len(g) >= 10:
        gene_sets[sid] = g
print(f"[p8_01] {len(gene_sets)} states with ≥10 mappable signature genes")

# --- ssGSEA ---
print("[p8_01] running ssGSEA (gseapy)…")
ss = gp.ssgsea(data=tpm, gene_sets=gene_sets, outdir=None,
               sample_norm_method="rank", no_plot=True, threads=4,
               min_size=5, max_size=500)
# ssgsea result: res2d has columns Name (sample), Term (gene set), ES/NES
res = ss.res2d.copy()
# pivot to patients × states (use NES if present else ES)
val_col = "NES" if "NES" in res.columns else ("ES" if "ES" in res.columns else res.columns[-1])
res[val_col] = pd.to_numeric(res[val_col], errors="coerce")
score_mat = res.pivot_table(index="Name", columns="Term", values=val_col)
score_mat.to_csv(PROC / "tcga_state_ssgsea.csv")
print(f"[p8_01] ssGSEA score matrix: {score_mat.shape} (samples × states)")

meta = {
    "n_tcga_samples": int(score_mat.shape[0]),
    "n_states_scored": int(score_mat.shape[1]),
    "value_column": val_col,
    "n_genes_mapped": int(tpm.shape[0]),
}
with open(OUTP8 / "tcga_scoring_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"[p8_01] wrote tcga_state_ssgsea.csv + meta")
print(json.dumps(meta, indent=2))
