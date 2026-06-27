"""Download MSigDB-style gene sets via Enrichr libraries (proxies for MSigDB), and stage LSC signatures.

Enrichr libraries that closely mirror MSigDB / KEGG / Reactome / GO:
    MSigDB_Hallmark_2020
    KEGG_2021_Human
    Reactome_2022
    GO_Biological_Process_2025
    WikiPathways_2024_Human

These are stored as Gene Matrix Transposed (GMT)-like text files for offline use.
"""
import gseapy as gp
import json
import os
from pathlib import Path

OUT_DIR = Path("./data/raw/msigdb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIBS = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "WikiPathways_2024_Human",
]

for lib in LIBS:
    print(f"[msigdb] fetching {lib}")
    try:
        gs = gp.get_library(name=lib, organism="Human")
        out = OUT_DIR / f"{lib}.gmt"
        with open(out, "w") as f:
            for term, genes in gs.items():
                line = "\t".join([term, ""] + list(genes))
                f.write(line + "\n")
        print(f"  wrote {out}  ({len(gs)} terms)")
    except Exception as e:
        print(f"  FAILED {lib}: {e}")

# Hardcoded LSC signatures from Eppert/Ng papers
LSC17 = [
    "DNMT3B","ZBTB46","NYNRIN","ARHGAP22","LAPTM4B","MMRN1","DPYSL3",
    "KIAA0125","CDK6","CPXM1","SOCS2","SMIM24","EMP1","NGFRAP1",
    "CD34","AKR1C3","GPR56",
]
LSC6 = ["DNMT3B","CD34","ADGRG1","SOCS2","SPINK2","FAM30A"]

# Tirosh cell cycle gene sets (used in scanpy.tl.score_genes_cell_cycle)
S_PHASE = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
    "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
    "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
    "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
]
G2M_PHASE = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
    "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
    "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
    "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA",
    "PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3",
    "CBX5","CENPA",
]

sigs = {
    "LSC17_Ng2016": LSC17,
    "LSC6_Elsayed2020": LSC6,
    "Tirosh_S_phase": S_PHASE,
    "Tirosh_G2M_phase": G2M_PHASE,
}
with open(OUT_DIR / "AML_signatures.json", "w") as f:
    json.dump(sigs, f, indent=2)

# also as GMT
with open(OUT_DIR / "AML_signatures.gmt", "w") as f:
    for name, genes in sigs.items():
        f.write("\t".join([name, ""] + genes) + "\n")

print(f"[msigdb] wrote AML_signatures.json and AML_signatures.gmt with {list(sigs.keys())}")
