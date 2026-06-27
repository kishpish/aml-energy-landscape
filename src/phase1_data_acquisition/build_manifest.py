"""Build data/raw/manifest.csv: one row per top-level downloaded artifact.

For each, capture: dataset, role, path (relative to project root),
size_bytes, md5 (first 8 chars only for top-level files; full per-dir
totals for directories), source URL/note.
"""
import csv
import hashlib
import os
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "data/raw"
OUT = RAW / "manifest.csv"

def md5_short(path: Path, max_bytes: int = 1 << 26) -> str:
    """MD5 of up to first 64 MB — enough to fingerprint without rescanning huge files."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
            h.update(data)
        return h.hexdigest()[:12]
    except Exception:
        return ""

def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                pass
    return total

entries = [
    # (dataset, role, relpath, source_url, notes)
    ("van_galen_GSE116256", "primary_scRNAseq",
     "data/raw/van_galen/extracted/",
     "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256",
     "43 samples (.dem + .anno), 41090 cells, 27899 genes"),
    ("van_galen_GSE116256_tar", "primary_scRNAseq_archive",
     "data/raw/van_galen/GSE116256_RAW.tar",
     "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE116256&format=file",
     "original GEO tarball"),
    ("AML_scAtlas", "validation_scale",
     "data/raw/scatlas/AML_scAtlas.h5ad",
     "https://datasets.cellxgene.cziscience.com/6e37b9b1-185c-4505-9842-8138157c1923.h5ad",
     "Whittle et al. eLife 2026; 748679 cells x 21588 genes; via CellxGene Census"),
    ("beataml2", "drug_sensitivity",
     "data/raw/beataml2_repo/",
     "https://github.com/biodev/beataml2.0_data",
     "drug AUC + RNA-seq + clinical + mutations, 805 patients / 942 specimens"),
    ("TCGA_LAML", "bulk_survival",
     "data/raw/tcga_laml/",
     "https://gdc-hub.s3.dualstack.us-east-1.amazonaws.com/download/TCGA-LAML.*",
     "STAR TPM + counts + clinical + survival + mutations; 151 samples / 250 survival rows"),
    ("BoneMarrowMap", "healthy_reference",
     "data/raw/bonemarrowmap/",
     "https://bonemarrowmap.s3.us-east-2.amazonaws.com/",
     "Symphony reference (.rds) + uwot UMAP model; 263k healthy BM cells / 55 states"),
    ("LINCS_L1000_phase1", "drug_connectivity",
     "data/raw/lincs/",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/",
     "Level 5 GCTX (473647 sigs x 12328 genes) + sig/gene/cell/pert metadata"),
    ("oetjen_GSE120221", "healthy_validation",
     "data/raw/oetjen/extracted/",
     "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120221",
     "25 healthy BM donors, 10x mtx triples (matrix/barcodes/genes)"),
    ("oetjen_GSE120221_tar", "healthy_validation_archive",
     "data/raw/oetjen/GSE120221_RAW.tar",
     "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120221&format=file",
     "original GEO tarball"),
    ("HCA_bone_marrow", "scale_reference",
     "data/raw/hca_bm/hca_bone_marrow.h5ad",
     "CellxGene Census stable (2025-11-08)",
     "759562 healthy BM cells, queried via cellxgene-census API"),
    ("lasry_GSE185381", "inflammation_validation",
     "data/raw/lasry/extracted/",
     "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185381",
     "63 samples scRNAseq+CITEseq (RNA + ADT + VDJ); 9.0 GB extracted"),
    ("msigdb_libraries", "enrichment",
     "data/raw/msigdb/",
     "https://maayanlab.cloud/Enrichr/",
     "Hallmark, KEGG, Reactome, GO BP, WikiPathways + LSC17/LSC6/Tirosh"),
]

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "role", "path_rel", "size_bytes", "size_human",
                "md5_first64MB_or_blank", "source", "notes"])
    for ds, role, rel, src, note in entries:
        p = ROOT / rel
        size = 0
        md5 = ""
        if p.is_dir():
            size = dir_size(p)
        elif p.is_file():
            size = p.stat().st_size
            md5 = md5_short(p)
        else:
            print(f"  MISSING: {p}")
        human = f"{size / (1024**3):.2f} GB" if size > 1e8 else f"{size / (1024**2):.2f} MB"
        w.writerow([ds, role, rel, size, human, md5, src, note])
        print(f"  {ds:30s} {human:>10s}  {rel}")

print(f"\n[manifest] wrote {OUT}")
