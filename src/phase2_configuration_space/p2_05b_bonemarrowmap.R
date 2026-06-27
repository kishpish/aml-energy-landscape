#!/usr/bin/env Rscript
# Phase 2.8 — BoneMarrowMap Symphony projection (run from R).
#
# Why a separate R script: BoneMarrowMap distributes its Symphony reference as
# an .rds object that requires Seurat + Symphony + uwot in R. Calling this from
# Python via rpy2 works but adds 5+ minutes of R-import overhead per run, and
# the R env install for Seurat is best done outside Python.
#
# Usage:
#   Rscript scripts/p2_05b_bonemarrowmap.R \
#     /path/to/van_galen_annotated.h5ad \
#     /path/to/van_galen_bmm_projected.csv
#
# What it does:
#   1) load BoneMarrowMap_SymphonyReference.rds
#   2) load query AnnData (via reticulate -> scanpy)
#   3) project query onto reference
#   4) write per-cell CSV: cell_barcode, predicted_state, projection_confidence,
#      pseudotime, projected_UMAP_x, projected_UMAP_y
#
# Pre-reqs (one-time, in R):
#   install.packages(c("BiocManager","reticulate","devtools","Seurat","uwot"))
#   BiocManager::install(c("SingleCellExperiment","Symphony","SeuratObject"))
#   devtools::install_github("andygxzeng/BoneMarrowMap")

args <- commandArgs(trailingOnly = TRUE)
query_h5ad <- args[1]
out_csv    <- args[2]

ROOT <- "."
REF_RDS  <- file.path(ROOT, "data/raw/bonemarrowmap/BoneMarrowMap_SymphonyReference.rds")
UWOT_F   <- file.path(ROOT, "data/raw/bonemarrowmap/BoneMarrowMap_uwot_model.uwot")

suppressPackageStartupMessages({
  library(BoneMarrowMap)
  library(Symphony)
  library(Seurat)
  library(uwot)
  library(reticulate)
})

# Load query via scanpy
sc  <- import("scanpy")
adata <- sc$read_h5ad(query_h5ad)
counts <- t(as.matrix(adata$X))  # genes x cells
rownames(counts) <- adata$var_names$tolist()
colnames(counts) <- adata$obs_names$tolist()

# Load reference
ref <- readRDS(REF_RDS)
uw  <- uwot::load_uwot(UWOT_F)

# Project
proj <- BoneMarrowMap::map_Query(
  exp_query       = counts,
  metadata_query  = data.frame(row.names = colnames(counts)),
  ref_obj         = ref
)
proj <- BoneMarrowMap::predict_CellTypes(query_obj = proj, ref_obj = ref)

# Pseudotime
proj <- BoneMarrowMap::project_Pseudotime(
  query_obj = proj, ref_obj = ref
)

out <- data.frame(
  cell_barcode          = colnames(counts),
  bmm_predicted_state   = proj$cell_type_pred_knn,
  bmm_confidence        = proj$cell_type_pred_knn_prob,
  bmm_pseudotime        = proj$pseudotime,
  bmm_umap_x            = proj$umap[,1],
  bmm_umap_y            = proj$umap[,2]
)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("[bmm] wrote %s with %d rows\n", out_csv, nrow(out)))
