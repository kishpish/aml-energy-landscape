#!/usr/bin/env bash
# Phase 2 runner (configuration space). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 2: p2_01_load_and_join"
$PY src/phase2_configuration_space/p2_01_load_and_join.py 2>&1 | tee logs/p2_01_load_and_join.log

echo ">> Phase 2: p2_02_qc_and_doublets"
$PY src/phase2_configuration_space/p2_02_qc_and_doublets.py 2>&1 | tee logs/p2_02_qc_and_doublets.log

echo ">> Phase 2: p2_03_hvg_and_scvi"
$PY src/phase2_configuration_space/p2_03_hvg_and_scvi.py 2>&1 | tee logs/p2_03_hvg_and_scvi.log

echo ">> Phase 2: p2_04_integration_qc"
$PY src/phase2_configuration_space/p2_04_integration_qc.py 2>&1 | tee logs/p2_04_integration_qc.log

echo ">> Phase 2: p2_05_celltype_annotation"
$PY src/phase2_configuration_space/p2_05_celltype_annotation.py 2>&1 | tee logs/p2_05_celltype_annotation.log

echo ">> Phase 2: p2_01b_attach_anno"
$PY src/phase2_configuration_space/p2_01b_attach_anno.py 2>&1 | tee logs/p2_01b_attach_anno.log

echo ">> Phase 2: p2_06_cnv_classification"
$PY src/phase2_configuration_space/p2_06_cnv_classification.py 2>&1 | tee logs/p2_06_cnv_classification.log

echo ">> Phase 2: p2_07_diffusion_tensor"
$PY src/phase2_configuration_space/p2_07_diffusion_tensor.py 2>&1 | tee logs/p2_07_diffusion_tensor.log

echo ">> Phase 2: p2_08_signature_scoring"
$PY src/phase2_configuration_space/p2_08_signature_scoring.py 2>&1 | tee logs/p2_08_signature_scoring.log

echo ">> Phase 2: p2_09_finalize"
$PY src/phase2_configuration_space/p2_09_finalize.py 2>&1 | tee logs/p2_09_finalize.log

echo ">> Phase 2 complete"
