#!/usr/bin/env bash
# Phase 6 runner (drug action). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 6: p6_01_prepare_lincs"
$PY src/phase6_drug_action/p6_01_prepare_lincs.py 2>&1 | tee logs/p6_01_prepare_lincs.log

echo ">> Phase 6: p6_02_connectivity"
$PY src/phase6_drug_action/p6_02_connectivity.py 2>&1 | tee logs/p6_02_connectivity.log

echo ">> Phase 6: p6_03_drug_latent_projection"
$PY src/phase6_drug_action/p6_03_drug_latent_projection.py 2>&1 | tee logs/p6_03_drug_latent_projection.log

echo ">> Phase 6: p6_04_basin_escape"
$PY src/phase6_drug_action/p6_04_basin_escape.py 2>&1 | tee logs/p6_04_basin_escape.log

echo ">> Phase 6: p6_05_beataml_concordance"
$PY src/phase6_drug_action/p6_05_beataml_concordance.py 2>&1 | tee logs/p6_05_beataml_concordance.log

echo ">> Phase 6: p6_06_combined_ranking"
$PY src/phase6_drug_action/p6_06_combined_ranking.py 2>&1 | tee logs/p6_06_combined_ranking.log

echo ">> Phase 6 complete"
