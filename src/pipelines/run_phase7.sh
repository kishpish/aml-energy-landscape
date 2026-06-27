#!/usr/bin/env bash
# Phase 7 runner (validation). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 7: p7_01_scatlas_reproduction"
$PY src/phase7_validation/p7_01_scatlas_reproduction.py 2>&1 | tee logs/p7_01_scatlas_reproduction.log

echo ">> Phase 7: p7_02_sensitivity"
$PY src/phase7_validation/p7_02_sensitivity.py 2>&1 | tee logs/p7_02_sensitivity.log

echo ">> Phase 7: p7_03_functional_concordance"
$PY src/phase7_validation/p7_03_functional_concordance.py 2>&1 | tee logs/p7_03_functional_concordance.log

echo ">> Phase 7: p7_04_confidence_tiering"
$PY src/phase7_validation/p7_04_confidence_tiering.py 2>&1 | tee logs/p7_04_confidence_tiering.log

echo ">> Phase 7 complete"
