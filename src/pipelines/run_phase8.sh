#!/usr/bin/env bash
# Phase 8 runner (survival and panels). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 8: p8_01_tcga_scoring"
$PY src/phase8_survival_panels/p8_01_tcga_scoring.py 2>&1 | tee logs/p8_01_tcga_scoring.log

echo ">> Phase 8: p8_02_survival"
$PY src/phase8_survival_panels/p8_02_survival.py 2>&1 | tee logs/p8_02_survival.log

echo ">> Phase 8: p8_03_panels"
$PY src/phase8_survival_panels/p8_03_panels.py 2>&1 | tee logs/p8_03_panels.log

echo ">> Phase 8: p8_04_finalize"
$PY src/phase8_survival_panels/p8_04_finalize.py 2>&1 | tee logs/p8_04_finalize.log

echo ">> Phase 8 complete"
