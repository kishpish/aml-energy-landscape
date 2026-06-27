#!/usr/bin/env bash
# Phase 4 runner (augmentation). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 4: p4_01_langevin_augmentation"
$PY src/phase4_augmentation/p4_01_langevin_augmentation.py 2>&1 | tee logs/p4_01_langevin_augmentation.log

echo ">> Phase 4: p4_02_decode_synthetic"
$PY src/phase4_augmentation/p4_02_decode_synthetic.py 2>&1 | tee logs/p4_02_decode_synthetic.log

echo ">> Phase 4: p4_03_validate"
$PY src/phase4_augmentation/p4_03_validate.py 2>&1 | tee logs/p4_03_validate.log

echo ">> Phase 4: p4_04_finalize"
$PY src/phase4_augmentation/p4_04_finalize.py 2>&1 | tee logs/p4_04_finalize.log

echo ">> Phase 4 complete"
