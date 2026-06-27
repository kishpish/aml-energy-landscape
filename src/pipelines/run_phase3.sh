#!/usr/bin/env bash
# Phase 3 runner (landscape reconstruction). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 3: p3_01_score_model"
$PY src/phase3_landscape/p3_01_score_model.py 2>&1 | tee logs/p3_01_score_model.log

echo ">> Phase 3: p3_02_critical_points"
$PY src/phase3_landscape/p3_02_critical_points.py 2>&1 | tee logs/p3_02_critical_points.log

echo ">> Phase 3: p3_03_basin_assignment"
$PY src/phase3_landscape/p3_03_basin_assignment.py 2>&1 | tee logs/p3_03_basin_assignment.log

echo ">> Phase 3: p3_04_barriers"
$PY src/phase3_landscape/p3_04_barriers.py 2>&1 | tee logs/p3_04_barriers.log

echo ">> Phase 3: p3_05_rare_cells"
$PY src/phase3_landscape/p3_05_rare_cells.py 2>&1 | tee logs/p3_05_rare_cells.log

echo ">> Phase 3: p3_06_finalize"
$PY src/phase3_landscape/p3_06_finalize.py 2>&1 | tee logs/p3_06_finalize.log

echo ">> Phase 3 complete"
