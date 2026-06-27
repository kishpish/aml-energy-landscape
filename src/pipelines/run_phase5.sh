#!/usr/bin/env bash
# Phase 5 runner (dynamical characterization). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 5: p5_01_state_definition"
$PY src/phase5_characterization/p5_01_state_definition.py 2>&1 | tee logs/p5_01_state_definition.log

echo ">> Phase 5: p5_02_biophysical_fingerprint"
$PY src/phase5_characterization/p5_02_biophysical_fingerprint.py 2>&1 | tee logs/p5_02_biophysical_fingerprint.log

echo ">> Phase 5: p5_03_de_and_enrichment"
$PY src/phase5_characterization/p5_03_de_and_enrichment.py 2>&1 | tee logs/p5_03_de_and_enrichment.log

echo ">> Phase 5: p5_04_lsc_subtyping"
$PY src/phase5_characterization/p5_04_lsc_subtyping.py 2>&1 | tee logs/p5_04_lsc_subtyping.log

echo ">> Phase 5: p5_05_persistence_ratios"
$PY src/phase5_characterization/p5_05_persistence_ratios.py 2>&1 | tee logs/p5_05_persistence_ratios.log

echo ">> Phase 5: p5_06_finalize"
$PY src/phase5_characterization/p5_06_finalize.py 2>&1 | tee logs/p5_06_finalize.log

echo ">> Phase 5 complete"
