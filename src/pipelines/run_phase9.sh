#!/usr/bin/env bash
# Phase 9 runner (catalog assembly). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY=python3

echo ">> Phase 9: p9_01_catalog"
$PY src/phase9_catalog/p9_01_catalog.py 2>&1 | tee logs/p9_01_catalog.log

echo ">> Phase 9 complete"
