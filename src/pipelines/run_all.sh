#!/usr/bin/env bash
# Run the full pipeline (phases 2 through 9) in order.
set -euo pipefail
DIR="$(dirname "$0")"

bash "$DIR/run_phase2.sh"
bash "$DIR/run_phase3.sh"
bash "$DIR/run_phase4.sh"
bash "$DIR/run_phase5.sh"
bash "$DIR/run_phase6.sh"
bash "$DIR/run_phase7.sh"
bash "$DIR/run_phase8.sh"
bash "$DIR/run_phase9.sh"
