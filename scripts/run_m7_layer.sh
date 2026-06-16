#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_m7_layer.sh — Run M7 layer sweep with auto-captured console output.
#
# Usage:
#   ./run_m7_layer.sh              # full sweep, ~3-4 hr
#   ./run_m7_layer.sh --quick      # 3 layers × 2 prompts, ~30 min
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results"
CONSOLE="$OUT_DIR/tasb_m7_layer_console_${TS}.txt"

mkdir -p "$OUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 LAYER SWEEP — console captured to $CONSOLE"
echo "  Started: $(date)"
echo "  Args:    $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL python tasb_m7_layer_sweep.py $EXTRA_ARGS 2>&1 \
    | tee -a "$CONSOLE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 LAYER SWEEP COMPLETE — console saved to $CONSOLE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
