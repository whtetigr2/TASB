#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_m7_K.sh — Run M7 K sweep with auto-captured console output.
#
# Usage:
#   ./run_m7_K.sh              # full sweep, ~1-2 hr
#   ./run_m7_K.sh --quick      # 3 K values × 2 prompts, ~15 min
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results"
CONSOLE="$OUT_DIR/tasb_m7_K_console_${TS}.txt"

mkdir -p "$OUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 K SWEEP — console captured to $CONSOLE"
echo "  Started: $(date)"
echo "  Args:    $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL python tasb_m7_K_sweep.py $EXTRA_ARGS 2>&1 \
    | tee -a "$CONSOLE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 K SWEEP COMPLETE — console saved to $CONSOLE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
