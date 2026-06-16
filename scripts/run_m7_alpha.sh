#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_m7_alpha.sh — Run M7 α fine sweep with auto-captured console.
#
# Usage:
#   ./run_m7_alpha.sh              # full, 12 α × 4 prompts, ~30-40 min
#   ./run_m7_alpha.sh --quick      # 4 α × 2 prompts, ~10 min
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results"
CONSOLE="$OUT_DIR/tasb_m7_alpha_console_${TS}.txt"

mkdir -p "$OUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 α SWEEP — console captured to $CONSOLE"
echo "  Started: $(date)"
echo "  Args:    $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL python tasb_m7_alpha_sweep.py $EXTRA_ARGS 2>&1 \
    | tee -a "$CONSOLE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 α SWEEP COMPLETE — console saved to $CONSOLE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
