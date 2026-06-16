#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_m7_seed.sh — Run M7 seed variance sweep with auto-captured console.
#
# Usage:
#   ./run_m7_seed.sh              # full, 12 seeds, ~1.5 hr
#   ./run_m7_seed.sh --quick      # 4 seeds × 2 prompts, ~15 min
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results"
CONSOLE="$OUT_DIR/tasb_m7_seed_console_${TS}.txt"

mkdir -p "$OUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 SEED VARIANCE — console captured to $CONSOLE"
echo "  Started: $(date)"
echo "  Args:    $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL python tasb_m7_seed_variance.py $EXTRA_ARGS 2>&1 \
    | tee -a "$CONSOLE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M7 SEED VARIANCE COMPLETE — console saved to $CONSOLE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
