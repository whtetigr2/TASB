#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_seedsweep.sh — Run the M6 seed-sweep diagnostic with auto-captured
# console output. No more lost scrollback when you step away.
#
# Usage:
#   ./run_seedsweep.sh              # full sweep, ~10-15 min
#   ./run_seedsweep.sh --quick      # quick smoke, ~5 min
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results"
CONSOLE="$OUT_DIR/tasb_m6_seedsweep_console_${TS}.txt"

mkdir -p "$OUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  SEED SWEEP — console captured to $CONSOLE"
echo "  Started: $(date)"
echo "  Args:    $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════════════════════"

# stdbuf -oL -eL: line-buffer so partial output is preserved on abort
# tee -a: append, so nothing is lost if the script restarts mid-run
stdbuf -oL -eL python tasb_m6_seedsweep.py $EXTRA_ARGS 2>&1 \
    | tee -a "$CONSOLE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  SEED SWEEP COMPLETE — console saved to $CONSOLE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
