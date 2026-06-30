#!/usr/bin/env bash
# run_cv_experiments.sh — Cv Experiment Suite (FIND-022 blocking items)
# Run from: validation/
# Estimated total: ~26 min on A100 (3+10+8+5)
#
# Usage:
#   bash run_cv_experiments.sh          # full suite
#   bash run_cv_experiments.sh --fast   # sanity-check only (~3 min, 1 prompt)

set -euo pipefail

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

TS=$(date +%Y%m%d_%H%M%S)
LOG="results/run_cv_experiments_${TS}.log"
mkdir -p results

log() { echo "$*" | tee -a "$LOG"; }

log "========================================================================"
log "  TASB Cv Experiment Suite — FIND-022 Blocking Items"
log "  Started: $(date)"
log "  Log:     $LOG"
log "========================================================================"

PASS=0
FAIL=0

run_step() {
    local label="$1"; shift
    log ""
    log "------------------------------------------------------------------------"
    log "  $label"
    log "------------------------------------------------------------------------"
    if python "$@" 2>&1 | tee -a "$LOG"; then
        log "  ✓ $label PASSED"
        PASS=$((PASS + 1))
    else
        log "  ✗ $label FAILED (exit $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Step 1: fast sanity check (always runs) ──────────────────────────────────
run_step "STEP 1 — Cv Layer Sweep (fast, 1 prompt, 28 layers, ~3 min)" \
    experiments/tasb_cv_layer_sweep.py --prompt-only

if [[ $FAST -eq 0 ]]; then

    # ── Step 2: full layer sweep ──────────────────────────────────────────────
    run_step "STEP 2 — Cv Layer Sweep (full, 4 prompts, ~10 min)" \
        experiments/tasb_cv_layer_sweep.py

    # ── Step 3: Gumbel π²/6 bias ─────────────────────────────────────────────
    run_step "STEP 3 — Cv Backend Bias (Gumbel π²/6, 200 MC trials, ~8 min)" \
        experiments/tasb_cv_backend_bias.py

    # ── Step 4: Cv × KL correlation ──────────────────────────────────────────
    run_step "STEP 4 — Cv×KL Pearson Correlation (4 prompts × 24 heads, ~5 min)" \
        experiments/tasb_cv_kl_correlation.py

fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "========================================================================"
log "  SUMMARY — $(date)"
log "  Passed: $PASS  |  Failed: $FAIL"
log "  Results written to: results/"
log "  Log: $LOG"
if [[ $FAIL -eq 0 ]]; then
    log "  OVERALL: ALL PASS ✓"
else
    log "  OVERALL: $FAIL step(s) FAILED — review log above"
fi
log "========================================================================"

exit $FAIL
