import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_cv_backend_bias.py — Cv Backend Bias Verification
==============================================================================
TASB Thermodynamic Validation — Gumbel π²/6 Bias Test
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
FIND-020 (derived 2026-06-29) predicts that Gumbel-max sampling introduces
a constant upward bias in Cv when Cv is computed from the Gumbel-perturbed
logits (a_j + G_j where G_j ~ Gumbel(0,1)):

    Cv_perturbed = Cv_true + Var(Gumbel(0,1)) = Cv_true + π²/6 ≈ Cv_true + 1.6449

This arises because: Var(a + G) = Var(a) + Var(G) (independence of QK logits
and Gumbel noise). Var(Gumbel(0,1)) = π²/6 exactly (Wolfram-verified).

Three tests:
  1. EXACT vs ANALYTICAL: Cv from K-sample p_thermo (exact backend) vs
     analytical softmax Cv — should be close for large K.
  2. GUMBEL PERTURBED BIAS: Monte Carlo estimate of E[Var_softmax(a+G)] - Var_softmax(a)
     — should equal π²/6 = 1.6449 ± 0.15.
  3. THRML vs ANALYTICAL: Cv from THRML p_thermo vs analytical — should be
     unbiased (IID draws per FIND-007).

PASS CONDITIONS (FIND-020 §Q3)
-------------------------------
    PASS: |Cv_exact_sampled - Cv_analytical| < 0.05 (mean across heads)
    PASS: |empirical Gumbel bias - π²/6| < 0.15
    PASS: |Cv_thrml - Cv_analytical| < 0.05 (if THRML available)

REPRODUCE
---------
    python experiments/tasb_cv_backend_bias.py
==============================================================================
"""

import csv
import datetime
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
LAYER_IDX = 18
ALPHA     = 0.0
K_LARGE   = 1000    # large K → p_thermo ≈ exact softmax
N_GUMBEL_TRIALS = 200   # Monte Carlo trials for π²/6 test
SEED      = 42

GUMBEL_BIAS_THEORY = math.pi**2 / 6   # ≈ 1.6449

PROMPT = "The relationship between thermodynamics and information theory reveals"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str):
    print(f"  Loading {model_id} in 4-bit (nf4)...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def compute_logits(capture) -> torch.Tensor:
    from transformers.models.llama.modeling_llama import repeat_kv
    Q = capture.q_post_rope.float()
    K = capture.k_post_rope.float()
    K_rep = repeat_kv(K, Q.shape[1] // K.shape[1])
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.float()
    return logits  # (B, n_q, S, S)


def cv_analytical(logits: torch.Tensor) -> np.ndarray:
    """Exact Cv = Var_softmax(logits) per head, positions 1..S-1."""
    p = torch.softmax(logits, dim=-1)
    n_q = logits.shape[1]
    cv = np.zeros(n_q)
    for h in range(n_q):
        log_h = logits[0, h, 1:, :]
        p_h   = p[0,   h, 1:, :]
        log_h = log_h.clamp(min=-1e9)  # HF causal sentinel (~-3.4e38) overflows float32 when squared
        E1 = (p_h * log_h).sum(-1)
        E2 = (p_h * log_h**2).sum(-1)
        cv[h] = (E2 - E1**2).mean().item()
    return cv


def cv_from_pthermo(p_thermo: torch.Tensor,
                    logits: torch.Tensor) -> np.ndarray:
    """Cv estimated via Var_{p_thermo}(original logits) per head."""
    n_q = logits.shape[1]
    cv = np.zeros(n_q)
    for h in range(n_q):
        log_h = logits[0, h, 1:, :]
        p_h   = p_thermo[0, h, 1:, :].float()
        p_h   = p_h / p_h.sum(-1, keepdim=True).clamp(min=1e-10)
        log_h = log_h.clamp(min=-1e9)  # HF causal sentinel (~-3.4e38) overflows float32 when squared
        E1 = (p_h * log_h).sum(-1)
        E2 = (p_h * log_h**2).sum(-1)
        cv[h] = (E2 - E1**2).mean().item()
    return cv


def measure_gumbel_perturbed_bias(logits: torch.Tensor,
                                   n_trials: int) -> tuple:
    """Monte Carlo estimate of E[Cv(a+G)] - Cv(a).

    Samples n_trials sets of Gumbel(0,1) noise, computes Cv(a+G) each time.
    Expected: mean gap ≈ π²/6 = 1.6449.

    Returns (mean_gap, std_gap).
    """
    cv_base = cv_analytical(logits)
    gaps = []
    for _ in range(n_trials):
        u = torch.rand_like(logits).clamp(min=1e-30)
        G = -torch.log(-torch.log(u))
        cv_pert = cv_analytical(logits + G)
        gaps.append(float((cv_pert - cv_base).mean()))
    return float(np.mean(gaps)), float(np.std(gaps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from thermobridge import bridge_forward

    print(f"\n{'='*70}")
    print(f"  TASB Cv Backend Bias Verification")
    print(f"  Layer {LAYER_IDX} | K={K_LARGE} | {N_GUMBEL_TRIALS} Gumbel MC trials")
    print(f"  Theoretical Gumbel bias: π²/6 = {GUMBEL_BIAS_THEORY:.6f}")
    print(f"{'='*70}")

    model, tok = load_model(MODEL_ID)

    # ── Step 1: exact backend capture ────────────────────────────────────
    print(f"\n  [1/4] Exact backend (K={K_LARGE})...", flush=True)
    result_exact = bridge_forward(
        model, tok, prompt=PROMPT,
        layer_idx=LAYER_IDX, alpha=ALPHA,
        backend='exact', K=K_LARGE, seed=SEED,
        return_intermediates=True,
    )
    capture = result_exact.layer_captures[LAYER_IDX]
    logits  = compute_logits(capture)

    cv_anal       = cv_analytical(logits)
    cv_samp_exact = cv_from_pthermo(
        result_exact.p_thermo_by_layer[LAYER_IDX], logits)

    # ── Step 2: Gumbel backend ────────────────────────────────────────────
    print(f"  [2/4] Gumbel backend (K={K_LARGE})...", flush=True)
    result_gumbel = bridge_forward(
        model, tok, prompt=PROMPT,
        layer_idx=LAYER_IDX, alpha=ALPHA,
        backend='gumbel', K=K_LARGE, seed=SEED,
        return_intermediates=True,
    )
    cv_samp_gumbel = cv_from_pthermo(
        result_gumbel.p_thermo_by_layer[LAYER_IDX], logits)

    # ── Step 3: Gumbel perturbed bias (π²/6 test) ────────────────────────
    print(f"  [3/4] Gumbel perturbed bias ({N_GUMBEL_TRIALS} MC trials)...",
          flush=True)
    t0 = time.perf_counter()
    bias_mean, bias_std = measure_gumbel_perturbed_bias(logits, N_GUMBEL_TRIALS)
    print(f"        Done in {time.perf_counter()-t0:.1f}s  "
          f"bias={bias_mean:.4f} ± {bias_std:.4f}")

    # ── Step 4: THRML backend ─────────────────────────────────────────────
    print(f"  [4/4] THRML backend...", flush=True)
    thrml_available = False
    cv_samp_thrml = np.full_like(cv_anal, float('nan'))
    try:
        result_thrml = bridge_forward(
            model, tok, prompt=PROMPT,
            layer_idx=LAYER_IDX, alpha=ALPHA,
            backend='thrml', K=K_LARGE, seed=SEED,
            return_intermediates=True,
        )
        cv_samp_thrml = cv_from_pthermo(
            result_thrml.p_thermo_by_layer[LAYER_IDX], logits)
        thrml_available = True
        print(f"        THRML OK")
    except Exception as e:
        print(f"        THRML unavailable: {e}")

    # ── Per-head table ────────────────────────────────────────────────────
    n_heads = len(cv_anal)
    print(f"\n  {'head':>4}  {'Cv_anal':>8}  {'Cv_exact':>8}  "
          f"{'Cv_gumbel':>9}  {'Cv_thrml':>8}  "
          f"{'Δexact':>7}  {'Δgumbel':>8}")
    print(f"  {'-'*72}")

    rows = []
    for h in range(n_heads):
        d_exact  = cv_samp_exact[h]  - cv_anal[h]
        d_gumbel = cv_samp_gumbel[h] - cv_anal[h]
        d_thrml  = cv_samp_thrml[h]  - cv_anal[h]
        print(f"  {h:>4}  {cv_anal[h]:>8.4f}  {cv_samp_exact[h]:>8.4f}  "
              f"{cv_samp_gumbel[h]:>9.4f}  {cv_samp_thrml[h]:>8.4f}  "
              f"{d_exact:>+7.4f}  {d_gumbel:>+8.4f}")
        rows.append({
            "head":                   h,
            "cv_analytical":          round(float(cv_anal[h]),       6),
            "cv_exact_sampled":       round(float(cv_samp_exact[h]), 6),
            "cv_gumbel_sampled":      round(float(cv_samp_gumbel[h]),6),
            "cv_thrml_sampled":       round(float(cv_samp_thrml[h]), 6),
            "diff_exact_vs_anal":     round(float(d_exact),          6),
            "diff_gumbel_vs_anal":    round(float(d_gumbel),         6),
            "diff_thrml_vs_anal":     round(float(d_thrml),          6),
            "gumbel_pert_bias_mean":  round(bias_mean,               6),
            "gumbel_pert_bias_std":   round(bias_std,                6),
            "pi2_over_6":             round(GUMBEL_BIAS_THEORY,      6),
        })

    mean_d_exact  = float(np.mean([r["diff_exact_vs_anal"]  for r in rows]))
    mean_d_gumbel = float(np.mean([r["diff_gumbel_vs_anal"] for r in rows]))
    mean_d_thrml  = float(np.nanmean([r["diff_thrml_vs_anal"] for r in rows]))

    # ── Verdict ───────────────────────────────────────────────────────────
    pass_exact        = abs(mean_d_exact)  < 0.05
    pass_gumbel_pert  = abs(bias_mean - GUMBEL_BIAS_THEORY) < 0.15
    pass_thrml        = (not thrml_available) or abs(mean_d_thrml) < 0.05

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"  [1] Exact sampled vs analytical:  "
          f"mean Δ = {mean_d_exact:+.4f}  "
          f"{'PASS ✓' if pass_exact else 'WARN ✗'} (threshold |Δ|<0.05)")
    print(f"  [2] Gumbel perturbed bias empirical: {bias_mean:.4f} ± {bias_std:.4f}")
    print(f"      Theoretical π²/6:               {GUMBEL_BIAS_THEORY:.4f}")
    print(f"      |empirical - theory|:            "
          f"{abs(bias_mean - GUMBEL_BIAS_THEORY):.4f}  "
          f"{'CONFIRMED ✓' if pass_gumbel_pert else 'NOT CONFIRMED ✗'} "
          f"(threshold 0.15)")
    if thrml_available:
        print(f"  [3] THRML vs analytical:  "
              f"mean Δ = {mean_d_thrml:+.4f}  "
              f"{'PASS ✓' if pass_thrml else 'WARN ✗'} (threshold |Δ|<0.05)")
    else:
        print(f"  [3] THRML: skipped (not available this session)")
    overall = pass_exact and pass_gumbel_pert and pass_thrml
    print(f"\n  OVERALL: {'PASS ✓' if overall else 'WARN — review above'}")
    print(f"{'='*70}\n")

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_cv_backend_bias_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
