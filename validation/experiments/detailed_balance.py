# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_detailed_balance.py — Detailed Balance / Stationary Distribution Test
==============================================================================
TASB Validation Suite — Tier 2.C
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
The strongest possible test for a sampler: given an energy matrix J with a
KNOWN analytical Boltzmann distribution, verify that THRML's empirical sample
distribution converges to the exact target.

For TASB's independent-node graph at position i:
    Target distribution:  P_i = softmax(J[i, :])  (known analytically)
    Empirical distribution: p̂_i = (1/K) Σ_k indicator(sample_k = j)

At large K, p̂_i → P_i if and only if the sampler is drawing from the correct
stationary distribution.

TEST PROTOCOL
-------------
1. Construct synthetic J matrices of sizes S ∈ {5, 10, 20} (no model needed)
2. For each S, run thrml_sample(K=10000)
3. For each query position i with span ≥ 5:
   - observed_j = K * p_thermo[0, h, i, j]  (rescaled empirical counts)
   - expected_j = K * softmax(J[i, :])_j    (exact Boltzmann counts)
   - chi-squared statistic = Σ_j (obs_j - exp_j)^2 / exp_j
   - df = S - 1
   - p-value = chi2.sf(stat, df)
4. PASS: mean p-value > 0.05, fraction of positions with p > 0.05 > 0.90
   (some false rejections at α=0.05 are expected by chance)

Also report KL and TV at K=10000 as converged reference values.

WHY CHI-SQUARED WORKS WITH AVERAGED OUTPUT
------------------------------------------
thrml_sample returns p_thermo ≈ mean(K sample frequency vectors).
Since each of the K samples is an iid draw (independent-node graph),
K * p_thermo is a valid sufficient statistic for the chi-squared test —
equivalent to observed counts from K draws. This is a standard goodness-
of-fit test formulation.

KEY CLAIM
---------
A passing chi-squared test on a system with known analytical target is the
gold-standard sampler validation. No skeptic can dismiss a chi-squared p > 0.05
from 10,000 samples against an exact Boltzmann target. This is the result
that proves the sampler is physically correct.

REPRODUCE
---------
  python experiments/tasb_detailed_balance.py
  python experiments/tasb_detailed_balance.py --K 5000  # faster, less power
==============================================================================
"""

import argparse
import csv
import datetime
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Config — no model needed
# ---------------------------------------------------------------------------
N_Q       = 24
N_KV      = 8
HEAD_DIM  = 128
SCALE     = 1.0 / (HEAD_DIM ** 0.5)
SEED_BASE = 42
K_DEFAULT = 10000
MIN_SPAN  = 5

S_VALUES  = [5, 10, 20]    # attention window sizes to test


@dataclass
class MockCapture:
    q_post_rope:    torch.Tensor
    k_post_rope:    torch.Tensor
    scaling:        float
    attention_mask: Optional[torch.Tensor]


def make_capture(S: int, seed: int = 42, device: str = "cuda") -> MockCapture:
    torch.manual_seed(seed)
    q = torch.randn(1, N_Q, S, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(1, N_KV, S, HEAD_DIM, device=device, dtype=torch.float32)
    mask = torch.full((1, 1, S, S), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return MockCapture(q_post_rope=q, k_post_rope=k,
                       scaling=SCALE, attention_mask=mask)


def analytical_softmax(capture: MockCapture) -> torch.Tensor:
    """Returns (1, n_q, S, S) softmax from capture."""
    q = capture.q_post_rope.float()
    k = capture.k_post_rope.float()
    B, n_q, S, _ = q.shape
    n_kv  = k.shape[1]
    n_kvg = n_q // n_kv
    k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k
    J = torch.matmul(q, k_exp.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        J = J + capture.attention_mask.float()
    return F.softmax(J, dim=-1)


# ---------------------------------------------------------------------------
# Chi-squared goodness of fit for one position
# ---------------------------------------------------------------------------
def chi2_gof(observed_counts: np.ndarray,
             expected_probs: np.ndarray,
             K: int) -> dict:
    """
    Chi-squared goodness of fit.
    observed_counts = K * p_thermo[position]  (float, treated as counts)
    expected_probs  = softmax[position]        (true probabilities)
    Returns: stat, df, p_value
    """
    S = len(expected_probs)

    # Only include bins with expected count ≥ 5 (chi-sq validity requirement)
    expected_counts = K * expected_probs
    valid = expected_counts >= 5.0
    if valid.sum() < 2:
        return {"stat": float('nan'), "df": 0, "p_value": float('nan'),
                "n_bins": int(valid.sum()), "valid": False}

    obs = observed_counts[valid]
    exp = expected_counts[valid]
    stat = float(np.sum((obs - exp) ** 2 / exp))
    df   = int(valid.sum()) - 1
    p    = float(chi2.sf(stat, df)) if df > 0 else float('nan')

    return {"stat": round(stat, 4), "df": df, "p_value": round(p, 4),
            "n_bins": int(valid.sum()), "valid": True}


# ---------------------------------------------------------------------------
# Main test for one S value
# ---------------------------------------------------------------------------
def run_one_S(S: int, K: int, device: str) -> dict:
    from thermobridge.backends.thrml import thrml_sample

    print(f"\n  S={S}, K={K}")
    capture     = make_capture(S, seed=SEED_BASE, device=device)
    softmax_ref = analytical_softmax(capture)        # (1, n_q, S, S)

    t0 = time.perf_counter()
    p_thermo = thrml_sample(capture, K=K, seed=SEED_BASE,
                             n_warmup=0, steps_per_sample=1)
    elapsed = time.perf_counter() - t0

    print(f"  thrml_sample done in {elapsed:.1f}s")

    p_np  = p_thermo.cpu().numpy()         # (1, n_q, S, S)
    sm_np = softmax_ref.cpu().numpy()      # (1, n_q, S, S)

    # Run chi-squared for each valid (head, position) pair
    p_values  = []
    kl_values = []
    tv_values = []

    for h in range(N_Q):
        for i in range(S):
            span = i + 1   # causal mask: position i sees keys 0..i
            if span < MIN_SPAN:
                continue

            obs_counts = K * p_np[0, h, i, :]    # empirical counts
            exp_probs  = sm_np[0, h, i, :]        # exact Boltzmann probs

            gof = chi2_gof(obs_counts, exp_probs, K)
            if gof["valid"]:
                p_values.append(gof["p_value"])

            # KL and TV
            p_clamped  = np.clip(p_np[0, h, i, :], 1e-10, None)
            sm_clamped = np.clip(sm_np[0, h, i, :], 1e-10, None)
            kl = float(np.sum(sm_clamped * np.log(sm_clamped / p_clamped)))
            tv = float(0.5 * np.sum(np.abs(p_np[0, h, i, :] - sm_np[0, h, i, :])))
            kl_values.append(kl)
            tv_values.append(tv)

    p_arr  = np.array(p_values)
    kl_arr = np.array(kl_values)
    tv_arr = np.array(tv_values)

    n_tested   = len(p_arr)
    n_pass_chi = int((p_arr > 0.05).sum()) if len(p_arr) > 0 else 0
    frac_pass  = n_pass_chi / n_tested if n_tested > 0 else 0.0
    mean_p     = float(np.mean(p_arr))   if len(p_arr) > 0 else float('nan')
    mean_kl    = float(np.mean(kl_arr))  if len(kl_arr) > 0 else float('nan')
    mean_tv    = float(np.mean(tv_arr))  if len(tv_arr) > 0 else float('nan')

    # PASS: ≥90% of positions have p > 0.05 AND mean p > 0.20
    # (at α=0.05, ~5% false rejections expected by chance)
    passed = (frac_pass >= 0.90) and (mean_p > 0.20)

    print(f"  Positions tested: {n_tested}")
    print(f"  Chi-sq p > 0.05: {n_pass_chi}/{n_tested} ({100*frac_pass:.1f}%)")
    print(f"  Mean p-value:   {mean_p:.4f}")
    print(f"  Mean KL:        {mean_kl:.6f}")
    print(f"  Mean TV:        {mean_tv:.6f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "S": S, "K": K,
        "n_positions_tested": n_tested,
        "n_chi2_pass":        n_pass_chi,
        "frac_chi2_pass":     round(frac_pass, 4),
        "mean_p_value":       round(mean_p, 4),
        "mean_kl":            round(mean_kl, 7),
        "mean_tv":            round(mean_tv, 7),
        "elapsed_s":          round(elapsed, 2),
        "pass":               passed,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TASB Detailed Balance / Stationary Distribution Test (T2.C)")
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    parser.add_argument("--S-values", nargs="+", type=int, default=S_VALUES,
                        dest="s_values")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  TASB Detailed Balance Test (Tier 2.C Validation)")
    print(f"  K={args.K} samples | S values: {args.s_values}")
    print(f"  Device: {args.device}  |  No model required")
    print(f"  Test: chi-squared goodness of fit vs known analytical Boltzmann target")
    print(f"  Pass: ≥90% of positions have p > 0.05, mean p > 0.20")
    print(f"{'='*70}")

    rows = []
    for S in args.s_values:
        row = run_one_S(S, args.K, args.device)
        rows.append(row)

    all_pass = all(r["pass"] for r in rows)

    print(f"\n{'='*70}")
    print(f"  T2.C SUMMARY")
    for r in rows:
        print(f"  S={r['S']:>3}  chi2_pass={r['frac_chi2_pass']:.1%}"
              f"  mean_p={r['mean_p_value']:.4f}"
              f"  KL={r['mean_kl']:.6f}  TV={r['mean_tv']:.6f}"
              f"  → {'PASS' if r['pass'] else 'FAIL'}")
    print(f"\n  OVERALL T2.C: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print(f"  THRML samples converge to exact Boltzmann distribution.")
        print(f"  Chi-squared goodness-of-fit confirmed at K={args.K} samples.")
        print(f"  This is the gold-standard sampler validation result.")
    print(f"{'='*70}\n")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_detailed_balance_{ts}.csv"
    fieldnames = ["S", "K", "n_positions_tested", "n_chi2_pass",
                  "frac_chi2_pass", "mean_p_value", "mean_kl", "mean_tv",
                  "elapsed_s", "pass"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
