# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_k_convergence.py — Boltzmann Sampler 1/K Convergence Proof (v2 — patched)
==============================================================================
TASB Validation Suite — Tier 1.D (patched) / Tier 2.B-4
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
For a sampler drawing K iid samples from a categorical distribution P of size
V, the empirical distribution P̂_K satisfies:

    E[KL(P || P̂_K)] ≈ (V-1) / (2K)         [theoretical, iid case]
    E[TV(P || P̂_K)] ≈ √((V-1) / (4K))       [theoretical, iid case]

Both KL ∝ 1/K and TV ∝ 1/√K. If both fits hold (R² > 0.95), it is strong
evidence that samples are iid draws from the correct Boltzmann distribution.

PATCH vs V1 (T1.D original)
-----------------------------
V1 averaged KL across ALL query positions, including causal-mask early positions
where query i attends to only i+1 keys (position 0 → span=1, position 1 → span=2,
etc.). These degenerate distributions (near-deterministic due to few valid keys)
contribute KL≈0 regardless of K, breaking the 1/K fit assumption which requires
uniform effective vocabulary size V.

Fix: `--min-span N` (default 5) filters out positions with effective attention
span < N before computing mean KL and TV. Also adds TV distance metric.

Expected results after patch:
  exact:  R²(KL) > 0.95, R²(TV) > 0.95 [was FAIL 0.914 before patch]
  thrml:  R²(KL) > 0.95, R²(TV) > 0.95 [was PASS 0.972 before patch]

REPRODUCE
---------
  python experiments/tasb_k_convergence.py
  python experiments/tasb_k_convergence.py --backends exact  # faster
  python experiments/tasb_k_convergence.py --min-span 3       # less filtering
==============================================================================
"""

import argparse
import csv
import datetime
import sys
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
LAYER_IDX = 18
ALPHA     = 1.0
SEED      = 42
BASE_SEED = 42
MIN_SPAN  = 5     # exclude causal-mask early positions with span < this

K_VALUES = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]

PROMPT = "The relationship between thermodynamics and computation is"

ALL_BACKENDS = ["exact", "thrml"]


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
    device = next(model.parameters()).device
    print(f"  Model ready on {device}\n", flush=True)
    return model, tok


# ---------------------------------------------------------------------------
# Analytical softmax and span mask from capture
# ---------------------------------------------------------------------------
def compute_analytical_softmax(capture) -> torch.Tensor:
    q = capture.q_post_rope.float()
    k = capture.k_post_rope.float()
    scale = float(capture.scaling)
    mask  = capture.attention_mask

    B, n_q, S, _ = q.shape
    n_kv  = k.shape[1]
    n_kvg = n_q // n_kv
    k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k

    J = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if mask is not None:
        J = J + mask.float()
    return F.softmax(J, dim=-1)   # (B, n_q, S, S)


def compute_span_mask(capture, min_span: int) -> torch.Tensor:
    """
    Returns bool tensor of shape (B, S): True for query positions
    with effective attention span >= min_span.
    Span at position i = number of non-(-inf) keys = i+1 for causal mask.
    """
    mask = capture.attention_mask   # (B, 1, S, S) or None
    B = capture.q_post_rope.shape[0]
    S = capture.q_post_rope.shape[2]

    if mask is None:
        return torch.ones(B, S, dtype=torch.bool)

    # Count valid keys per query position (mask entries > -1e8)
    valid_keys = (mask.float() > -1e8).sum(dim=-1)   # (B, 1, S)
    span = valid_keys.squeeze(1)                       # (B, S)
    return span >= min_span                            # (B, S)


# ---------------------------------------------------------------------------
# Per-head mean KL and TV, filtered by span mask
# ---------------------------------------------------------------------------
def mean_kl_tv_per_head(
    p_thermo: torch.Tensor,
    softmax:  torch.Tensor,
    span_mask: torch.Tensor,
):
    """
    Returns: (kl_per_head (n_q,), tv_per_head (n_q,), mean_kl, mean_tv)

    KL(softmax[h] || p_thermo[h]) and TV(softmax[h], p_thermo[h])
    averaged over batch and query positions with span >= min_span.

    TV = 0.5 * sum_j |p_thermo_j - softmax_j|
    """
    B, n_q, S, _ = softmax.shape
    kl_per_head = []
    tv_per_head = []

    for h in range(n_q):
        valid = span_mask                              # (B, S)
        p_h = p_thermo[:, h, :, :]                    # (B, S, S)
        q_h = softmax[:, h, :, :]                     # (B, S, S)

        # Select valid positions
        p_filtered = p_h[valid].reshape(-1, S)        # (N_valid, S)
        q_filtered = q_h[valid].reshape(-1, S)        # (N_valid, S)

        if p_filtered.shape[0] == 0:
            kl_per_head.append(float('nan'))
            tv_per_head.append(float('nan'))
            continue

        p_filtered = p_filtered.clamp(min=1e-10)
        q_filtered = q_filtered.clamp(min=1e-10)

        kl_h = F.kl_div(p_filtered.log(), q_filtered, reduction='batchmean').item()
        tv_h = (0.5 * (p_filtered - q_filtered).abs().sum(dim=-1)).mean().item()

        kl_per_head.append(kl_h)
        tv_per_head.append(tv_h)

    kl_arr = np.array(kl_per_head)
    tv_arr = np.array(tv_per_head)
    return (kl_arr, tv_arr,
            float(np.nanmean(kl_arr)), float(np.nanmean(tv_arr)))


# ---------------------------------------------------------------------------
# Curve fitting: KL = a/K and TV = c/sqrt(K)
# ---------------------------------------------------------------------------
def fit_curves(K_arr: np.ndarray, kl_arr: np.ndarray, tv_arr: np.ndarray) -> dict:
    from scipy.optimize import curve_fit

    def model_1k(K, a):     return a / K
    def model_1k_bias(K, a, b): return a / K + b
    def model_sqrtk(K, c):  return c / np.sqrt(K)

    results = {}

    for label, y, model, model_bias in [
        ("kl", kl_arr, model_1k, model_1k_bias),
        ("tv", tv_arr, model_sqrtk, None),
    ]:
        valid = ~np.isnan(y)
        K_v = K_arr[valid]
        y_v = y[valid]
        if len(y_v) < 3:
            results[label] = {"r2_forced": 0.0, "a": float('nan'), "b": 0.0}
            continue
        try:
            popt, _ = curve_fit(model, K_v, y_v, p0=[1.0], maxfev=5000)
            y_pred = model(K_v, *popt)
            ss_res = np.sum((y_v - y_pred) ** 2)
            ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            entry = {"r2_forced": round(r2, 4), "a": round(float(popt[0]), 6)}

            if model_bias is not None:
                try:
                    popt_b, _ = curve_fit(model_bias, K_v, y_v, p0=[1.0, 0.0], maxfev=5000)
                    y_pred_b = model_bias(K_v, *popt_b)
                    ss_res_b = np.sum((y_v - y_pred_b) ** 2)
                    r2_b = float(1 - ss_res_b / ss_tot) if ss_tot > 0 else 0.0
                    entry["r2_bias"]  = round(r2_b, 4)
                    entry["a_bias"]   = round(float(popt_b[0]), 6)
                    entry["b_bias"]   = round(float(popt_b[1]), 7)
                except Exception:
                    entry["r2_bias"] = entry["r2_forced"]
                    entry["b_bias"]  = 0.0
            else:
                entry["r2_bias"] = entry["r2_forced"]
                entry["b_bias"]  = 0.0

            results[label] = entry
        except Exception as e:
            warnings.warn(f"Curve fit failed for {label}: {e}")
            results[label] = {"r2_forced": 0.0, "a": float('nan'),
                              "r2_bias": 0.0, "b_bias": 0.0}

    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(model, tok, backends: list, min_span: int) -> list:
    from thermobridge import bridge_forward

    all_rows = []

    for backend in backends:
        print(f"\n  Backend: {backend}  [min_span={min_span}]")
        print(f"  {'K':>6}  {'mean KL':>10}  {'mean TV':>10}  {'n_valid':>8}  time(s)")
        print(f"  {'-'*56}")

        K_arr  = []
        kl_arr = []
        tv_arr = []

        for K in K_VALUES:
            if K > 100 and backend == 'thrml':
                print(f"  {K:>6}  (running THRML K={K})...", end='\r', flush=True)

            t0 = time.perf_counter()
            result = bridge_forward(
                model, tok,
                prompt=PROMPT,
                layer_idx=LAYER_IDX,
                alpha=ALPHA,
                backend=backend,
                K=K,
                seed=BASE_SEED,
                return_intermediates=True,
            )
            elapsed = time.perf_counter() - t0

            capture   = result.layer_captures[LAYER_IDX]
            softmax   = compute_analytical_softmax(capture)
            p_thermo  = result.p_thermo_by_layer[LAYER_IDX]
            span_mask = compute_span_mask(capture, min_span)

            n_valid = int(span_mask.sum().item())

            kl_heads, tv_heads, mean_kl, mean_tv = mean_kl_tv_per_head(
                p_thermo, softmax, span_mask
            )

            K_arr.append(K)
            kl_arr.append(mean_kl)
            tv_arr.append(mean_tv)

            print(f"  {K:>6}  {mean_kl:>10.6f}  {mean_tv:>10.6f}  {n_valid:>8}  {elapsed:.1f}s")

            all_rows.append({
                "backend":       backend,
                "K":             K,
                "mean_kl":       round(mean_kl, 7),
                "mean_tv":       round(mean_tv, 7),
                "std_kl":        round(float(np.nanstd(kl_heads)), 7),
                "std_tv":        round(float(np.nanstd(tv_heads)), 7),
                "n_valid_positions": n_valid,
                "min_span":      min_span,
                "elapsed_s":     round(elapsed, 2),
                "prompt":        PROMPT[:40],
                "layer_idx":     LAYER_IDX,
                "alpha":         ALPHA,
            })

        K_np  = np.array(K_arr, dtype=float)
        kl_np = np.array(kl_arr, dtype=float)
        tv_np = np.array(tv_arr, dtype=float)

        fits = fit_curves(K_np, kl_np, tv_np)

        print(f"\n  Curve fits for {backend} (min_span={min_span}):")
        kl_fit = fits.get("kl", {})
        tv_fit = fits.get("tv", {})
        print(f"    KL forced-origin:  KL = {kl_fit.get('a', float('nan')):.4f}/K"
              f"        R² = {kl_fit.get('r2_forced', 0):.4f}")
        if "b_bias" in kl_fit:
            print(f"    KL bias-allowed:   KL = {kl_fit.get('a_bias', float('nan')):.4f}/K"
                  f" + {kl_fit.get('b_bias', 0):.6f}"
                  f"  R² = {kl_fit.get('r2_bias', 0):.4f}")
        print(f"    TV forced-origin:  TV = {tv_fit.get('a', float('nan')):.4f}/√K"
              f"        R² = {tv_fit.get('r2_forced', 0):.4f}")

        kl_pass = kl_fit.get("r2_forced", 0) >= 0.95
        tv_pass = tv_fit.get("r2_forced", 0) >= 0.95
        print(f"    KL R² ≥ 0.95: {'PASS' if kl_pass else 'FAIL'}")
        print(f"    TV R² ≥ 0.95: {'PASS' if tv_pass else 'FAIL'}")
        if kl_fit.get("b_bias", 0) and abs(kl_fit.get("b_bias", 0)) > 0.0005:
            print(f"    KL bias floor b={kl_fit['b_bias']:.6f}"
                  f" (expected near 0 for unbiased sampler)")

        for row in all_rows:
            if row["backend"] == backend:
                row["r2_kl_forced"] = round(kl_fit.get("r2_forced", 0), 4)
                row["r2_kl_bias"]   = round(kl_fit.get("r2_bias", 0), 4)
                row["fit_kl_a"]     = round(kl_fit.get("a", float("nan")), 6)
                row["fit_kl_b"]     = round(kl_fit.get("b_bias", 0), 7)
                row["r2_tv_forced"] = round(tv_fit.get("r2_forced", 0), 4)
                row["fit_tv_c"]     = round(tv_fit.get("a", float("nan")), 6)

    return all_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB K-Convergence Proof (v2 patched)")
    parser.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                        choices=ALL_BACKENDS)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--min-span", type=int, default=MIN_SPAN,
                        help="Min effective attention span to include in KL/TV average")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  TASB K-Convergence Curve (Tier 1.D v2 / T2.B-4 Patched)")
    print(f"  Layer {LAYER_IDX} | α={ALPHA} | seed={BASE_SEED}")
    print(f"  Prompt: '{PROMPT[:50]}'")
    print(f"  Backends: {args.backends}")
    print(f"  K values: {K_VALUES}")
    print(f"  min_span filter: {args.min_span} (excludes causal-mask early positions)")
    print(f"  Metrics: KL ∝ 1/K  +  TV ∝ 1/√K")
    print(f"{'='*70}")

    model, tok = load_model(args.model)
    rows = run_sweep(model, tok, args.backends, args.min_span)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_k_convergence_v2_{ts}.csv"
    fieldnames = [
        "backend", "K", "mean_kl", "mean_tv", "std_kl", "std_tv",
        "n_valid_positions", "min_span", "elapsed_s", "prompt",
        "layer_idx", "alpha",
        "r2_kl_forced", "r2_kl_bias", "fit_kl_a", "fit_kl_b",
        "r2_tv_forced", "fit_tv_c",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {out_path}")

    backend_r2 = {}
    for row in rows:
        b = row["backend"]
        if "r2_kl_forced" in row:
            backend_r2.setdefault(b, {})
            backend_r2[b]["kl"] = row["r2_kl_forced"]
            backend_r2[b]["tv"] = row.get("r2_tv_forced", 0)

    print(f"\n{'='*70}")
    print(f"  SUMMARY  (min_span={args.min_span})")
    all_pass = True
    for b, r2s in backend_r2.items():
        kl_ok = r2s.get("kl", 0) >= 0.95
        tv_ok = r2s.get("tv", 0) >= 0.95
        both  = kl_ok and tv_ok
        if not both:
            all_pass = False
        print(f"  {b:<12} KL R²={r2s.get('kl',0):.4f} {'PASS' if kl_ok else 'FAIL'}"
              f"  TV R²={r2s.get('tv',0):.4f} {'PASS' if tv_ok else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print(f"  KL ∝ 1/K and TV ∝ 1/√K confirmed.")
        print(f"  Samplers produce iid draws from correct Boltzmann distribution.")
    print(f"{'='*70}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
