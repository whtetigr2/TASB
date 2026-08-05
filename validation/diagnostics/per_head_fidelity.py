# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_per_head_fidelity.py — Per-Head Boltzmann Fidelity Breakdown
==============================================================================
TASB Validation Suite — Tier 1.C
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
Prior validation (M5–M7, BUG_REGISTRY #24) reports GLOBAL KL averages:
mean KL across all heads, positions, and prompts. Global averages can hide
systematic per-head errors — if head 7 always diverges while the others are
perfect, the average stays low.

This test breaks out KL(analytical_softmax || p_thermo) and TV(softmax, p_thermo)
per head for each backend. For LLaMA 3.2-3B at L18: 24 query heads.
Expected result: uniform fidelity across all heads.

WHAT WE ARE LOOKING FOR
-----------------------
  PASS: std(per-head KL) < 0.001 at K=5000 — fidelity is head-uniform
  FLAG: Any single head with KL > 5x median — systematic per-head bias
  FLAG: Head-specific pattern correlates with attention head function
        (e.g., induction heads, positional heads) — might indicate
        structure-dependent sampling failure

  TV(P, Q) = 0.5 * sum_i |P_i - Q_i|  (range [0, 1])
  Provides a second, distribution-free faithfulness metric alongside KL.

DESIGN
------
  - Layer 18, alpha=1.0 (full substitution — measures sampler, not blend)
  - K=5000 (converged finite-K floor — isolates distributional vs. sampling error)
  - 4 prompts covering different domains (factual, code, reasoning, creative)
  - 4 backends: exact, gumbel, rbm, thrml

REPRODUCE
---------
  python diagnostics/tasb_per_head_fidelity.py
  python diagnostics/tasb_per_head_fidelity.py --prompt-only  # quick, 1 prompt
==============================================================================
"""

import argparse
import csv
import datetime
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
LAYER_IDX = 18
ALPHA     = 1.0     # full substitution
K         = 5000    # converged — isolates distribution, not finite-K noise
SEED      = 42

PROMPTS = {
    "factual":   "The capital of France is Paris. The capital of Germany is",
    "code":      "def fibonacci(n): if n <= 1: return n else: return fibonacci",
    "reasoning": "All mammals are warm-blooded. Whales are mammals. Therefore,",
    "creative":  "In the year 2150, when gravity was finally understood as",
}

ALL_BACKENDS = ["exact", "gumbel", "rbm", "thrml"]


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
# Compute analytical softmax from capture
# ---------------------------------------------------------------------------
def compute_analytical_softmax(capture) -> torch.Tensor:
    q = capture.q_post_rope.float()   # (B, n_q, S, head_dim)
    k = capture.k_post_rope.float()   # (B, n_kv, S, head_dim)
    scale = float(capture.scaling)
    mask  = capture.attention_mask

    B, n_q, S, _ = q.shape
    n_kv = k.shape[1]
    n_kvg = n_q // n_kv

    k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k
    J = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if mask is not None:
        J = J + mask.float()
    return F.softmax(J, dim=-1)   # (B, n_q, S, S)


# ---------------------------------------------------------------------------
# Per-head KL + TV computation
# ---------------------------------------------------------------------------
def per_head_metrics(p_thermo: torch.Tensor, softmax_ref: torch.Tensor) -> tuple:
    """
    Returns (kl_per_head, tv_per_head), each shape (n_q,).

    KL(softmax[h] || p_thermo[h]) — how surprising is softmax under the
    thermodynamic model, averaged over batch and sequence positions.

    TV(softmax[h], p_thermo[h]) = 0.5 * sum_i |P_i - Q_i| — range [0, 1].
    Provides a distribution-free complement to KL.
    """
    B, n_q, S, _ = softmax_ref.shape
    kl_per_head = np.zeros(n_q, dtype=np.float64)
    tv_per_head = np.zeros(n_q, dtype=np.float64)

    for h in range(n_q):
        p_h = p_thermo[:, h, :, :].reshape(-1, S).clamp(min=1e-10)
        q_h = softmax_ref[:, h, :, :].reshape(-1, S).clamp(min=1e-10)
        kl = F.kl_div(p_h.log(), q_h, reduction='batchmean').item()
        tv = (0.5 * torch.abs(q_h - p_h).sum(dim=-1)).mean().item()
        kl_per_head[h] = kl
        tv_per_head[h] = tv

    return kl_per_head, tv_per_head


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(model, tok, prompts: dict, backends: list) -> list:
    from thermobridge import bridge_forward

    all_rows = []

    for backend in backends:
        print(f"\n  Backend: {backend}")
        print(f"  {'head':>5}  ", end='')
        for pid in prompts:
            print(f"{pid:>10}", end='')
        print(f"  {'mean_kl':>8}  {'mean_tv':>8}  {'flag':>6}")
        print(f"  {'-'*70}")

        per_head_kl_acc = {}   # head_idx -> list of KL values
        per_head_tv_acc = {}   # head_idx -> list of TV values

        for prompt_id, prompt in prompts.items():
            t0 = time.perf_counter()
            result = bridge_forward(
                model, tok,
                prompt=prompt,
                layer_idx=LAYER_IDX,
                alpha=ALPHA,
                backend=backend,
                K=K,
                seed=SEED,
                return_intermediates=True,
            )
            elapsed = time.perf_counter() - t0

            capture  = result.layer_captures[LAYER_IDX]
            softmax  = compute_analytical_softmax(capture)
            p_thermo = result.p_thermo_by_layer[LAYER_IDX]

            kl_heads, tv_heads = per_head_metrics(p_thermo, softmax)

            for h, (kl, tv) in enumerate(zip(kl_heads, tv_heads)):
                per_head_kl_acc.setdefault(h, []).append(kl)
                per_head_tv_acc.setdefault(h, []).append(tv)

            all_rows.append({
                "backend":   backend,
                "prompt_id": prompt_id,
                "elapsed_s": round(elapsed, 2),
                **{f"head_{h:02d}_kl": round(float(kl), 7)
                   for h, kl in enumerate(kl_heads)},
                **{f"head_{h:02d}_tv": round(float(tv), 7)
                   for h, tv in enumerate(tv_heads)},
                "mean_kl": round(float(kl_heads.mean()), 7),
                "std_kl":  round(float(kl_heads.std()),  7),
                "max_kl":  round(float(kl_heads.max()),  7),
                "min_kl":  round(float(kl_heads.min()),  7),
                "mean_tv": round(float(tv_heads.mean()), 7),
                "std_tv":  round(float(tv_heads.std()),  7),
                "max_tv":  round(float(tv_heads.max()),  7),
                "min_tv":  round(float(tv_heads.min()),  7),
            })

        # Print per-head summary across prompts
        n_heads = len(per_head_kl_acc)
        median_kl = np.median([np.mean(v) for v in per_head_kl_acc.values()])

        for h in range(n_heads):
            kl_vals = per_head_kl_acc[h]
            mean_kl_h = np.mean(kl_vals)
            mean_tv_h = np.mean(per_head_tv_acc[h])
            flag = "OUTLIER" if mean_kl_h > 5 * median_kl else ""
            print(f"  {h:>5}  ", end='')
            for v in kl_vals:
                print(f"{v:>10.5f}", end='')
            print(f"  {mean_kl_h:>8.5f}  {mean_tv_h:>8.5f}  {flag:>6}")

        # Summary stats
        all_kl_means = [np.mean(v) for v in per_head_kl_acc.values()]
        all_tv_means = [np.mean(v) for v in per_head_tv_acc.values()]
        overall_kl_mean = np.mean(all_kl_means)
        overall_kl_std  = np.std(all_kl_means)
        overall_tv_mean = np.mean(all_tv_means)
        overall_tv_std  = np.std(all_tv_means)
        outliers        = [h for h, v in enumerate(all_kl_means) if v > 5 * median_kl]

        print(f"\n  {backend} summary:")
        print(f"    KL : mean={overall_kl_mean:.5f}  std={overall_kl_std:.5f}  "
              f"{'PASS' if overall_kl_std < 0.001 else 'WARN'} (threshold std<0.001)")
        print(f"    TV : mean={overall_tv_mean:.5f}  std={overall_tv_std:.5f}")
        if outliers:
            print(f"    HEADS {outliers} show >5x median KL — investigate!")

    return all_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB Per-Head Fidelity")
    parser.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                        choices=ALL_BACKENDS)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt-only", action="store_true",
                        help="Run only the factual prompt (fast mode)")
    args = parser.parse_args()

    prompts = ({"factual": PROMPTS["factual"]} if args.prompt_only
               else PROMPTS)

    print(f"\n{'='*70}")
    print(f"  TASB Per-Head Fidelity Breakdown (Tier 1.C Validation)")
    print(f"  Layer {LAYER_IDX} | alpha={ALPHA} | K={K} | seed={SEED}")
    print(f"  Backends: {args.backends}")
    print(f"  Prompts:  {list(prompts.keys())}")
    print(f"{'='*70}")

    model, tok = load_model(args.model)
    rows = run_sweep(model, tok, prompts, args.backends)

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_per_head_fidelity_{ts}.csv"

    n_heads = 24
    head_kl_fields = [f"head_{h:02d}_kl" for h in range(n_heads)]
    head_tv_fields = [f"head_{h:02d}_tv" for h in range(n_heads)]
    fieldnames = (
        ["backend", "prompt_id", "elapsed_s"] +
        head_kl_fields +
        head_tv_fields +
        ["mean_kl", "std_kl", "max_kl", "min_kl",
         "mean_tv", "std_tv", "max_tv", "min_tv"]
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {out_path}")

    # Overall verdict
    std_by_backend = {}
    for row in rows:
        b = row['backend']
        std_by_backend.setdefault(b, []).append(row.get('std_kl', 1.0))

    print(f"\n{'='*70}")
    print(f"  SUMMARY — std(per-head KL) and mean TV across all prompts")
    all_pass = True
    for b, stds in std_by_backend.items():
        mean_std = np.mean(stds)
        ok = mean_std < 0.001
        if not ok:
            all_pass = False
        tv_vals = [r.get('mean_tv', 0) for r in rows if r['backend'] == b]
        mean_tv = np.mean(tv_vals) if tv_vals else float('nan')
        print(f"  {b:<12} KL std={mean_std:.5f}  TV mean={mean_tv:.5f}  "
              f"{'PASS' if ok else 'WARN — head heterogeneity detected'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'WARN'}")
    print(f"{'='*70}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
