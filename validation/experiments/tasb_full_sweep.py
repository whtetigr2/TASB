import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_full_sweep.py — Full Thermodynamic Landscape: All Layers × All Heads × 5 Prompts
==============================================================================
TASB Demo Data Collection
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
Collects four thermodynamic observables per (prompt, layer, head) across all
28 layers and 24 heads of LLaMA 3.2-3B. Primary purpose: feed the TASB demo
with static real-model data for heatmaps and prompt-comparison panels.

Four observables per (prompt, layer, head):

  cv      = Var_softmax(logits)            [Kim 2026 specific heat, analytical]
  kl      = KL(softmax || p_thermo)        [sampling fidelity at K=10]
  entropy = -sum(p * log(p))               [Shannon entropy, nats]
  p_max   = max(softmax)                   [attention concentration]

Settings consistent with the validated r(Cv,KL) = 0.8241 result (FIND-022):
  alpha=1.0, backend='exact', K=10, layer_idx=[0..27]

One bridge_forward pass per prompt (all 28 layers captured simultaneously):
  5 prompts × 1 pass = 5 total forward passes.
  Output: 5 × 28 × 24 = 3,360 rows — tidy CSV, one row per (prompt, layer, head).

REPRODUCE
---------
  python experiments/tasb_full_sweep.py
  python experiments/tasb_full_sweep.py --prompt factual   # single-prompt fast mode
==============================================================================
"""

import argparse
import csv
import datetime
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
ALL_LAYERS = list(range(28))
ALPHA      = 1.0        # full bridge — consistent with FIND-022 validated result
K          = 10         # production K
SEED       = 42

PROMPTS = {
    "factual":      "The capital of France is Paris. The capital of Germany is",
    "code":         "def fibonacci(n): if n <= 1: return n else: return fibonacci",
    "reasoning":    "All mammals are warm-blooded. Whales are mammals. Therefore,",
    "creative":     "In the year 2150, when gravity was finally understood as",
    "mathematical": "The Riemann zeta function zeta(s) has non-trivial zeros at s =",
}


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
# Logit computation from capture
# ---------------------------------------------------------------------------
def compute_logits(capture) -> torch.Tensor:
    from transformers.models.llama.modeling_llama import repeat_kv
    Q = capture.q_post_rope.float()
    K = capture.k_post_rope.float()
    n_q, n_kv = Q.shape[1], K.shape[1]
    K_rep = repeat_kv(K, n_q // n_kv)
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.float()
    return logits  # (B, n_q, S, S)


# ---------------------------------------------------------------------------
# Per-head metrics for one layer
# ---------------------------------------------------------------------------
def per_head_metrics(logits: torch.Tensor,
                     p_thermo: torch.Tensor) -> dict:
    """
    Returns dict of arrays, each shape (n_q,).

    cv      = Var_softmax(logits) — analytical, positions 1..S-1 averaged
    kl      = KL(softmax || p_thermo) per head
    entropy = Shannon entropy of softmax per head (nats)
    p_max   = max(softmax) per head

    Positions 0 is excluded from cv/entropy/p_max (trivially attends only itself).
    """
    p = torch.softmax(logits, dim=-1)  # (B, n_q, S, S)
    n_q = logits.shape[1]
    S   = logits.shape[-1]

    cv      = np.zeros(n_q)
    kl      = np.zeros(n_q)
    entropy = np.zeros(n_q)
    p_max   = np.zeros(n_q)

    for h in range(n_q):
        log_h = logits[0, h, 1:, :]   # (S-1, S) — skip position 0
        p_h   = p[0,   h, 1:, :]      # (S-1, S)

        log_h = log_h.clamp(min=-1e9)  # HF causal sentinel overflows float32 when squared

        # Cv = Var_softmax(logits)
        E1    = (p_h * log_h).sum(-1)
        E2    = (p_h * log_h**2).sum(-1)
        cv[h] = (E2 - E1**2).mean().item()

        # Shannon entropy (nats), excluding position 0
        p_clamped   = p_h.clamp(min=1e-10)
        ent_per_pos = -(p_clamped * p_clamped.log()).sum(-1)
        entropy[h]  = ent_per_pos.mean().item()

        # p_max: mean of max-token probability per position
        p_max[h] = p_h.max(dim=-1).values.mean().item()

        # KL(softmax || p_thermo) — sampling fidelity at K=10
        pt_h = p_thermo[0, h, 1:, :].float().clamp(min=1e-10)
        pt_h = pt_h / pt_h.sum(-1, keepdim=True)
        q_clamped = p_h.clamp(min=1e-10)
        kl[h] = F.kl_div(pt_h.log(), q_clamped, reduction='batchmean').item()

    return {"cv": cv, "kl": kl, "entropy": entropy, "p_max": p_max}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(model, tok, prompts: dict) -> list:
    from thermobridge import bridge_forward

    all_rows = []

    for prompt_id, prompt in prompts.items():
        print(f"\n  Prompt: {prompt_id!r} — '{prompt[:55]}...'", flush=True)

        t0 = time.perf_counter()
        result = bridge_forward(
            model, tok,
            prompt=prompt,
            layer_idx=ALL_LAYERS,
            alpha=ALPHA,
            backend='exact',
            K=K,
            seed=SEED,
            return_intermediates=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Forward pass: {elapsed:.1f}s  |  "
              f"captured {len(result.layer_captures)} layers", flush=True)

        print(f"  {'layer':>5}  {'mean_cv':>8}  {'mean_kl':>8}  "
              f"{'mean_H':>7}  {'mean_pmax':>9}")
        print(f"  {'-'*50}")

        for layer in ALL_LAYERS:
            capture  = result.layer_captures[layer]
            p_thermo = result.p_thermo_by_layer[layer]
            logits   = compute_logits(capture)
            metrics  = per_head_metrics(logits, p_thermo)

            cv      = metrics["cv"]
            kl_vals = metrics["kl"]
            ent     = metrics["entropy"]
            pmax    = metrics["p_max"]

            print(f"  {layer:>5}  {cv.mean():>8.4f}  {kl_vals.mean():>8.5f}  "
                  f"{ent.mean():>7.4f}  {pmax.mean():>9.4f}", flush=True)

            for h in range(len(cv)):
                all_rows.append({
                    "prompt_id": prompt_id,
                    "layer":     layer,
                    "head":      h,
                    "cv":        round(float(cv[h]),      6),
                    "kl":        round(float(kl_vals[h]), 7),
                    "entropy":   round(float(ent[h]),     6),
                    "p_max":     round(float(pmax[h]),    6),
                })

    return all_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB Full Thermodynamic Sweep")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt", choices=list(PROMPTS.keys()),
                        help="Run a single prompt (default: all 5)")
    args = parser.parse_args()

    prompts = ({args.prompt: PROMPTS[args.prompt]} if args.prompt
               else PROMPTS)

    n_rows_expected = len(prompts) * 28 * 24

    print(f"\n{'='*70}")
    print(f"  TASB Full Thermodynamic Sweep — Demo Data Collection")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {list(prompts.keys())}")
    print(f"  All 28 layers × 24 heads | alpha={ALPHA} | backend=exact | K={K}")
    print(f"  Expected output rows: {n_rows_expected} (prompt × layer × head)")
    print(f"  Metrics: cv, kl, entropy, p_max")
    print(f"{'='*70}")

    model, tok = load_model(args.model)
    rows = run_sweep(model, tok, prompts)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_full_sweep_{ts}.csv"

    fieldnames = ["prompt_id", "layer", "head", "cv", "kl", "entropy", "p_max"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {out_path}  ({len(rows)} rows)")

    # Quick sanity report
    import collections
    cv_by_layer = collections.defaultdict(list)
    for row in rows:
        cv_by_layer[row["layer"]].append(row["cv"])

    print(f"\n  Layer Cv profile (mean across heads × prompts):")
    for layer in [0, 4, 9, 13, 18, 22, 27]:
        vals = cv_by_layer[layer]
        print(f"    Layer {layer:>2}: mean_cv = {np.mean(vals):.4f}  "
              f"std = {np.std(vals):.4f}  range [{np.min(vals):.3f}, {np.max(vals):.3f}]")

    print(f"\n{'='*70}")
    print(f"  Done. {len(rows)} rows → {out_path}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
