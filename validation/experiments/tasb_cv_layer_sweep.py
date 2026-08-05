import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_cv_layer_sweep.py — Specific Heat Cv Across All 28 Layers
==============================================================================
TASB Thermodynamic Validation — Kim (2026) Inference-Time Cv Profile
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
Kim (2026, arXiv:2602.08216) derives Cv = Var_ρ(attention_logits) as the
thermodynamic specific heat of each attention layer. His training-time result:
Cv peaks before grokking, then decreases as weights converge.

At inference on a frozen pretrained model, the prediction is:
    Cv(early layers) > Cv(mid layers) > Cv(deep layers)

because deeper layers have larger trained weight norms → smaller effective
temperature T_eff = √d_k / ‖W_QK‖² → more concentrated attention → lower Cv.

This is the FIRST measurement of Kim's Cv observable at inference time on a
frozen pretrained transformer. Kim measured this during training only.

FORMULA (exact, no sampling needed)
-------------------------------------
    logits = Q @ K^T * (1/√d_k) + causal_mask
    p      = softmax(logits)              (exact Boltzmann distribution)
    Cv[h]  = E_p[logits²] - E_p[logits]² (variance of logits under p)

Cv is computed analytically from the captured QK matrices — no sampling bias.
Averaged over sequence positions 1..S-1 (position 0 trivially has Cv=0).

PASS CONDITIONS (FIND-020)
--------------------------
    CONFIRM: mean_Cv(layers 0-4) > mean_Cv(layer 18) > mean_Cv(layers 23-27)
    CONFIRM: per-head Cv at layer 18 ∈ [0.1, 2.5] for most heads
    FLAG:    any head Cv > 10 consistently (suggests computation bug)

REPRODUCE
---------
    python experiments/tasb_cv_layer_sweep.py
    python experiments/tasb_cv_layer_sweep.py --prompt-only   # fast, 1 prompt
==============================================================================
"""

import argparse
import collections
import csv
import datetime
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
ALL_LAYERS = list(range(28))
ALPHA      = 0.0    # capture only — no injection
K          = 10     # minimal; Cv is computed analytically from captures
SEED       = 42

PROMPTS = {
    "factual":   "The capital of France is Paris. The capital of Germany is",
    "code":      "def fibonacci(n): if n <= 1: return n else: return fibonacci",
    "reasoning": "All mammals are warm-blooded. Whales are mammals. Therefore,",
    "creative":  "In the year 2150, when gravity was finally understood as",
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
# Cv computation (analytical — no sampling)
# ---------------------------------------------------------------------------
def compute_logits_from_capture(capture) -> torch.Tensor:
    """Compute scaled attention logits from a LayerCapture.

    Mirrors thermobridge.sampler._compute_logits without importing private API.
    Returns (B, n_q, S, S) float32.
    """
    from transformers.models.llama.modeling_llama import repeat_kv
    Q = capture.q_post_rope.float()
    K = capture.k_post_rope.float()
    n_q, n_kv = Q.shape[1], K.shape[1]
    K_rep = repeat_kv(K, n_q // n_kv)
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.float()
    return logits  # (B, n_q, S, S)


def compute_cv_per_head(capture) -> np.ndarray:
    """Exact Cv = Var_softmax(logits) per head, averaged over positions 1..S-1.

    Position 0 always attends only to itself (trivially Cv=0) and is excluded.
    Returns array of shape (n_q,).
    """
    logits = compute_logits_from_capture(capture)  # (1, n_q, S, S)
    p = torch.softmax(logits, dim=-1)              # (1, n_q, S, S)

    n_q = logits.shape[1]
    cv_heads = np.zeros(n_q)

    for h in range(n_q):
        log_h = logits[0, h, 1:, :]   # (S-1, S) — skip position 0
        p_h   = p[0,   h, 1:, :]      # (S-1, S)
        log_h = log_h.clamp(min=-1e9)  # HF causal sentinel (~-3.4e38) overflows float32 when squared
        E_a   = (p_h * log_h).sum(-1)       # (S-1,)
        E_a2  = (p_h * log_h**2).sum(-1)    # (S-1,)
        cv_heads[h] = (E_a2 - E_a**2).mean().item()

    return cv_heads


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(model, tok, prompts: dict) -> list:
    from thermobridge import bridge_forward

    all_rows = []

    for prompt_id, prompt in prompts.items():
        print(f"\n  Prompt: {prompt_id!r} — '{prompt[:60]}...'", flush=True)

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
        print(f"  Capture pass: {elapsed:.1f}s")

        print(f"\n  {'layer':>5}  {'mean_cv':>8}  {'std_cv':>7}  "
              f"{'min_cv':>7}  {'max_cv':>7}")
        print(f"  {'-'*45}")

        for layer in ALL_LAYERS:
            capture = result.layer_captures[layer]
            cv = compute_cv_per_head(capture)

            print(f"  {layer:>5}  {cv.mean():>8.4f}  {cv.std():>7.4f}  "
                  f"{cv.min():>7.4f}  {cv.max():>7.4f}")

            row = {
                "layer":     layer,
                "prompt_id": prompt_id,
                "mean_cv":   round(float(cv.mean()), 6),
                "std_cv":    round(float(cv.std()),  6),
                "min_cv":    round(float(cv.min()),  6),
                "max_cv":    round(float(cv.max()),  6),
                "elapsed_s": round(elapsed, 2),
            }
            row.update({f"head_{h:02d}_cv": round(float(cv[h]), 6)
                        for h in range(len(cv))})
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB Cv Layer Sweep")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt-only", action="store_true",
                        help="Run only the factual prompt (fast mode)")
    args = parser.parse_args()

    prompts = ({"factual": PROMPTS["factual"]} if args.prompt_only
               else PROMPTS)

    print(f"\n{'='*70}")
    print(f"  TASB Cv Layer Sweep — Kim (2026) Inference-Time Specific Heat")
    print(f"  Model: {args.model}")
    print(f"  All 28 layers | analytical Cv | alpha=0 (capture only)")
    print(f"  Prediction: Cv(early) > Cv(mid) > Cv(deep)")
    print(f"{'='*70}")

    model, tok = load_model(args.model)
    rows = run_sweep(model, tok, prompts)

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_cv_layer_sweep_{ts}.csv"

    n_heads = 24
    fieldnames = (
        ["layer", "prompt_id", "elapsed_s",
         "mean_cv", "std_cv", "min_cv", "max_cv"] +
        [f"head_{h:02d}_cv" for h in range(n_heads)]
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {out_path}")

    # Verdict
    layer_means: dict = collections.defaultdict(list)
    for row in rows:
        layer_means[row["layer"]].append(row["mean_cv"])

    avg_early = np.mean([np.mean(layer_means[l]) for l in range(5)])
    avg_mid   = np.mean([np.mean(layer_means[l]) for l in [18]])
    avg_deep  = np.mean([np.mean(layer_means[l]) for l in range(23, 28)])

    print(f"\n{'='*70}")
    print(f"  VERDICT — Kim (2026) layer-depth Cv ordering")
    print(f"  Early layers (0–4):   mean Cv = {avg_early:.4f}")
    print(f"  Mid   layer  (18):    mean Cv = {avg_mid:.4f}")
    print(f"  Deep  layers (23–27): mean Cv = {avg_deep:.4f}")
    ordered = avg_early > avg_mid > avg_deep
    print(f"  Early > Mid > Deep: {'CONFIRMED ✓' if ordered else 'NOT CONFIRMED — review CSV'}")
    print(f"{'='*70}\n")

    return 0 if ordered else 1


if __name__ == "__main__":
    sys.exit(main())
