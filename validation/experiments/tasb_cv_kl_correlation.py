import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_cv_kl_correlation.py — Cv × KL Pearson Correlation Across Heads
==============================================================================
TASB Thermodynamic Validation — FIND-020 Correlation Prediction
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
FIND-020 predicts: across the 24 attention heads at layer 18, heads with
higher thermodynamic specific heat Cv = Var_softmax(logits) should also
have higher KL divergence KL(softmax || p_thermo_K=10).

Physical reasoning:
  - LOW Cv  → concentrated attention (few tokens dominate)
              → K=10 samples nearly always land on the right token
              → KL ≈ 0
  - HIGH Cv → diffuse attention (many tokens compete)
              → K=10 samples sample different subsets each time
              → KL is larger

Prediction: Pearson r(Cv, KL) > 0.70 across all 24 heads × 4 prompts.

FORMULA
-------
    Cv[h]  = Var_softmax(logits)[h]    (analytical, exact, zero-bias)
    KL[h]  = KL(softmax[h] || p_thermo[h])  at K=10, exact backend

PASS CONDITION (FIND-020 §Q2)
------------------------------
    PASS: Pearson r(Cv, KL) > 0.70  (96 data points: 4 prompts × 24 heads)

REPRODUCE
---------
    python experiments/tasb_cv_kl_correlation.py
    python experiments/tasb_cv_kl_correlation.py --prompt-only   # fast
==============================================================================
"""

import argparse
import csv
import datetime
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
LAYER_IDX = 18
ALPHA     = 1.0   # full substitution — measures sampler, not blend
K         = 10    # production K (tests real-world fidelity)
SEED      = 42

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
    return model, tok


# ---------------------------------------------------------------------------
# Per-head metrics
# ---------------------------------------------------------------------------
def compute_logits(capture) -> torch.Tensor:
    from transformers.models.llama.modeling_llama import repeat_kv
    Q = capture.q_post_rope.float()
    K = capture.k_post_rope.float()
    K_rep = repeat_kv(K, Q.shape[1] // K.shape[1])
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.float()
    return logits


def cv_per_head(logits: torch.Tensor) -> np.ndarray:
    """Analytical Cv per head, positions 1..S-1."""
    p = torch.softmax(logits, dim=-1)
    n_q = logits.shape[1]
    cv = np.zeros(n_q)
    for h in range(n_q):
        log_h = logits[0, h, 1:, :]
        p_h   = p[0,   h, 1:, :]
        log_h = log_h.masked_fill(~torch.isfinite(log_h), 0.0)
        E1 = (p_h * log_h).sum(-1)
        E2 = (p_h * log_h**2).sum(-1)
        cv[h] = (E2 - E1**2).mean().item()
    return cv


def kl_per_head(p_thermo: torch.Tensor,
                softmax_ref: torch.Tensor) -> np.ndarray:
    """KL(softmax[h] || p_thermo[h]) per head, averaged over batch×positions."""
    B, n_q, S, _ = softmax_ref.shape
    kl = np.zeros(n_q)
    for h in range(n_q):
        p_h = p_thermo[:, h, :, :].reshape(-1, S).clamp(min=1e-10)
        q_h = softmax_ref[:, h, :, :].reshape(-1, S).clamp(min=1e-10)
        kl[h] = F.kl_div(p_h.log(), q_h, reduction='batchmean').item()
    return kl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from thermobridge import bridge_forward

    parser = argparse.ArgumentParser(description="TASB Cv×KL Correlation")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt-only", action="store_true",
                        help="Run only the factual prompt (fast mode)")
    args = parser.parse_args()

    prompts = ({"factual": PROMPTS["factual"]} if args.prompt_only
               else PROMPTS)

    print(f"\n{'='*70}")
    print(f"  TASB Cv × KL Pearson Correlation — FIND-020 Prediction r > 0.70")
    print(f"  Layer {LAYER_IDX} | K={K} | exact backend | {len(prompts)} prompts × 24 heads")
    print(f"{'='*70}")

    model, tok = load_model(MODEL_ID)

    cv_all, kl_all = [], []
    rows = []

    for prompt_id, prompt in prompts.items():
        print(f"\n  Prompt: {prompt_id!r}", flush=True)

        result = bridge_forward(
            model, tok,
            prompt=prompt,
            layer_idx=LAYER_IDX,
            alpha=ALPHA,
            backend='exact',
            K=K,
            seed=SEED,
            return_intermediates=True,
        )

        capture  = result.layer_captures[LAYER_IDX]
        p_thermo = result.p_thermo_by_layer[LAYER_IDX]
        logits   = compute_logits(capture)
        softmax  = torch.softmax(logits, dim=-1)

        cv = cv_per_head(logits)
        kl = kl_per_head(p_thermo, softmax)

        print(f"  Cv: min={cv.min():.4f}  max={cv.max():.4f}  "
              f"mean={cv.mean():.4f}  std={cv.std():.4f}")
        print(f"  KL: min={kl.min():.5f}  max={kl.max():.5f}  "
              f"mean={kl.mean():.5f}  std={kl.std():.5f}")

        for h in range(len(cv)):
            cv_all.append(float(cv[h]))
            kl_all.append(float(kl[h]))
            rows.append({
                "prompt_id": prompt_id,
                "head":      h,
                "cv":        round(float(cv[h]), 6),
                "kl":        round(float(kl[h]), 7),
            })

    # Pearson correlation
    cv_arr = np.array(cv_all)
    kl_arr = np.array(kl_all)

    # scipy may not be available — implement Pearson manually as fallback
    try:
        from scipy import stats
        r, p_val = stats.pearsonr(cv_arr, kl_arr)
    except ImportError:
        cv_z = (cv_arr - cv_arr.mean()) / (cv_arr.std() + 1e-12)
        kl_z = (kl_arr - kl_arr.mean()) / (kl_arr.std() + 1e-12)
        r = float((cv_z * kl_z).mean())
        p_val = float('nan')
        print("  (scipy not available — p-value not computed)")

    r_pass = r > 0.70
    n_pts  = len(cv_arr)

    print(f"\n{'='*70}")
    print(f"  VERDICT — Pearson r(Cv, KL)")
    print(f"  n = {n_pts} data points ({len(prompts)} prompts × 24 heads)")
    print(f"  r = {r:.4f}   p = {p_val:.4e}")
    print(f"  Threshold: r > 0.70")
    print(f"  Result: {'CONFIRMED ✓' if r_pass else 'NOT CONFIRMED ✗'}")
    if not r_pass and not np.isnan(r):
        print(f"  *** r = {r:.4f} below threshold")
        print(f"      Possible reasons: K=10 too small for Cv range at this layer,")
        print(f"      or head Cv distribution is not bimodal enough. Try K=100.")
    print(f"{'='*70}\n")

    # Save CSV — add summary row at end
    rows.append({
        "prompt_id": "PEARSON_R",
        "head":      -1,
        "cv":        round(float(r),     6),
        "kl":        round(float(p_val) if not np.isnan(p_val) else -1, 8),
    })

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_cv_kl_correlation_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "head", "cv", "kl"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")
    print(f"  (PEARSON_R row: cv=r={r:.4f}, kl=p_value)")

    return 0 if r_pass else 1


if __name__ == "__main__":
    sys.exit(main())
