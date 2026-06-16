"""
tasb_m7_multilayer.py — M7 Sub-sweep 5: Multi-layer composition
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Tests open-loop multi-layer composition: independently vanilla-captured
bridge substitutions injected together in one forward pass. Samples at each
layer are drawn from the VANILLA capture (pass 1), not from the hidden state
perturbed by upstream injections.

WHAT THIS MEASURES
------------------
Does injecting at multiple layers simultaneously preserve confident-bucket
integrity? Does KL grow roughly linearly with layer count (bounded) or
super-linearly (destructive compounding)?

STAGE 1 PROTOCOL (this script)
-------------------------------
Uniform alpha=0.3, K=10, seed=42, 4-prompt battery x 40 tokens.
Five configs tested:

  C1: [L18]                        single-layer baseline (regression check)
  C2: [L18, L24]                   2-layer, well separated
  C3: [L18, L21, L24]              3-layer, evenly spaced late stack
  C4: [L15, L18, L21, L24, L27]   5-layer, late stack spread
  C5: [L18, L19, L20]             3-layer, adjacent (composition shape check)

HEADLINE METRIC
---------------
Confident-bucket flips at each config (prob_gap >= 0.5 AND top1_agree=0).
Zero confident flips across all configs = composition is bounded.
Any confident flips = composition compounds destructively at that config.

BUCKET DEFINITIONS (cite alongside any zero-flip claim)
--------------------------------------------------------
  confident: prob_gap >= 0.5
  moderate:  0.1 <= prob_gap < 0.5
  ambiguous: prob_gap < 0.1

CSV SCHEMA
----------
Matches M7 sub-sweep 1-4 schema plus two new columns:
  config_id:   string  e.g. "C1", "C2", etc.
  n_layers:    int     number of injection layers in this config
  layers_str:  string  e.g. "18" or "18,24" — the actual injected layers
  alpha:       float   uniform alpha used (0.3)
  k_value:     int     K used (10)
  seed:        int     base seed (42)
  prompt_id:   string  prompt identifier
  domain:      string  prompt domain
  step:        int     teacher-forced position (1-indexed)
  vanilla_top1:  int   vanilla top-1 token id
  bridge_top1:   int   bridge top-1 token id
  top1_agree:    int   1 if agree, 0 if flip
  top5_agree:    int   1 if vanilla top-1 in bridge top-5
  kl:          float   KL(vanilla || bridge) at this position
  js:          float   JS divergence at this position
  prob_gap:    float   vanilla top-1 probability minus vanilla top-2
  alpha0_max_abs_diff: float  per-step alpha=0 identity check (must be 0.0)

OPEN-LOOP NOTE
--------------
Results from this script must be described as "open-loop multi-layer
composition." p_thermo at each layer is derived from the vanilla forward
pass, not from the residual stream after upstream injections. Closed-loop
composition is a future milestone (M7-6 / M8-adjacent).

OUTPUT
------
  tasb_m7_multilayer_<timestamp>.csv   — per-step rows for all configs
  Console summary printed at end.
==============================================================================
"""

import csv
import os
import sys
import zlib
from datetime import datetime

import torch
import torch.nn.functional as F

# Path setup — canonical TASB_Refactor root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasb_pipeline_v2 import bridge_forward, seed_for_layer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
ALPHA      = 0.3
K          = 10
BASE_SEED  = 42
N_STEPS    = 40   # teacher-forced positions per prompt

# Stage 1 layer configs — (config_id, layers_list, description)
CONFIGS = [
    ("C1", [18],                  "1-layer baseline"),
    ("C2", [18, 24],              "2-layer well-separated"),
    ("C3", [18, 21, 24],          "3-layer evenly spaced late stack"),
    ("C4", [15, 18, 21, 24, 27],  "5-layer late stack spread"),
    ("C5", [18, 19, 20],          "3-layer adjacent"),
]

# Prompt battery — matches M7 sub-sweep domain coverage
PROMPTS = [
    ("M75_HC", "FACTUAL",
     "The capital of France is Paris. The capital of Germany is Berlin. "
     "The capital of Japan is Tokyo. The capital of Australia is"),
    ("M75_TC", "CODE",
     "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
     "return fibonacci(n-1) + fibonacci(n-"),
    ("M75_RS", "REASONING",
     "All mammals are warm-blooded. All whales are mammals. "
     "Therefore, all whales are"),
    ("M75_CR", "CREATIVE",
     "In a world where gravity worked in reverse, the first thing "
     "people noticed was that"),
]

# Bucket boundaries (cite alongside any zero-flip claim)
CONFIDENT_THRESHOLD = 0.5
MODERATE_THRESHOLD  = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bucket(prob_gap: float) -> str:
    if prob_gap >= CONFIDENT_THRESHOLD:
        return "CONFIDENT"
    elif prob_gap >= MODERATE_THRESHOLD:
        return "MODERATE"
    else:
        return "AMBIGUOUS"


def compute_row_metrics(vanilla_logits, bridge_logits, step_idx):
    """Compute per-step metrics from logits at one position.

    Args:
        vanilla_logits: (vocab_size,) float32
        bridge_logits:  (vocab_size,) float32
        step_idx:       0-indexed step position

    Returns:
        dict of metric values for one CSV row
    """
    v_log = F.log_softmax(vanilla_logits, dim=-1)
    b_log = F.log_softmax(bridge_logits,  dim=-1)
    v_prob = v_log.exp()
    b_prob = b_log.exp()

    vanilla_top1 = vanilla_logits.argmax().item()
    bridge_top1  = bridge_logits.argmax().item()
    top1_agree   = int(vanilla_top1 == bridge_top1)

    # top5_agree: vanilla top-1 appears in bridge top-5
    bridge_top5 = bridge_logits.topk(5).indices.tolist()
    top5_agree  = int(vanilla_top1 in bridge_top5)

    # KL(vanilla || bridge) — Bug #3/#10: log_softmax, no clamp
    kl = F.kl_div(b_log, v_prob, reduction='sum').item()
    kl = max(kl, 0.0)  # numerical floor only, not suppressive clamp

    # JS divergence
    m_prob = 0.5 * (v_prob + b_prob)
    m_log  = m_prob.clamp(min=1e-40).log()
    js = 0.5 * (F.kl_div(m_log, v_prob, reduction='sum').item() +
                F.kl_div(m_log, b_prob, reduction='sum').item())
    js = max(js, 0.0)

    # prob_gap: vanilla top-1 prob minus vanilla top-2 prob
    top2_probs = v_prob.topk(2).values
    if top2_probs.shape[0] >= 2:
        prob_gap = (top2_probs[0] - top2_probs[1]).item()
    else:
        prob_gap = top2_probs[0].item()

    return {
        "step":         step_idx + 1,
        "vanilla_top1": vanilla_top1,
        "bridge_top1":  bridge_top1,
        "top1_agree":   top1_agree,
        "top5_agree":   top5_agree,
        "kl":           kl,
        "js":           js,
        "prob_gap":     prob_gap,
    }


def run_alpha0_check(model, tok, input_ids):
    """Run alpha=0 identity check on this input. Returns max_abs_diff."""
    result = bridge_forward(
        model, tok,
        input_ids=input_ids,
        layer_idx=18,
        alpha=0.0,
        backend='exact',
        K=K,
        seed=BASE_SEED,
        return_intermediates=True,
    )
    diff = (result.logits.float() - result.vanilla_logits.float()).abs()
    return diff.max().item()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(model, tok):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"tasb_m7_multilayer_{timestamp}.csv"

    fieldnames = [
        "config_id", "n_layers", "layers_str",
        "alpha", "k_value", "seed",
        "prompt_id", "domain",
        "step", "vanilla_top1", "bridge_top1",
        "top1_agree", "top5_agree",
        "kl", "js", "prob_gap",
        "alpha0_max_abs_diff",
    ]

    # Summary accumulators: {config_id: {bucket: count}}
    summary = {
        cfg_id: {
            "total": 0,
            "agree": 0,
            "confident_flips": 0,
            "moderate_flips":  0,
            "ambiguous_flips": 0,
            "kl_sum": 0.0,
        }
        for cfg_id, _, _ in CONFIGS
    }

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cfg_id, layers, cfg_desc in CONFIGS:
            n_layers   = len(layers)
            layers_str = ",".join(str(l) for l in layers)
            print(f"\n{'='*70}")
            print(f"Config {cfg_id}: {cfg_desc}")
            print(f"  layers={layers}, alpha={ALPHA}, K={K}, seed={BASE_SEED}")
            print(f"{'='*70}")

            for prompt_id, domain, prompt_text in PROMPTS:
                print(f"\n  Prompt {prompt_id} ({domain}): "
                      f"{prompt_text[:60]}...")

                # Tokenize once; use first N_STEPS+1 tokens for teacher forcing
                full_ids = tok(
                    prompt_text, return_tensors='pt'
                ).to(model.device)['input_ids']

                # Ensure we have enough tokens
                seq_len = full_ids.shape[1]
                if seq_len < 2:
                    print(f"    SKIP: sequence too short ({seq_len} tokens)")
                    continue
                n_steps = min(N_STEPS, seq_len - 1)

                # alpha=0 identity check (scalar, L18 only — regression guard)
                # Use the full available context for the check
                check_ids = full_ids[:, :min(seq_len, n_steps + 1)]
                alpha0_diff = run_alpha0_check(model, tok, check_ids)
                if alpha0_diff > 0.0:
                    print(f"    WARNING: alpha=0 identity check FAILED "
                          f"max_abs_diff={alpha0_diff:.2e}")

                # Teacher-forced loop: at step i, feed tokens [0..i],
                # measure prediction at position i (predict token i+1)
                prompt_agree = 0
                prompt_confident_flips = 0

                for step_i in range(n_steps):
                    # Context: tokens 0..step_i (inclusive)
                    ctx_ids = full_ids[:, :step_i + 1]

                    # Vanilla forward (for reference logits)
                    with torch.no_grad():
                        vanilla_out = model(
                            input_ids=ctx_ids, use_cache=False)
                    vanilla_logits_full = vanilla_out.logits[0, -1].float()

                    # Bridge forward at this config
                    bridge_logits_full = bridge_forward(
                        model, tok,
                        input_ids=ctx_ids,
                        layer_idx=layers,
                        alpha=ALPHA,
                        backend='exact',
                        K=K,
                        seed=BASE_SEED,
                        return_intermediates=False,
                    )
                    bridge_logits_pos = bridge_logits_full[0, -1].float()

                    metrics = compute_row_metrics(
                        vanilla_logits_full, bridge_logits_pos, step_i)

                    row = {
                        "config_id":   cfg_id,
                        "n_layers":    n_layers,
                        "layers_str":  layers_str,
                        "alpha":       ALPHA,
                        "k_value":     K,
                        "seed":        BASE_SEED,
                        "prompt_id":   prompt_id,
                        "domain":      domain,
                        "alpha0_max_abs_diff": alpha0_diff,
                        **metrics,
                    }
                    writer.writerow(row)

                    # Accumulate summary
                    s = summary[cfg_id]
                    s["total"]  += 1
                    s["agree"]  += metrics["top1_agree"]
                    s["kl_sum"] += metrics["kl"]

                    if metrics["top1_agree"] == 0:
                        b = bucket(metrics["prob_gap"])
                        if b == "CONFIDENT":
                            s["confident_flips"] += 1
                            prompt_confident_flips += 1
                        elif b == "MODERATE":
                            s["moderate_flips"] += 1
                        else:
                            s["ambiguous_flips"] += 1
                    else:
                        prompt_agree += 1

                print(f"    steps={n_steps}, "
                      f"top1_agree={prompt_agree}/{n_steps} "
                      f"({100*prompt_agree/n_steps:.1f}%), "
                      f"confident_flips={prompt_confident_flips}")

    # Console summary
    print(f"\n\n{'='*70}")
    print(f"M7-5 STAGE 1 SUMMARY")
    print(f"alpha={ALPHA}, K={K}, seed={BASE_SEED}, "
          f"prompts={len(PROMPTS)}, steps_per_prompt={N_STEPS}")
    print(f"Confident bucket: prob_gap >= {CONFIDENT_THRESHOLD}")
    print(f"{'='*70}")
    print(f"{'Config':<6} {'Layers':<22} {'N':<4} "
          f"{'Top1%':>7} {'MeanKL':>10} "
          f"{'ConfFlip':>9} {'ModFlip':>8} {'AmbFlip':>8}")
    print(f"{'-'*70}")

    for cfg_id, layers, cfg_desc in CONFIGS:
        s = summary[cfg_id]
        total = s["total"]
        if total == 0:
            continue
        top1_pct = 100.0 * s["agree"] / total
        mean_kl  = s["kl_sum"] / total
        layers_str = str(layers)
        print(f"{cfg_id:<6} {layers_str:<22} {total:<4} "
              f"{top1_pct:>7.2f}% {mean_kl:>10.5f} "
              f"{s['confident_flips']:>9} "
              f"{s['moderate_flips']:>8} "
              f"{s['ambiguous_flips']:>8}")

    print(f"{'='*70}")
    total_conf_flips = sum(s["confident_flips"] for s in summary.values())
    if total_conf_flips == 0:
        print("RESULT: ZERO confident-bucket flips across all configs.")
        print("        Composition is bounded. KL growth check above.")
    else:
        print(f"RESULT: {total_conf_flips} confident-bucket flip(s) detected.")
        print("        Check which configs produced flips.")
    print(f"\nCSV written: {out_csv}")
    print(f"{'='*70}")

    return out_csv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    print("M7 Sub-sweep 5: Multi-layer composition")
    print(f"Configs: {[c[0] for c in CONFIGS]}")
    print(f"Prompts: {len(PROMPTS)}, Steps: {N_STEPS}, "
          f"Alpha: {ALPHA}, K: {K}, Seed: {BASE_SEED}")
    print()
    print("Loading LLaMA 3.2-3B (4-bit)...", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    print(f"Model ready on {next(model.parameters()).device}")
    print()

    out_csv = run_sweep(model, tok)
