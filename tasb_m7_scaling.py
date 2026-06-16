"""
tasb_m7_scaling.py — M7 Sub-sweep 6: Composition scaling curve
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Extends M7-5 (Stage 1, C1-C5) by pushing layer count higher to characterize
the full composition scaling curve. Answers three questions:

  1. Where does the boundary actually sit? (if it exists at all)
  2. Is KL growth linear, logarithmic, or saturating with layer count?
  3. Do confident-bucket flips appear at high layer counts, and if so,
     at what threshold?

This data feeds directly into the interactive demo: the scaling curve is
the "physics" behind the alpha/layer sliders. It also feeds the Extropic
pitch: "progressive substrate adoption" — how many layers can the Z1 chip
absorb before the bridge needs recalibration?

STAGE 2 PROTOCOL
----------------
Uniform alpha=0.3, K=10, seed=42, 4-prompt battery x 40 tokens.
Late-stack focus (layers 12-27), progressively denser:

  S1:  [18]                               (1L  — M7-5 C1 regression check)
  S2:  [18, 24]                           (2L  — M7-5 C2 regression check)
  S3:  [15, 18, 21, 24, 27]              (5L  — M7-5 C4 regression check)
  S4:  [12, 15, 18, 21, 24, 27]          (6L  — add one more early)
  S5:  [12, 14, 16, 18, 20, 22, 24, 26]  (8L  — dense late stack)
  S6:  [10, 12, 14, 16, 18, 20, 22, 24, 26, 27]  (10L — heavy offload)
  S7:  [18, 19, 20, 21, 22, 23, 24]      (7L  — adjacent cluster)
  S8:  all even layers 12-26             (8L  — uniform spread)

S1/S2/S3 are regression checks against M7-5 C1/C2/C4. If these differ,
something changed in the stack and the new results are invalid.

HEADLINE METRICS
----------------
Per config:
  - top-1 agreement %
  - mean KL
  - confident/moderate/ambiguous flip counts
  - KL growth rate vs prior config (delta)

Cross-config:
  - KL vs layer_count scatter (is it linear? logarithmic? saturating?)
  - First config with any confident flip (if any)

OUTPUT
------
  results/tasb_m7_scaling_<timestamp>.csv     per-step rows
  results/tasb_m7_scaling_summary_<timestamp>.csv  per-config summary
  Console summary table printed at end.

OPEN-LOOP NOTE
--------------
All samples drawn from VANILLA capture (pass 1). Open-loop composition
only. Describe results as "open-loop multi-layer composition scaling."
==============================================================================
"""

import csv
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasb_pipeline_v2 import bridge_forward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
ALPHA     = 0.3
K         = 10
BASE_SEED = 42
N_STEPS   = 40

# Stage 2 configs — (config_id, layers, description)
CONFIGS = [
    ("S1", [18],
     "1L baseline (M7-5 C1 regression)"),
    ("S2", [18, 24],
     "2L well-separated (M7-5 C2 regression)"),
    ("S3", [15, 18, 21, 24, 27],
     "5L late-stack (M7-5 C4 regression)"),
    ("S4", [12, 15, 18, 21, 24, 27],
     "6L late-stack + one earlier"),
    ("S5", [12, 14, 16, 18, 20, 22, 24, 26],
     "8L dense late stack"),
    ("S6", [10, 12, 14, 16, 18, 20, 22, 24, 26, 27],
     "10L heavy offload"),
    ("S7", [18, 19, 20, 21, 22, 23, 24],
     "7L adjacent cluster"),
    ("S8", [12, 14, 16, 18, 20, 22, 24, 26],
     "8L uniform spread (same as S5 — shape check)"),
]

# Prompt battery — same 4 prompts as M7-5 for direct comparison
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

CONFIDENT_THRESHOLD = 0.5
MODERATE_THRESHOLD  = 0.1

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bucket(prob_gap: float) -> str:
    if prob_gap >= CONFIDENT_THRESHOLD:
        return "CONFIDENT"
    elif prob_gap >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "AMBIGUOUS"


def compute_metrics(vanilla_logits_pos, bridge_logits_pos, step_idx):
    v_log  = F.log_softmax(vanilla_logits_pos, dim=-1)
    b_log  = F.log_softmax(bridge_logits_pos,  dim=-1)
    v_prob = v_log.exp()
    b_prob = b_log.exp()

    vanilla_top1 = vanilla_logits_pos.argmax().item()
    bridge_top1  = bridge_logits_pos.argmax().item()
    top1_agree   = int(vanilla_top1 == bridge_top1)

    bridge_top5  = bridge_logits_pos.topk(5).indices.tolist()
    top5_agree   = int(vanilla_top1 in bridge_top5)

    # KL(vanilla || bridge) — log_softmax, no clamp (Bug #3/#10)
    kl = max(F.kl_div(b_log, v_prob, reduction='sum').item(), 0.0)

    # JS divergence
    m_prob = 0.5 * (v_prob + b_prob)
    m_log  = m_prob.clamp(min=1e-40).log()
    js = max(0.5 * (
        F.kl_div(m_log, v_prob, reduction='sum').item() +
        F.kl_div(m_log, b_prob, reduction='sum').item()
    ), 0.0)

    top2 = v_prob.topk(2).values
    prob_gap = (top2[0] - top2[1]).item() if top2.shape[0] >= 2 else top2[0].item()

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


# ---------------------------------------------------------------------------
# Regression check against M7-5 C1 result
# ---------------------------------------------------------------------------
M75_REGRESSION = {
    # config_id -> (expected_top1_pct, expected_mean_kl, tolerance)
    "S1": (100.0, 0.00138, 0.0005),
    "S2": (100.0, 0.00207, 0.0005),
    "S3": (98.82, 0.00515, 0.005),
}


def regression_check(cfg_id, top1_pct, mean_kl):
    if cfg_id not in M75_REGRESSION:
        return
    exp_top1, exp_kl, tol = M75_REGRESSION[cfg_id]
    top1_ok = abs(top1_pct - exp_top1) < 0.5
    kl_ok   = abs(mean_kl - exp_kl) < tol
    status  = "PASS" if (top1_ok and kl_ok) else "WARN"
    print(f"    [M7-5 regression {cfg_id}] {status}: "
          f"top1={top1_pct:.2f}% (exp {exp_top1:.2f}%), "
          f"KL={mean_kl:.5f} (exp {exp_kl:.5f})")
    if status == "WARN":
        print(f"    WARNING: {cfg_id} regression mismatch — "
              f"check stack for changes since M7-5")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(model, tok):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv     = os.path.join(RESULTS_DIR, f"tasb_m7_scaling_{timestamp}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"tasb_m7_scaling_summary_{timestamp}.csv")

    row_fieldnames = [
        "config_id", "n_layers", "layers_str",
        "alpha", "k_value", "seed",
        "prompt_id", "domain",
        "step", "vanilla_top1", "bridge_top1",
        "top1_agree", "top5_agree",
        "kl", "js", "prob_gap",
        "alpha0_max_abs_diff",
    ]

    summary_fieldnames = [
        "config_id", "n_layers", "layers_str",
        "total_steps", "top1_agree", "top1_pct",
        "mean_kl", "kl_delta_vs_prev",
        "confident_flips", "moderate_flips", "ambiguous_flips",
        "first_flip_step", "first_flip_prob_gap",
    ]

    # Per-config accumulators
    summaries = []
    prev_mean_kl = None

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_fieldnames)
        writer.writeheader()

        for cfg_id, layers, cfg_desc in CONFIGS:
            n_layers   = len(layers)
            layers_str = ",".join(str(l) for l in layers)

            print(f"\n{'='*72}")
            print(f"Config {cfg_id} ({n_layers}L): {cfg_desc}")
            print(f"  layers={layers}")
            print(f"  alpha={ALPHA}, K={K}, seed={BASE_SEED}")
            print(f"{'='*72}")

            total = 0
            agree = 0
            kl_sum = 0.0
            conf_flips = mod_flips = amb_flips = 0
            first_flip_step = None
            first_flip_pg   = None
            alpha0_diff     = 0.0

            for prompt_id, domain, prompt_text in PROMPTS:
                print(f"\n  [{prompt_id}] {prompt_text[:55]}...")

                full_ids = tok(
                    prompt_text, return_tensors='pt'
                ).to(model.device)['input_ids']
                seq_len = full_ids.shape[1]
                if seq_len < 2:
                    print(f"    SKIP: too short ({seq_len} tokens)")
                    continue
                n_steps = min(N_STEPS, seq_len - 1)

                # alpha=0 identity check (L18 scalar, once per prompt)
                check_ids = full_ids[:, :min(seq_len, n_steps + 1)]
                with torch.no_grad():
                    v_out = model(input_ids=check_ids, use_cache=False)
                v_ref = v_out.logits.detach().clone()
                b_ref = bridge_forward(
                    model, tok,
                    input_ids=check_ids,
                    layer_idx=18, alpha=0.0,
                    backend='exact', K=K, seed=BASE_SEED,
                    return_intermediates=False,
                )
                alpha0_diff = (b_ref.float() - v_ref.float()).abs().max().item()
                if alpha0_diff > 0.0:
                    print(f"    WARNING: alpha=0 identity FAILED "
                          f"max_abs_diff={alpha0_diff:.2e}")

                prompt_agree = 0
                prompt_conf  = 0

                for step_i in range(n_steps):
                    ctx_ids = full_ids[:, :step_i + 1]

                    # Vanilla reference
                    with torch.no_grad():
                        v_out = model(input_ids=ctx_ids, use_cache=False)
                    v_logits = v_out.logits[0, -1].float()

                    # Bridge forward
                    b_logits_full = bridge_forward(
                        model, tok,
                        input_ids=ctx_ids,
                        layer_idx=layers,
                        alpha=ALPHA,
                        backend='exact',
                        K=K,
                        seed=BASE_SEED,
                        return_intermediates=False,
                    )
                    b_logits = b_logits_full[0, -1].float()

                    m = compute_metrics(v_logits, b_logits, step_i)

                    writer.writerow({
                        "config_id":   cfg_id,
                        "n_layers":    n_layers,
                        "layers_str":  layers_str,
                        "alpha":       ALPHA,
                        "k_value":     K,
                        "seed":        BASE_SEED,
                        "prompt_id":   prompt_id,
                        "domain":      domain,
                        "alpha0_max_abs_diff": alpha0_diff,
                        **m,
                    })

                    total  += 1
                    agree  += m["top1_agree"]
                    kl_sum += m["kl"]

                    if m["top1_agree"] == 0:
                        b = bucket(m["prob_gap"])
                        if b == "CONFIDENT":
                            conf_flips += 1
                            prompt_conf += 1
                            if first_flip_step is None:
                                first_flip_step = m["step"]
                                first_flip_pg   = m["prob_gap"]
                        elif b == "MODERATE":
                            mod_flips += 1
                            if first_flip_step is None:
                                first_flip_step = m["step"]
                                first_flip_pg   = m["prob_gap"]
                        else:
                            amb_flips += 1
                    else:
                        prompt_agree += 1

                print(f"    steps={n_steps}, agree={prompt_agree}/{n_steps} "
                      f"({100*prompt_agree/n_steps:.1f}%), "
                      f"conf_flips={prompt_conf}")

            # Config summary
            top1_pct = 100.0 * agree / total if total > 0 else 0.0
            mean_kl  = kl_sum / total if total > 0 else 0.0
            kl_delta = (mean_kl - prev_mean_kl) if prev_mean_kl is not None else 0.0
            prev_mean_kl = mean_kl

            summaries.append({
                "config_id":          cfg_id,
                "n_layers":           n_layers,
                "layers_str":         layers_str,
                "total_steps":        total,
                "top1_agree":         agree,
                "top1_pct":           round(top1_pct, 4),
                "mean_kl":            round(mean_kl, 6),
                "kl_delta_vs_prev":   round(kl_delta, 6),
                "confident_flips":    conf_flips,
                "moderate_flips":     mod_flips,
                "ambiguous_flips":    amb_flips,
                "first_flip_step":    first_flip_step if first_flip_step else "",
                "first_flip_prob_gap": round(first_flip_pg, 4) if first_flip_pg is not None else "",
            })

            # Regression check for S1/S2/S3
            regression_check(cfg_id, top1_pct, mean_kl)

    # Write summary CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    # Console summary table
    print(f"\n\n{'='*76}")
    print(f"M7-6 SCALING SWEEP SUMMARY")
    print(f"alpha={ALPHA}, K={K}, seed={BASE_SEED}, "
          f"prompts={len(PROMPTS)}, max_steps={N_STEPS}")
    print(f"Confident bucket: prob_gap >= {CONFIDENT_THRESHOLD}")
    print(f"{'='*76}")
    print(f"{'Cfg':<5} {'N':>3} {'Layers':<34} "
          f"{'Top1%':>7} {'MeanKL':>9} {'dKL':>8} "
          f"{'CF':>4} {'MF':>4} {'AF':>4}")
    print(f"{'-'*76}")

    for s in summaries:
        layers_short = s["layers_str"]
        if len(layers_short) > 32:
            layers_short = layers_short[:29] + "..."
        print(f"{s['config_id']:<5} {s['n_layers']:>3} {layers_short:<34} "
              f"{s['top1_pct']:>7.2f}% {s['mean_kl']:>9.5f} "
              f"{s['kl_delta_vs_prev']:>+8.5f} "
              f"{s['confident_flips']:>4} "
              f"{s['moderate_flips']:>4} "
              f"{s['ambiguous_flips']:>4}")

    print(f"{'='*76}")
    print(f"CF=confident flips, MF=moderate flips, AF=ambiguous flips")
    print(f"dKL=KL delta vs previous config (growth rate)")

    total_conf = sum(s["confident_flips"] for s in summaries)
    if total_conf == 0:
        print(f"\nRESULT: ZERO confident-bucket flips across all {len(CONFIGS)} configs.")
        print(f"        Composition remains bounded through "
              f"{max(s['n_layers'] for s in summaries)} layers.")
    else:
        first_conf = next(
            s for s in summaries if s["confident_flips"] > 0)
        print(f"\nRESULT: First confident flip at config "
              f"{first_conf['config_id']} "
              f"({first_conf['n_layers']} layers).")
        print(f"        Boundary identified. See summary CSV for details.")

    # KL growth shape analysis
    print(f"\nKL GROWTH SHAPE:")
    kl_vals = [(s["n_layers"], s["mean_kl"]) for s in summaries]
    # Check if growth is slowing (logarithmic/saturating) or accelerating
    deltas = [s["kl_delta_vs_prev"] for s in summaries[1:]]
    if len(deltas) >= 2:
        accel = deltas[-1] - deltas[0]
        if accel < -0.0005:
            shape = "DECELERATING (sub-linear / saturating)"
        elif accel > 0.0005:
            shape = "ACCELERATING (super-linear / compounding)"
        else:
            shape = "ROUGHLY LINEAR"
        print(f"  KL growth appears {shape}")
        print(f"  First delta: {deltas[0]:+.5f}, Last delta: {deltas[-1]:+.5f}")

    print(f"\nFull CSV:     {out_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"{'='*76}")

    return out_csv, summary_csv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    print("M7 Sub-sweep 6: Composition scaling curve")
    print(f"Configs: {[c[0] for c in CONFIGS]} "
          f"(layer counts: {[len(c[1]) for c in CONFIGS]})")
    print(f"Prompts: {len(PROMPTS)}, Max steps: {N_STEPS}, "
          f"Alpha: {ALPHA}, K: {K}, Seed: {BASE_SEED}")
    print(f"Output dir: results/")
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

    run_sweep(model, tok)
