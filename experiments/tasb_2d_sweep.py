"""
tasb_2d_sweep.py — 2D alpha × layer config sweep
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Full characterization sweep across the alpha × layer config space.
Every slider position in the interactive demo gets real measured numbers
behind it. No interpolation, no extrapolation — actual bridge_forward
runs at every (alpha, config) pair.

WHY THIS IS NEEDED
------------------
All prior sweeps fixed one axis and varied the other:
  - M7-4 alpha sweep:    alpha 0.0-1.0 at single layer L18 only
  - M7-5/M7-6 layer sweep: fixed alpha=0.3, varied layer count

The demo has TWO sliders (alpha + layer config) and layer toggles.
The user can move both simultaneously. We need measured data at every
combination so the demo's metrics panel shows honest numbers everywhere,
not interpolated guesses.

SWEEP DESIGN
------------
Alpha axis (13 values):
  0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00

Layer configs (8 configs — full demo layer toggle space):
  L0:  [18]                              1L  single layer (baseline)
  L1:  [18, 24]                          2L  well-separated
  L2:  [15, 18, 21, 24, 27]             5L  spread (sweet spot)
  L3:  [10, 12, 14, 16, 18, 20, 22, 24, 26, 27]  10L  heavy
  L4:  [18, 19, 20]                      3L  adjacent
  L5:  [12, 15, 18, 21, 24, 27]         6L  extended spread
  L6:  [18, 21, 24]                      3L  evenly spaced
  L7:  [15, 18, 21, 24]                  4L  late stack

13 alpha × 8 configs = 104 (alpha, config) pairs
4 prompts × 40 steps each = 160 rows per pair
Total rows: ~16,640

At 2 forward passes per row (vanilla + bridge) and ~0.05s per pass:
Estimated runtime: ~2.5-3 hours.

PROMPT BATTERY (same 4 as M7 sweeps for direct comparison)
-----------------------------------------------------------
  M75_HC: FACTUAL  (capitals)
  M75_TC: CODE     (fibonacci)
  M75_RS: REASONING (mammals/whales)
  M75_CR: CREATIVE  (gravity)

METRICS PER ROW (matches M7 schema exactly)
--------------------------------------------
  config_id, n_layers, layers_str, alpha, k_value, seed,
  prompt_id, domain, step,
  vanilla_top1, bridge_top1, top1_agree, top5_agree,
  kl, js, prob_gap, alpha0_max_abs_diff

SUMMARY OUTPUT (the demo data layer)
--------------------------------------
Per (alpha, config) pair:
  top1_pct, mean_kl, mean_js, mean_prob_gap,
  confident_flips, moderate_flips, ambiguous_flips,
  total_steps

This summary CSV is what the demo's backend serves. One row per
(alpha, config) pair = 104 rows. Small enough to serve as static JSON.

OUTPUT
------
  results/tasb_2d_sweep_rows_<timestamp>.csv      full per-step data
  results/tasb_2d_sweep_summary_<timestamp>.csv   104-row demo data layer
  results/tasb_2d_sweep_console_<timestamp>.txt   via tee
==============================================================================
"""

import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasb_pipeline_v2 import bridge_forward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
K         = 10
BASE_SEED = 42
N_STEPS   = 40

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIDENT_THRESHOLD = 0.5
MODERATE_THRESHOLD  = 0.1

# Alpha axis — 13 values covering full [0, 1] range with fine resolution
# near production alpha (0.3) and coarser at extremes
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
          0.40, 0.50, 0.60, 0.70, 0.85, 1.00]

# Layer configs — 8 configs covering the full demo toggle space
LAYER_CONFIGS = [
    ("L0", [18],                              "1L single"),
    ("L1", [18, 24],                          "2L separated"),
    ("L2", [15, 18, 21, 24, 27],              "5L spread"),
    ("L3", [10, 12, 14, 16, 18, 20, 22, 24, 26, 27], "10L heavy"),
    ("L4", [18, 19, 20],                      "3L adjacent"),
    ("L5", [12, 15, 18, 21, 24, 27],          "6L extended"),
    ("L6", [18, 21, 24],                      "3L evenly spaced"),
    ("L7", [15, 18, 21, 24],                  "4L late stack"),
]

# Prompt battery — matches M7 sweeps exactly for cross-sweep comparison
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bucket(prob_gap):
    if prob_gap >= CONFIDENT_THRESHOLD:
        return "CONFIDENT"
    elif prob_gap >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "AMBIGUOUS"


def compute_metrics(v_logits, b_logits, step_idx):
    v_log  = F.log_softmax(v_logits, dim=-1)
    b_log  = F.log_softmax(b_logits, dim=-1)
    v_prob = v_log.exp()
    b_prob = b_log.exp()

    vanilla_top1 = v_logits.argmax().item()
    bridge_top1  = b_logits.argmax().item()
    top1_agree   = int(vanilla_top1 == bridge_top1)
    bridge_top5  = b_logits.topk(5).indices.tolist()
    top5_agree   = int(vanilla_top1 in bridge_top5)

    kl = max(F.kl_div(b_log, v_prob, reduction='sum').item(), 0.0)

    m_prob = 0.5 * (v_prob + b_prob)
    m_log  = m_prob.clamp(min=1e-40).log()
    js = max(0.5 * (
        F.kl_div(m_log, v_prob, reduction='sum').item() +
        F.kl_div(m_log, b_prob, reduction='sum').item()
    ), 0.0)

    top2     = v_prob.topk(2).values
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
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total_pairs):
        self.total   = total_pairs
        self.done    = 0
        self.t_start = time.time()

    def tick(self, cfg_id, alpha):
        self.done += 1
        elapsed  = time.time() - self.t_start
        per_pair = elapsed / self.done
        remaining = (self.total - self.done) * per_pair
        pct = 100 * self.done / self.total
        print(f"  [{self.done}/{self.total} {pct:.0f}%] "
              f"{cfg_id} α={alpha:.2f} done — "
              f"elapsed {elapsed/60:.1f}m  "
              f"est remaining {remaining/60:.1f}m")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_2d_sweep(model, tok):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_csv    = os.path.join(RESULTS_DIR,
                               f"tasb_2d_sweep_rows_{timestamp}.csv")
    summary_csv = os.path.join(RESULTS_DIR,
                               f"tasb_2d_sweep_summary_{timestamp}.csv")

    row_fields = [
        "config_id", "n_layers", "layers_str",
        "alpha", "k_value", "seed",
        "prompt_id", "domain",
        "step", "vanilla_top1", "bridge_top1",
        "top1_agree", "top5_agree",
        "kl", "js", "prob_gap",
        "alpha0_max_abs_diff",
    ]

    summary_fields = [
        "config_id", "n_layers", "layers_str", "alpha",
        "total_steps", "top1_agree_count", "top1_pct",
        "mean_kl", "mean_js", "mean_prob_gap",
        "confident_flips", "moderate_flips", "ambiguous_flips",
        "kl_at_confident",   # mean KL at positions where vanilla was confident
        "kl_at_ambiguous",   # mean KL at positions where vanilla was ambiguous
    ]

    total_pairs = len(ALPHAS) * len(LAYER_CONFIGS)
    tracker     = ProgressTracker(total_pairs)

    # Pre-compute alpha=0 checks once per prompt (config-independent)
    print("  Pre-computing alpha=0 identity checks...")
    alpha0_diffs = {}
    device = next(model.parameters()).device

    for prompt_id, domain, prompt_text in PROMPTS:
        full_ids = tok(
            prompt_text, return_tensors='pt'
        ).to(device)['input_ids']
        seq_len = full_ids.shape[1]
        n_steps = min(N_STEPS, seq_len - 1)
        check_ids = full_ids[:, :min(seq_len, n_steps + 1)]

        # One alpha=0 check per prompt at L18 scalar
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
        diff = (b_ref.float() - v_ref.float()).abs().max().item()
        alpha0_diffs[prompt_id] = diff
        status = "OK" if diff == 0.0 else f"WARN {diff:.2e}"
        print(f"    {prompt_id}: alpha0_diff={diff:.2e} [{status}]")

    print()

    # Cache vanilla logits per prompt×step (same for all configs/alphas)
    # This halves the number of forward passes — vanilla runs once per
    # (prompt, step), bridge runs once per (alpha, config, prompt, step)
    print("  Pre-caching vanilla logits...")
    vanilla_cache = {}  # (prompt_id, step_i) -> (ctx_ids, v_logits)

    for prompt_id, domain, prompt_text in PROMPTS:
        full_ids = tok(
            prompt_text, return_tensors='pt'
        ).to(device)['input_ids']
        seq_len = full_ids.shape[1]
        n_steps = min(N_STEPS, seq_len - 1)

        for step_i in range(n_steps):
            ctx_ids = full_ids[:, :step_i + 1]
            with torch.no_grad():
                v_out = model(input_ids=ctx_ids, use_cache=False)
            v_logits = v_out.logits[0, -1].float().cpu()
            vanilla_cache[(prompt_id, step_i)] = (
                ctx_ids.cpu(), v_logits, n_steps)

    print(f"  Cached {len(vanilla_cache)} (prompt, step) pairs.")
    print()

    # Main sweep
    summaries = []

    with open(rows_csv, "w", newline="") as fr:
        row_writer = csv.DictWriter(fr, fieldnames=row_fields)
        row_writer.writeheader()

        for alpha in ALPHAS:
            for cfg_id, layers, cfg_desc in LAYER_CONFIGS:
                layers_str = ",".join(str(l) for l in layers)
                n_layers   = len(layers)

                # Per-(alpha, config) accumulators
                total      = 0
                agree      = 0
                kl_sum     = 0.0
                js_sum     = 0.0
                pg_sum     = 0.0
                conf_flips = mod_flips = amb_flips = 0
                kl_conf_sum = kl_conf_n = 0
                kl_amb_sum  = kl_amb_n  = 0

                print(f"\n  {cfg_id} ({n_layers}L) α={alpha:.2f} — {cfg_desc}")

                for prompt_id, domain, prompt_text in PROMPTS:
                    full_ids = tok(
                        prompt_text, return_tensors='pt'
                    ).to(device)['input_ids']
                    seq_len = full_ids.shape[1]
                    n_steps = min(N_STEPS, seq_len - 1)
                    a0_diff = alpha0_diffs.get(prompt_id, 0.0)

                    for step_i in range(n_steps):
                        ctx_ids_cpu, v_logits_cpu, _ = vanilla_cache[
                            (prompt_id, step_i)]
                        ctx_ids  = ctx_ids_cpu.to(device)
                        v_logits = v_logits_cpu.to(device)

                        # Bridge forward (alpha=0 is handled by fast path
                        # in injector — returns vanilla bit-exact)
                        b_logits_full = bridge_forward(
                            model, tok,
                            input_ids=ctx_ids,
                            layer_idx=layers,
                            alpha=alpha,
                            backend='exact',
                            K=K,
                            seed=BASE_SEED,
                            return_intermediates=False,
                        )
                        b_logits = b_logits_full[0, -1].float()

                        m = compute_metrics(v_logits, b_logits, step_i)

                        row_writer.writerow({
                            "config_id":   cfg_id,
                            "n_layers":    n_layers,
                            "layers_str":  layers_str,
                            "alpha":       alpha,
                            "k_value":     K,
                            "seed":        BASE_SEED,
                            "prompt_id":   prompt_id,
                            "domain":      domain,
                            "alpha0_max_abs_diff": a0_diff,
                            **m,
                        })

                        total  += 1
                        agree  += m["top1_agree"]
                        kl_sum += m["kl"]
                        js_sum += m["js"]
                        pg_sum += m["prob_gap"]

                        pg = m["prob_gap"]
                        b  = bucket(pg)

                        # KL by vanilla confidence level
                        if b == "CONFIDENT":
                            kl_conf_sum += m["kl"]
                            kl_conf_n   += 1
                        elif b == "AMBIGUOUS":
                            kl_amb_sum  += m["kl"]
                            kl_amb_n    += 1

                        if m["top1_agree"] == 0:
                            if b == "CONFIDENT":
                                conf_flips += 1
                            elif b == "MODERATE":
                                mod_flips += 1
                            else:
                                amb_flips += 1

                fr.flush()

                top1_pct   = 100.0 * agree / total if total > 0 else 0.0
                mean_kl    = kl_sum / total if total > 0 else 0.0
                mean_js    = js_sum / total if total > 0 else 0.0
                mean_pg    = pg_sum / total if total > 0 else 0.0
                kl_at_conf = kl_conf_sum / kl_conf_n if kl_conf_n > 0 else 0.0
                kl_at_amb  = kl_amb_sum  / kl_amb_n  if kl_amb_n  > 0 else 0.0

                print(f"    top1={top1_pct:.2f}%  KL={mean_kl:.5f}  "
                      f"CF={conf_flips} MF={mod_flips} AF={amb_flips}")

                summaries.append({
                    "config_id":      cfg_id,
                    "n_layers":       n_layers,
                    "layers_str":     layers_str,
                    "alpha":          alpha,
                    "total_steps":    total,
                    "top1_agree_count": agree,
                    "top1_pct":       round(top1_pct, 4),
                    "mean_kl":        round(mean_kl, 6),
                    "mean_js":        round(mean_js, 6),
                    "mean_prob_gap":  round(mean_pg, 6),
                    "confident_flips": conf_flips,
                    "moderate_flips":  mod_flips,
                    "ambiguous_flips": amb_flips,
                    "kl_at_confident": round(kl_at_conf, 6),
                    "kl_at_ambiguous": round(kl_at_amb, 6),
                })

                tracker.tick(cfg_id, alpha)
                gc.collect()
                torch.cuda.empty_cache()

    # Write summary CSV
    with open(summary_csv, "w", newline="") as fs:
        writer = csv.DictWriter(fs, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    # Console summary — alpha × config heatmap of top1_pct
    print(f"\n\n{'='*76}")
    print(f"2D SWEEP COMPLETE — TOP-1 AGREEMENT % HEATMAP")
    print(f"{'='*76}")
    cfg_ids = [c[0] for c in LAYER_CONFIGS]
    header  = f"{'Alpha':>6}  " + "  ".join(f"{c:>5}" for c in cfg_ids)
    print(header)
    print("-" * len(header))

    for alpha in ALPHAS:
        row_vals = []
        for cfg_id, _, _ in LAYER_CONFIGS:
            s = next((s for s in summaries
                      if s["alpha"] == alpha and s["config_id"] == cfg_id), None)
            row_vals.append(f"{s['top1_pct']:>5.1f}" if s else "  N/A")
        print(f"  {alpha:>4.2f}  " + "  ".join(row_vals))

    print(f"\n{'='*76}")
    print(f"TOP-1 AGREEMENT % — KL DIVERGENCE HEATMAP")
    print(f"{'='*76}")
    print(header)
    print("-" * len(header))

    for alpha in ALPHAS:
        row_vals = []
        for cfg_id, _, _ in LAYER_CONFIGS:
            s = next((s for s in summaries
                      if s["alpha"] == alpha and s["config_id"] == cfg_id), None)
            row_vals.append(f"{s['mean_kl']:>5.4f}" if s else "   N/A")
        print(f"  {alpha:>4.2f}  " + "  ".join(row_vals))

    print(f"\n{'='*76}")
    print(f"CONFIDENT FLIPS HEATMAP (should be all zeros)")
    print(f"{'='*76}")
    print(header)
    print("-" * len(header))

    total_conf_flips = 0
    for alpha in ALPHAS:
        row_vals = []
        for cfg_id, _, _ in LAYER_CONFIGS:
            s = next((s for s in summaries
                      if s["alpha"] == alpha and s["config_id"] == cfg_id), None)
            cf = s["confident_flips"] if s else -1
            total_conf_flips += max(cf, 0)
            row_vals.append(f"{cf:>5}" if s else "   N/A")
        print(f"  {alpha:>4.2f}  " + "  ".join(row_vals))

    print(f"\n  Total confident flips: {total_conf_flips}")
    if total_conf_flips == 0:
        print(f"  RESULT: ZERO confident flips across full 2D space.")
        print(f"  Demo slider is honest at every position.")
    else:
        print(f"  RESULT: {total_conf_flips} confident flip(s) detected.")
        print(f"  Check summary CSV for affected (alpha, config) pairs.")

    print(f"\n  Rows CSV:    {rows_csv}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"{'='*76}")

    return rows_csv, summary_csv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    n_pairs = len(ALPHAS) * len(LAYER_CONFIGS)
    n_rows  = n_pairs * len(PROMPTS) * N_STEPS
    # Each pair: N_STEPS * len(PROMPTS) bridge passes
    # Plus one vanilla pass per (prompt, step) cached at start
    vanilla_passes = len(PROMPTS) * N_STEPS
    bridge_passes  = n_pairs * len(PROMPTS) * N_STEPS
    est_min = (vanilla_passes + bridge_passes) * 0.05 / 60

    print("TASB 2D Sweep: alpha × layer config")
    print(f"Alpha values ({len(ALPHAS)}): {ALPHAS}")
    print(f"Layer configs ({len(LAYER_CONFIGS)}): "
          f"{[c[0] for c in LAYER_CONFIGS]}")
    print(f"Prompts: {len(PROMPTS)}  Steps: {N_STEPS}")
    print(f"Total (alpha, config) pairs: {n_pairs}")
    print(f"Total rows: ~{n_rows:,}")
    print(f"Bridge passes: {bridge_passes:,}  "
          f"(vanilla cached: {vanilla_passes})")
    print(f"Estimated runtime: ~{est_min:.0f} min")
    print()
    print("Loading model...", flush=True)

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

    run_2d_sweep(model, tok)
