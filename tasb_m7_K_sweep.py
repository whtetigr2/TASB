"""
tasb_m7_K_sweep.py — M7 sub-sweep 2: characterization across K (sample size)
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

ANSWERS: "What's the smallest K at which the bridge is faithful enough?"

K = number of stochastic samples drawn per attention slot per step. On TSU
silicon, the cost of K=10 vs K=100 is nearly identical (parallel p-bit
sampling), but on GPU the cost scales roughly linearly. This sweep maps
the Pareto curve: where does mean KL plateau as a function of K?

If the curve flattens by K=10, we know our M5/M6/M7-layer headline numbers
(all at K=10) sit on a robust plateau. If it keeps improving up to K=50
or K=100, the production K should be higher and we lean harder on the
TSU's "free" sample budget.

METHOD: Teacher-forced, single layer (L18, the M5/M6 baseline), single α
(0.3, the production design point), all other settings carried from M5.
Sweep K ∈ {1, 3, 5, 10, 25, 50, 100}.

OUTPUT: per-K CSV + combined CSV with mean KL, top-1 agreement, JS, and
confident-bucket integrity at each K.

USAGE
-----
    python tasb_m7_K_sweep.py             # full sweep, ~1-2 hr
    python tasb_m7_K_sweep.py --quick     # 3 K values × 2 prompts, ~15 min
==============================================================================
"""

import argparse
import csv
import os
import sys
import time
import zlib
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasb_pipeline_v2 import bridge_forward


# ── Color helpers ─────────────────────────────────────────────────────────
def _c(code, t):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def gray(t):   return _c("90", t)


# ── Configuration ─────────────────────────────────────────────────────────
LAYER_IDX = 18           # fixed at M5/M6/M7-layer baseline
ALPHA     = 0.3          # production design point
BACKEND   = 'exact'

# K values to sweep. The sparse log-ish grid covers the regime where the
# curve is most likely to bend:
#   K=1: maximum-noise lower bound
#   K=3, 5: very low sample budgets — what cheapest silicon might deliver
#   K=10: M5/M6/M7-layer baseline
#   K=25, 50: realistic mid-range TSU configurations
#   K=100: approaches the exact distribution
K_FULL  = [1, 3, 5, 10, 25, 50, 100]
K_QUICK = [3, 10, 50]

# 4-prompt characterization battery (same as M7-layer for direct comparability)
PROMPTS_FULL = [
    {"id": "L7_HC", "domain": "FACTUAL",
     "text": "The capital of France is"},
    {"id": "L7_TC", "domain": "CODE",
     "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"id": "L7_RS", "domain": "REASONING",
     "text": "Explain why a compass points north in simple terms for a curious teenager."},
    {"id": "L7_CR", "domain": "CREATIVE",
     "text": "Write a short story that begins: The old lighthouse keeper"},
]
PROMPTS_QUICK = PROMPTS_FULL[:2]


# ── Metrics (log-space, no clamp — carried from M5/M6/M7-layer) ───────────

def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    return max(0.0, float((p * (log_p - log_q)).sum().item()))


def _js(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + 1e-30).log()
    kl_pm = float((p * (log_p - log_m)).sum().item())
    kl_qm = float((q * (log_q - log_m)).sum().item())
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def _prob_gap(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), dim=-1)
    top2 = torch.topk(p, 2).values
    return float((top2[0] - top2[1]).item())


# ── Single-K measurement (one prompt, fixed α, fixed layer, varying K) ────

def measure_K_on_prompt(model, tok, prompt: dict, tokens: int,
                        k_value: int, base_seed: int) -> list[dict]:
    """Teacher-forced measurement at one (prompt, K) cell. Single α=0.3."""
    records = []
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(tokens):
        sampler_seed = (base_seed + zlib.crc32(
            f"sampler|{prompt['id']}|K{k_value}|{step}".encode('utf-8'))
            ) & 0x7FFFFFFF

        # α=0 base (gives vanilla logits + identity regression test)
        base = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND, K=k_value,
            seed=sampler_seed, return_intermediates=True)
        vanilla_logits = base.vanilla_logits[0, -1, :]
        bridge_a0 = base.logits[0, -1, :]
        alpha0_max_abs_diff = float(
            (vanilla_logits.float() - bridge_a0.float()).abs().max().item())

        vanilla_token = int(vanilla_logits.argmax().item())
        van_top5 = torch.topk(vanilla_logits, 5).indices.tolist()
        prob_gap = _prob_gap(vanilla_logits)

        # α=0.3 (the only non-zero α — K is the variable here)
        result = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=LAYER_IDX, alpha=ALPHA, backend=BACKEND, K=k_value,
            seed=sampler_seed, return_intermediates=True)
        bridge_logits = result.logits[0, -1, :]
        bridge_top1 = int(bridge_logits.argmax().item())

        records.append({
            'k_value':       k_value,
            'layer_idx':     LAYER_IDX,
            'alpha':         ALPHA,
            'prompt_id':     prompt['id'],
            'domain':        prompt['domain'],
            'step':          step + 1,
            'vanilla_top1':  vanilla_token,
            'bridge_top1':   bridge_top1,
            'top1_agree':    int(bridge_top1 == vanilla_token),
            'top5_agree':    int(bridge_top1 in van_top5),
            'kl':            _kl(vanilla_logits, bridge_logits),
            'js':            _js(vanilla_logits, bridge_logits),
            'prob_gap':      prob_gap,
            'alpha0_max_abs_diff': alpha0_max_abs_diff,
        })

        # Teacher-force: advance by vanilla's choice
        ids = torch.cat(
            [ids, torch.tensor([[vanilla_token]], device=model.device)], dim=1)
        if vanilla_token == tok.eos_token_id:
            break

    return records


# ── Aggregate & report ────────────────────────────────────────────────────

def summarize_K_sweep(records: list[dict], k_values: list):
    """Print the K curve: how does faithfulness scale with sample size?"""
    print(f"\n{'═'*78}")
    print(bold(f"  M7 K SWEEP — top-1 agreement, mean KL, mean JS by K"))
    print(f"  (Layer=L{LAYER_IDX}, α={ALPHA}, fixed across the sweep)")
    print(f"{'═'*78}\n")

    print(f"  {'K':>5}  {'n':>5}  {'top1%':>8}  {'top5%':>8}  "
          f"{'mean_KL':>12}  {'mean_JS':>12}  {'α=0 max_diff':>14}")
    print(f"  {'─'*78}")

    for k in k_values:
        rows = [r for r in records if r['k_value'] == k]
        if not rows:
            continue
        n = len(rows)
        top1 = np.mean([r['top1_agree'] for r in rows]) * 100
        top5 = np.mean([r['top5_agree'] for r in rows]) * 100
        mkl  = np.mean([r['kl'] for r in rows])
        mjs  = np.mean([r['js'] for r in rows])
        max_a0 = max(r['alpha0_max_abs_diff'] for r in rows)
        color = green if top1 >= 95 else yellow if top1 >= 80 else red
        a0_str = (green(f"{max_a0:.2e}") if max_a0 < 1e-5
                  else red(f"{max_a0:.2e}"))
        print(f"  {k:>5}  {n:>5}  {color(f'{top1:>6.1f}%')}  "
              f"{top5:>6.1f}%  {mkl:>12.6f}  {mjs:>12.6f}  {a0_str:>14}")

    # Confident-bucket integrity per K
    print(f"\n{'═'*78}")
    print(bold(f"  M7 K SWEEP — Confident-bucket integrity by K"))
    print(f"  (zero confident-bucket flips is the structural M5/M6/M7-layer property)")
    print(f"{'═'*78}\n")
    print(f"  {'K':>5}  {'CONFIDENT %':>14}  {'MODERATE %':>14}  "
          f"{'AMBIGUOUS %':>14}  {'CONF flips':>12}")
    print(f"  {'─'*70}")
    for k in k_values:
        rows = [r for r in records if r['k_value'] == k]
        if not rows:
            continue
        cells = []
        flips = 0
        for lo, hi, label in [(0.5, 1.01, 'CONFIDENT'),
                              (0.1, 0.5, 'MODERATE'),
                              (0.0, 0.1, 'AMBIGUOUS')]:
            buck = [r for r in rows if lo <= r['prob_gap'] < hi]
            if buck:
                pct = np.mean([r['top1_agree'] for r in buck]) * 100
                cells.append(f"{pct:>5.1f}% (n={len(buck):>3})")
                if label == 'CONFIDENT':
                    flips = sum(1 for r in buck if r['top1_agree'] == 0)
            else:
                cells.append("—")
        flips_str = green(f"{flips:>3}") if flips == 0 else red(f"{flips:>3}")
        print(f"  {k:>5}  {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}  "
              f"{flips_str:>12}")

    # Pareto-style commentary: where does KL plateau?
    print(f"\n{'═'*78}")
    print(bold(f"  M7 K SWEEP — KL plateau analysis"))
    print(f"{'═'*78}\n")
    sorted_ks = sorted(k_values)
    kl_by_k = {}
    for k in sorted_ks:
        rows = [r for r in records if r['k_value'] == k]
        if rows:
            kl_by_k[k] = np.mean([r['kl'] for r in rows])
    print(f"  K progression of mean KL:")
    prev_kl = None
    for k in sorted_ks:
        kl = kl_by_k.get(k)
        if kl is None:
            continue
        if prev_kl is not None and prev_kl > 0:
            pct_change = (kl - prev_kl) / prev_kl * 100
            change_str = f"  ({pct_change:+.1f}% vs prev)"
        else:
            change_str = ""
        print(f"    K={k:>3}:  mean KL = {kl:.6f}{change_str}")
        prev_kl = kl
    print()
    print(f"  Interpretation:")
    print(f"    - If KL drops sharply as K grows then plateaus, the plateau K")
    print(f"      is the smallest sample budget that produces \"faithful enough\" output.")
    print(f"    - If KL keeps falling at K=100, the production K should be higher.")
    print(f"    - K=10 is the M5/M6/M7-layer baseline — check whether it sits")
    print(f"      on the plateau or in the decreasing region.")


def write_csv(records: list[dict], path: str):
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
    print(f"\n[SYS_INIT] Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),
        attn_implementation='eager', device_map='auto')
    mdl.eval()
    print(f"[SYS_INIT] Ready on {next(mdl.parameters()).device}\n", flush=True)
    return mdl, tok


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--tokens', type=int, default=40)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--quick',  action='store_true')
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        prompts = PROMPTS_QUICK
        k_values = K_QUICK
    else:
        prompts = PROMPTS_FULL
        k_values = K_FULL

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M7 K SWEEP  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:        {args.model}")
    print(f"  Layer:        L{LAYER_IDX}  (M5/M6/M7-layer baseline)")
    print(f"  α:            {ALPHA}  (production design point)")
    print(f"  Backend:      {BACKEND},  seed={args.seed}")
    print(f"  K sweep:      {k_values}")
    print(f"  Prompts:      {len(prompts)} × {args.tokens} tokens")
    print(f"  Cells:        {len(k_values)} K values × {len(prompts)} prompts")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    all_records = []
    t_start = time.perf_counter()

    for k_value in k_values:
        t_k = time.perf_counter()
        print(f"\n{bold(f'── K={k_value} ──')}")
        k_records = []
        for prompt in prompts:
            t_prompt = time.perf_counter()
            print(f"  {cyan(prompt['id'])} [{prompt['domain']}]: ",
                  end='', flush=True)
            recs = measure_K_on_prompt(
                model=model, tok=tok, prompt=prompt,
                tokens=args.tokens, k_value=k_value, base_seed=args.seed)
            elapsed = time.perf_counter() - t_prompt
            k_records.extend(recs)
            agree = np.mean([r['top1_agree'] for r in recs]) * 100 if recs else 0
            kl = np.mean([r['kl'] for r in recs]) if recs else 0
            print(f"top1={agree:>5.1f}%, KL={kl:.5f}  ({elapsed:.0f}s)")
        all_records.extend(k_records)
        k_elapsed = time.perf_counter() - t_k
        print(f"  K={k_value} done in {k_elapsed:.0f}s, {len(k_records)} rows")

        # Per-K CSV
        k_csv = os.path.join(
            args.outdir, f"tasb_m7_K{k_value:03d}_{ts}.csv")
        write_csv(k_records, k_csv)

    # Combined CSV
    combined_csv = os.path.join(
        args.outdir, f"tasb_m7_K_sweep_{ts}.csv")
    write_csv(all_records, combined_csv)
    print(f"\n  Combined CSV: {combined_csv}  ({len(all_records)} rows)")

    summarize_K_sweep(all_records, k_values)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═'*78}")
    print(bold(f"  M7 K SWEEP COMPLETE  ({elapsed:.1f}s = {elapsed/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
