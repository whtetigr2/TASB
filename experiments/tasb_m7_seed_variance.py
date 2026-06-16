"""
tasb_m7_seed_variance.py — M7 sub-sweep 3: seed variance at production K
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

ANSWERS: "How much does the bridge's output vary just from different sampler
seeds at the production configuration? Is K=10 too noisy, or is the
variance bounded and well-behaved?"

This is the diagnostic that turns "single-seed result" into "characterized
stochastic behavior." A reviewing lab will absolutely ask: "your K=10
headline number is 98.9% top-1, but how much could that number have been
if you'd used a different seed? ±5%? ±0.5%?"

We answer by holding everything else constant and running the M5/M6
protocol at 12 different seeds.

METHOD: Teacher-forced, L18, α=0.3, K=10 (M5/M6 baseline). 12 different
sampler seeds. Same 4-prompt characterization battery × 40 tokens. Each
seed produces its own complete CSV; we aggregate to show:
  - Distribution of top-1 agreement across seeds (mean ± std, min, max)
  - Distribution of mean KL across seeds
  - Per-position seed-agreement: at each (prompt, step) position, how many
    of the 12 seeds agree with vanilla? This is the stochasticity portrait.
  - Confident-bucket integrity per seed (should be 0 flips at all 12)

WHY 12 SEEDS:
  - 5 was the seed-sweep count for M6 trajectory analysis (sufficient for
    pairwise comparisons but small for variance estimation)
  - 12 gives us 11 degrees of freedom for std estimation, which is enough
    to produce honest CIs without being statistically ambitious
  - Codex would call out fewer than 10 as underpowered for variance claims

OUTPUT
------
- Per-seed CSV (12 files): identical M5-shaped format
- Combined CSV with all 12 seeds
- Position-level seed-agreement matrix CSV (one row per prompt-step, with
  agreement count out of 12)
- Console summary with variance characterization

USAGE
-----
    python tasb_m7_seed_variance.py             # full, 12 seeds, ~1.5 hr
    python tasb_m7_seed_variance.py --quick     # 4 seeds × 2 prompts, ~15 min
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
LAYER_IDX = 18                  # M5/M6/M7-layer baseline
ALPHA     = 0.3                 # production design point
K_SAMPLES = 10                  # M5/M6 baseline; K=50 is "better" per M7-K,
                                # but staying at K=10 keeps comparability
BACKEND   = 'exact'

# 12 seeds chosen for diversity (mix of small ints, primes, common defaults).
# Codex-style: enough for honest variance, not so many that we overclaim.
SEEDS_FULL  = [42, 137, 271, 314, 1729, 2718, 7, 99, 12345, 8675309, 0, 65537]
SEEDS_QUICK = [42, 137, 271, 314]

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


# ── Metrics (log-space, no clamp — carried from M5/M6/M7) ─────────────────

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


# ── Single-seed measurement ───────────────────────────────────────────────

def measure_seed_on_prompt(model, tok, prompt: dict, tokens: int,
                           base_seed: int) -> list[dict]:
    """Teacher-forced measurement at one (prompt, seed) cell. Single α=0.3,
    single K, fixed layer — only the seed varies across runs of this fn."""
    records = []
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(tokens):
        sampler_seed = (base_seed + zlib.crc32(
            f"sampler|{prompt['id']}|seed{base_seed}|{step}".encode('utf-8'))
            ) & 0x7FFFFFFF

        # α=0 base for vanilla logits + identity regression
        base = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND, K=K_SAMPLES,
            seed=sampler_seed, return_intermediates=True)
        vanilla_logits = base.vanilla_logits[0, -1, :]
        bridge_a0 = base.logits[0, -1, :]
        alpha0_max_abs_diff = float(
            (vanilla_logits.float() - bridge_a0.float()).abs().max().item())

        vanilla_token = int(vanilla_logits.argmax().item())
        van_top5 = torch.topk(vanilla_logits, 5).indices.tolist()
        prob_gap = _prob_gap(vanilla_logits)

        # α=0.3 (the only non-zero α — seed is the variable here)
        result = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=LAYER_IDX, alpha=ALPHA, backend=BACKEND, K=K_SAMPLES,
            seed=sampler_seed, return_intermediates=True)
        bridge_logits = result.logits[0, -1, :]
        bridge_top1 = int(bridge_logits.argmax().item())

        records.append({
            'base_seed':     base_seed,
            'sampler_seed':  sampler_seed,
            'layer_idx':     LAYER_IDX,
            'alpha':         ALPHA,
            'k_value':       K_SAMPLES,
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

def summarize_seed_variance(records: list[dict], seeds: list):
    """Per-seed summary + variance characterization."""
    print(f"\n{'═'*78}")
    print(bold(f"  M7 SEED VARIANCE — per-seed top-1 agreement and KL"))
    print(f"  (L{LAYER_IDX}, α={ALPHA}, K={K_SAMPLES}, fixed)")
    print(f"{'═'*78}\n")

    print(f"  {'seed':>10}  {'n':>5}  {'top1%':>8}  {'top5%':>8}  "
          f"{'mean_KL':>12}  {'CONF flips':>12}  {'α=0 max_diff':>14}")
    print(f"  {'─'*82}")

    per_seed_top1 = []
    per_seed_kl = []
    per_seed_conf_flips = []

    for seed in seeds:
        rows = [r for r in records if r['base_seed'] == seed]
        if not rows:
            continue
        n = len(rows)
        top1 = np.mean([r['top1_agree'] for r in rows]) * 100
        top5 = np.mean([r['top5_agree'] for r in rows]) * 100
        mkl  = np.mean([r['kl'] for r in rows])
        max_a0 = max(r['alpha0_max_abs_diff'] for r in rows)

        # Confident-bucket flips for this seed
        conf_rows = [r for r in rows if r['prob_gap'] >= 0.5]
        conf_flips = sum(1 for r in conf_rows if r['top1_agree'] == 0)

        per_seed_top1.append(top1)
        per_seed_kl.append(mkl)
        per_seed_conf_flips.append(conf_flips)

        color = green if top1 >= 95 else yellow if top1 >= 80 else red
        flips_str = (green(f"{conf_flips:>3}") if conf_flips == 0
                     else red(f"{conf_flips:>3}"))
        a0_str = (green(f"{max_a0:.2e}") if max_a0 < 1e-5
                  else red(f"{max_a0:.2e}"))
        print(f"  {seed:>10}  {n:>5}  {color(f'{top1:>6.1f}%')}  "
              f"{top5:>6.1f}%  {mkl:>12.6f}  {flips_str:>12}  {a0_str:>14}")

    # Variance summary
    print(f"\n{'═'*78}")
    print(bold(f"  M7 SEED VARIANCE — across-seed statistics ({len(seeds)} seeds)"))
    print(f"{'═'*78}\n")

    if per_seed_top1:
        t1_arr = np.array(per_seed_top1)
        kl_arr = np.array(per_seed_kl)
        cf_arr = np.array(per_seed_conf_flips)

        print(f"  {bold('Top-1 agreement % across seeds:')}")
        print(f"    mean   = {t1_arr.mean():.2f}%")
        print(f"    std    = {t1_arr.std(ddof=1):.3f}%")
        print(f"    min    = {t1_arr.min():.2f}%  (seed {seeds[int(t1_arr.argmin())]})")
        print(f"    max    = {t1_arr.max():.2f}%  (seed {seeds[int(t1_arr.argmax())]})")
        print(f"    range  = {t1_arr.max() - t1_arr.min():.2f}%")

        print(f"\n  {bold('Mean KL across seeds:')}")
        print(f"    mean   = {kl_arr.mean():.6f}")
        print(f"    std    = {kl_arr.std(ddof=1):.6f}")
        print(f"    min    = {kl_arr.min():.6f}")
        print(f"    max    = {kl_arr.max():.6f}")
        print(f"    CV     = {kl_arr.std(ddof=1)/kl_arr.mean()*100:.1f}%  "
              f"(coefficient of variation)")

        print(f"\n  {bold('Confident-bucket flips across seeds:')}")
        max_flips = int(cf_arr.max())
        total_flips = int(cf_arr.sum())
        if max_flips == 0:
            print(green(f"    Zero confident flips on every single seed (12/12). "
                        f"Total: 0."))
            print(green(f"    The structural property is seed-independent."))
        else:
            print(yellow(f"    max flips on any single seed: {max_flips}"))
            print(yellow(f"    total flips across all seeds:  {total_flips}"))
            print(yellow(f"    seeds with ≥1 flip: "
                         f"{sum(1 for c in cf_arr if c > 0)}/{len(seeds)}"))

    # Position-level seed agreement matrix
    print(f"\n{'═'*78}")
    print(bold(f"  M7 SEED VARIANCE — per-position seed agreement"))
    print(f"  (At each (prompt, step), how many of the {len(seeds)} seeds agreed "
          f"with vanilla?)")
    print(f"{'═'*78}\n")

    pos_agree = defaultdict(list)
    for r in records:
        key = (r['prompt_id'], r['step'])
        pos_agree[key].append(r['top1_agree'])

    n_seeds = len(seeds)
    counts = defaultdict(int)
    for key, agrees in pos_agree.items():
        n_agree = sum(agrees)
        counts[n_agree] += 1

    print(f"  {'seeds agreeing':>15}  {'positions':>10}  {'fraction':>10}")
    print(f"  {'─'*40}")
    total_positions = sum(counts.values())
    for n_agree in range(n_seeds + 1):
        n_pos = counts.get(n_agree, 0)
        bar = '█' * int(40 * n_pos / max(total_positions, 1))
        marker = (green(f"{n_agree}/{n_seeds}")
                  if n_agree == n_seeds
                  else yellow(f"{n_agree}/{n_seeds}")
                  if n_agree >= n_seeds // 2
                  else red(f"{n_agree}/{n_seeds}"))
        print(f"  {marker:>20}  {n_pos:>10}  "
              f"{100*n_pos/max(total_positions, 1):>8.1f}%  {bar}")

    full_agreement_pct = 100 * counts.get(n_seeds, 0) / max(total_positions, 1)
    print(f"\n  Positions where ALL {n_seeds} seeds agreed with vanilla: "
          f"{counts.get(n_seeds, 0)}/{total_positions} = "
          f"{full_agreement_pct:.1f}%")
    print(f"  Positions where ZERO seeds agreed with vanilla: "
          f"{counts.get(0, 0)}/{total_positions} = "
          f"{100*counts.get(0, 0)/max(total_positions, 1):.1f}%")


def write_position_matrix_csv(records: list[dict], n_seeds: int, path: str):
    """One row per (prompt, step) with seed-agreement count."""
    pos_data = defaultdict(lambda: {'agrees': 0, 'prob_gap': 0.0,
                                     'vanilla_top1': None})
    for r in records:
        key = (r['prompt_id'], r['step'])
        pos_data[key]['agrees'] += r['top1_agree']
        pos_data[key]['prob_gap'] = r['prob_gap']
        pos_data[key]['vanilla_top1'] = r['vanilla_top1']

    rows = []
    for (pid, step), d in sorted(pos_data.items()):
        rows.append({
            'prompt_id':       pid,
            'step':            step,
            'vanilla_top1':    d['vanilla_top1'],
            'prob_gap':        d['prob_gap'],
            'n_seeds_total':   n_seeds,
            'n_seeds_agree':   d['agrees'],
            'fraction_agree':  d['agrees'] / n_seeds if n_seeds else 0.0,
        })
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_csv(records: list[dict], path: str):
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows := records)


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
    args = ap.parse_args()

    if args.quick:
        prompts = PROMPTS_QUICK
        seeds = SEEDS_QUICK
    else:
        prompts = PROMPTS_FULL
        seeds = SEEDS_FULL

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M7 SEED VARIANCE  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:        {args.model}")
    print(f"  Layer:        L{LAYER_IDX}  (M5/M6 baseline)")
    print(f"  α:            {ALPHA}  (production design point)")
    print(f"  K:            {K_SAMPLES}  (M5/M6 baseline)")
    print(f"  Backend:      {BACKEND}")
    print(f"  Seeds:        {len(seeds)} values: {seeds}")
    print(f"  Prompts:      {len(prompts)} × {args.tokens} tokens")
    print(f"  Cells:        {len(seeds)} seeds × {len(prompts)} prompts × "
          f"{args.tokens} steps")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    all_records = []
    t_start = time.perf_counter()

    for seed in seeds:
        t_seed = time.perf_counter()
        print(f"\n{bold(f'── Seed {seed} ──')}")
        seed_records = []
        for prompt in prompts:
            t_prompt = time.perf_counter()
            print(f"  {cyan(prompt['id'])} [{prompt['domain']}]: ",
                  end='', flush=True)
            recs = measure_seed_on_prompt(
                model=model, tok=tok, prompt=prompt,
                tokens=args.tokens, base_seed=seed)
            elapsed = time.perf_counter() - t_prompt
            seed_records.extend(recs)
            agree = np.mean([r['top1_agree'] for r in recs]) * 100 if recs else 0
            kl = np.mean([r['kl'] for r in recs]) if recs else 0
            print(f"top1={agree:>5.1f}%, KL={kl:.5f}  ({elapsed:.0f}s)")
        all_records.extend(seed_records)
        seed_elapsed = time.perf_counter() - t_seed
        print(f"  seed {seed} done in {seed_elapsed:.0f}s, "
              f"{len(seed_records)} rows")

        # Per-seed CSV
        seed_csv = os.path.join(
            args.outdir, f"tasb_m7_seed_s{seed:08d}_{ts}.csv")
        write_csv(seed_records, seed_csv)

    # Combined CSV + position-matrix CSV
    combined_csv = os.path.join(
        args.outdir, f"tasb_m7_seed_variance_{ts}.csv")
    write_csv(all_records, combined_csv)

    matrix_csv = os.path.join(
        args.outdir, f"tasb_m7_seed_position_matrix_{ts}.csv")
    write_position_matrix_csv(all_records, len(seeds), matrix_csv)

    print(f"\n  Combined CSV:        {combined_csv}  ({len(all_records)} rows)")
    print(f"  Position matrix CSV: {matrix_csv}")

    summarize_seed_variance(all_records, seeds)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═'*78}")
    print(bold(f"  M7 SEED VARIANCE COMPLETE  ({elapsed:.1f}s = {elapsed/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
