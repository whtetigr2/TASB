"""
tasb_m7_alpha_sweep.py — M7 sub-sweep 4: α dose-response fine grid
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

ANSWERS: "What is the detailed shape of the bridge's response curve as we
sweep α from 0 to 1.0, with high resolution near the production design
point?"

M5/M6/M7-layer/M7-K/M7-seed have all used a coarse α grid (typically
{0, 0.1, 0.3, 0.5, 1.0}). The dose-response within α=0–0.5 has only been
measured at 0.0, 0.1, 0.3, 0.5 — leaving gaps at 0.05, 0.15, 0.2, 0.25,
0.35, 0.4, 0.7.

This sweep maps the curve at 12 α values, with denser sampling in the
0–0.5 range where production deployment will live.

METHOD: Teacher-forced, L18, K=10, seed=42 (M5/M6 baseline configuration).
Same 4-prompt battery × 40 tokens. Only α varies.

OUTPUT: per-α CSV + combined CSV + dose-response summary showing the
curve shape, where structural property starts to degrade (if it does),
and per-α confident-bucket integrity.

USAGE
-----
    python tasb_m7_alpha_sweep.py             # full, 12 α values, ~30-40 min
    python tasb_m7_alpha_sweep.py --quick     # 4 α × 2 prompts, ~10 min
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
LAYER_IDX = 18
K_SAMPLES = 10
SEED      = 42
BACKEND   = 'exact'

# Fine α grid: dense in 0-0.5 (production range), sparser above.
ALPHA_FULL  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.0]
ALPHA_QUICK = [0.0, 0.1, 0.3, 0.5]

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


# ── Metrics (log-space, no clamp) ─────────────────────────────────────────

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


# ── Per-α measurement ─────────────────────────────────────────────────────

def measure_alpha_on_prompt(model, tok, prompt: dict, tokens: int,
                            alpha: float) -> list[dict]:
    """Teacher-forced measurement at one (prompt, α) cell. Fixed everything
    except α — only the bridge strength varies."""
    records = []
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(tokens):
        sampler_seed = (SEED + zlib.crc32(
            f"sampler|{prompt['id']}|α{alpha}|{step}".encode('utf-8'))
            ) & 0x7FFFFFFF

        # α=0 base (for vanilla logits and identity regression)
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

        # Test α
        if alpha == 0.0:
            # Reuse the base run for the α=0 row (consistent with M5/M6/M7)
            bridge_logits = bridge_a0
        else:
            result = bridge_forward(
                model=model, tok=None, input_ids=ids,
                layer_idx=LAYER_IDX, alpha=alpha, backend=BACKEND,
                K=K_SAMPLES, seed=sampler_seed, return_intermediates=True)
            bridge_logits = result.logits[0, -1, :]

        bridge_top1 = int(bridge_logits.argmax().item())

        records.append({
            'alpha':         alpha,
            'layer_idx':     LAYER_IDX,
            'k_value':       K_SAMPLES,
            'seed':          SEED,
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

def summarize_alpha_sweep(records: list[dict], alphas: list):
    """Print the dose-response curve."""
    print(f"\n{'═'*78}")
    print(bold(f"  M7 α SWEEP — top-1 agreement, KL, JS by α"))
    print(f"  (L{LAYER_IDX}, K={K_SAMPLES}, seed={SEED}, fixed)")
    print(f"{'═'*78}\n")

    print(f"  {'α':>6}  {'n':>5}  {'top1%':>8}  {'top5%':>8}  "
          f"{'mean_KL':>12}  {'mean_JS':>12}  {'α=0 max_diff':>14}")
    print(f"  {'─'*78}")

    for a in alphas:
        rows = [r for r in records if abs(r['alpha'] - a) < 1e-9]
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
        print(f"  {a:>6.2f}  {n:>5}  {color(f'{top1:>6.1f}%')}  "
              f"{top5:>6.1f}%  {mkl:>12.6f}  {mjs:>12.6f}  {a0_str:>14}")

    # Confident-bucket integrity per α
    print(f"\n{'═'*78}")
    print(bold(f"  M7 α SWEEP — Confident-bucket integrity by α"))
    print(f"{'═'*78}\n")
    print(f"  {'α':>6}  {'CONFIDENT %':>14}  {'MODERATE %':>14}  "
          f"{'AMBIGUOUS %':>14}  {'CONF flips':>12}")
    print(f"  {'─'*70}")
    for a in alphas:
        rows = [r for r in records if abs(r['alpha'] - a) < 1e-9]
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
        print(f"  {a:>6.2f}  {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}  "
              f"{flips_str:>12}")

    # Dose-response curve commentary
    print(f"\n{'═'*78}")
    print(bold(f"  M7 α SWEEP — dose-response curve"))
    print(f"{'═'*78}\n")
    sorted_alphas = sorted(set(r['alpha'] for r in records))
    kl_by_alpha = {}
    top1_by_alpha = {}
    for a in sorted_alphas:
        rows = [r for r in records if abs(r['alpha'] - a) < 1e-9]
        if rows:
            kl_by_alpha[a] = np.mean([r['kl'] for r in rows])
            top1_by_alpha[a] = np.mean([r['top1_agree'] for r in rows]) * 100

    print(f"  α progression of mean KL:")
    prev = None
    for a in sorted_alphas:
        kl = kl_by_alpha[a]
        change = f"  ({(kl-prev)/prev*100:+.1f}% vs prev)" if prev and prev > 0 else ""
        print(f"    α={a:>5.2f}:  mean KL = {kl:.6f}{change}")
        prev = kl

    # Find the largest single jump
    print(f"\n  α progression of top-1 agreement:")
    for a in sorted_alphas:
        bar_len = int(top1_by_alpha[a] / 2.5)  # 0-40 chars for 0-100%
        print(f"    α={a:>5.2f}:  {top1_by_alpha[a]:>5.1f}%  {'█'*bar_len}")


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
    args = ap.parse_args()

    if args.quick:
        prompts = PROMPTS_QUICK
        alphas = ALPHA_QUICK
    else:
        prompts = PROMPTS_FULL
        alphas = ALPHA_FULL

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M7 α FINE SWEEP  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:        {args.model}")
    print(f"  Layer:        L{LAYER_IDX}  (M5/M6/M7 baseline)")
    print(f"  K:            {K_SAMPLES}  (M5/M6 baseline)")
    print(f"  Seed:         {SEED}")
    print(f"  Backend:      {BACKEND}")
    print(f"  α grid:       {alphas}")
    print(f"  Prompts:      {len(prompts)} × {args.tokens} tokens")
    print(f"  Cells:        {len(alphas)} α × {len(prompts)} prompts")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    all_records = []
    t_start = time.perf_counter()

    for alpha in alphas:
        t_a = time.perf_counter()
        print(f"\n{bold(f'── α={alpha} ──')}")
        a_records = []
        for prompt in prompts:
            t_prompt = time.perf_counter()
            print(f"  {cyan(prompt['id'])} [{prompt['domain']}]: ",
                  end='', flush=True)
            recs = measure_alpha_on_prompt(
                model=model, tok=tok, prompt=prompt,
                tokens=args.tokens, alpha=alpha)
            elapsed = time.perf_counter() - t_prompt
            a_records.extend(recs)
            agree = np.mean([r['top1_agree'] for r in recs]) * 100 if recs else 0
            kl = np.mean([r['kl'] for r in recs]) if recs else 0
            print(f"top1={agree:>5.1f}%, KL={kl:.5f}  ({elapsed:.0f}s)")
        all_records.extend(a_records)
        a_elapsed = time.perf_counter() - t_a
        print(f"  α={alpha} done in {a_elapsed:.0f}s, {len(a_records)} rows")

        # Per-α CSV
        a_csv = os.path.join(
            args.outdir, f"tasb_m7_alpha_a{int(alpha*100):03d}_{ts}.csv")
        write_csv(a_records, a_csv)

    # Combined CSV
    combined_csv = os.path.join(
        args.outdir, f"tasb_m7_alpha_sweep_{ts}.csv")
    write_csv(all_records, combined_csv)
    print(f"\n  Combined CSV: {combined_csv}  ({len(all_records)} rows)")

    summarize_alpha_sweep(all_records, alphas)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═'*78}")
    print(bold(f"  M7 α SWEEP COMPLETE  ({elapsed:.1f}s = {elapsed/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
