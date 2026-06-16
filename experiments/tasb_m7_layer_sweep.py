"""
tasb_m7_layer_sweep.py — M7: characterization across layer indices
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

ANSWERS: "Is L18 special, or representative?"

M5/M6 were run at L18 (single-layer injection). A reviewing lab will ask
whether L18 is uniquely faithful or whether the bridge works at any layer.
This sweep tests injection at L0, L3, L6, ..., L27 (every 3 layers) on
LLaMA 3.2-3B's 28-layer stack.

METHOD: Teacher-forced, same protocol as M5 (post-RoPE corrected stack).
Each layer gets the full α sweep on a 4-prompt characterization battery.
Smaller battery than M5 — M7 is about scanning parameter space, not
statistical power per cell.

OUTPUT: one CSV per layer + combined summary CSV showing how M5's headline
numbers (top-1, KL, confident-bucket integrity) vary by layer.

USAGE
-----
    python tasb_m7_layer_sweep.py             # full sweep, ~3-4 hr
    python tasb_m7_layer_sweep.py --quick     # 3 layers × 2 prompts, ~30 min
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
BACKEND   = 'exact'
K_SAMPLES = 10

ALPHA_SWEEP = [0.0, 0.1, 0.3, 0.5, 1.0]

# LLaMA 3.2-3B has 28 layers (L0..L27). Sweep every 3.
LAYERS_FULL = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
LAYERS_QUICK = [6, 18, 27]

# 4-prompt characterization battery (factual / code / reasoning / creative)
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


# ── Metrics (carried from M5/M6: log-space, no clamp) ─────────────────────

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


# ── Single-layer measurement (one prompt, all α, teacher-forced) ──────────

def measure_layer_on_prompt(model, tok, prompt: dict, tokens: int,
                            alphas: list, layer_idx: int,
                            base_seed: int) -> list[dict]:
    """Teacher-forced measurement at one (prompt, layer) cell. Identical
    protocol to M5 but parameterized over layer_idx."""
    records = []
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(tokens):
        sampler_seed = (base_seed + zlib.crc32(
            f"sampler|{prompt['id']}|L{layer_idx}|{step}".encode('utf-8'))
            ) & 0x7FFFFFFF

        # α=0 base run (measures identity per step + gives vanilla logits)
        base = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=layer_idx, alpha=0.0, backend=BACKEND, K=K_SAMPLES,
            seed=sampler_seed, return_intermediates=True)
        vanilla_logits = base.vanilla_logits[0, -1, :]
        bridge_a0 = base.logits[0, -1, :]
        alpha0_max_abs_diff = float(
            (vanilla_logits.float() - bridge_a0.float()).abs().max().item())

        vanilla_token = int(vanilla_logits.argmax().item())
        van_top5 = torch.topk(vanilla_logits, 5).indices.tolist()
        prob_gap = _prob_gap(vanilla_logits)

        # α=0 row (measured)
        records.append({
            'layer_idx':    layer_idx,
            'prompt_id':    prompt['id'],
            'domain':       prompt['domain'],
            'step':         step + 1,
            'alpha':        0.0,
            'vanilla_top1': vanilla_token,
            'bridge_top1':  int(bridge_a0.argmax().item()),
            'top1_agree':   int(int(bridge_a0.argmax().item()) == vanilla_token),
            'top5_agree':   int(int(bridge_a0.argmax().item()) in van_top5),
            'kl':           _kl(vanilla_logits, bridge_a0),
            'js':           _js(vanilla_logits, bridge_a0),
            'prob_gap':     prob_gap,
            'alpha0_max_abs_diff': alpha0_max_abs_diff,
        })

        # Non-zero α
        for alpha in alphas:
            if alpha == 0.0:
                continue
            result = bridge_forward(
                model=model, tok=None, input_ids=ids,
                layer_idx=layer_idx, alpha=alpha, backend=BACKEND,
                K=K_SAMPLES, seed=sampler_seed, return_intermediates=True)
            bridge_logits = result.logits[0, -1, :]
            bridge_top1 = int(bridge_logits.argmax().item())
            records.append({
                'layer_idx':    layer_idx,
                'prompt_id':    prompt['id'],
                'domain':       prompt['domain'],
                'step':         step + 1,
                'alpha':        alpha,
                'vanilla_top1': vanilla_token,
                'bridge_top1':  bridge_top1,
                'top1_agree':   int(bridge_top1 == vanilla_token),
                'top5_agree':   int(bridge_top1 in van_top5),
                'kl':           _kl(vanilla_logits, bridge_logits),
                'js':           _js(vanilla_logits, bridge_logits),
                'prob_gap':     prob_gap,
                'alpha0_max_abs_diff': alpha0_max_abs_diff,
            })

        # Teacher-force: advance by vanilla's choice
        ids = torch.cat(
            [ids, torch.tensor([[vanilla_token]], device=model.device)], dim=1)
        if vanilla_token == tok.eos_token_id:
            break

    return records


# ── Aggregate & report ────────────────────────────────────────────────────

def summarize_layer_sweep(records: list[dict], alphas: list, layers: list,
                          alpha_focus: float = 0.3):
    """Print the headline table: layer × α faithfulness."""
    print(f"\n{'═'*78}")
    print(bold(f"  M7 LAYER SWEEP — top-1 agreement by (layer, α)"))
    print(f"{'═'*78}\n")

    hdr = f"  {'layer':>6}  "
    for a in alphas:
        hdr += f"{f'α={a}':>10}  "
    hdr += f"{'α=0 max_diff':>14}"
    print(hdr)
    print(f"  {'─'*len(hdr)}")

    for layer in layers:
        line = f"  L{layer:>4}  "
        for a in alphas:
            rows = [r for r in records
                    if r['layer_idx'] == layer and r['alpha'] == a]
            if rows:
                pct = np.mean([r['top1_agree'] for r in rows]) * 100
                color = green if pct >= 95 else yellow if pct >= 80 else red
                line += color(f"{pct:>9.1f}%") + "  "
            else:
                line += f"{'—':>10}  "
        a0_diffs = [r['alpha0_max_abs_diff'] for r in records
                    if r['layer_idx'] == layer and r['alpha'] == 0.0]
        max_a0 = max(a0_diffs) if a0_diffs else float('inf')
        a0_str = green(f"{max_a0:.2e}") if max_a0 < 1e-5 else red(f"{max_a0:.2e}")
        line += f"{a0_str:>14}"
        print(line)

    print(f"\n{'═'*78}")
    print(bold(f"  M7 LAYER SWEEP — mean KL by (layer, α)"))
    print(f"{'═'*78}\n")
    hdr = f"  {'layer':>6}  "
    for a in alphas:
        hdr += f"{f'α={a}':>12}  "
    print(hdr)
    print(f"  {'─'*len(hdr)}")
    for layer in layers:
        line = f"  L{layer:>4}  "
        for a in alphas:
            rows = [r for r in records
                    if r['layer_idx'] == layer and r['alpha'] == a]
            if rows:
                kl = np.mean([r['kl'] for r in rows])
                line += f"{kl:>11.6f}  "
            else:
                line += f"{'—':>12}  "
        print(line)

    # Confident-bucket structure at α=alpha_focus
    print(f"\n{'═'*78}")
    print(bold(f"  M7 LAYER SWEEP — Confident-bucket integrity @ α={alpha_focus}"))
    print(f"  (zero confident-bucket flips is the structural M5 property)")
    print(f"{'═'*78}\n")
    print(f"  {'layer':>6}  {'CONFIDENT %':>14}  {'MODERATE %':>14}  "
          f"{'AMBIGUOUS %':>14}  {'CONF flips':>12}")
    print(f"  {'─'*70}")
    for layer in layers:
        rows = [r for r in records
                if r['layer_idx'] == layer and r['alpha'] == alpha_focus]
        if not rows:
            continue
        cells = []
        flips = 0
        for lo, hi, _label in [(0.5, 1.01, 'CONFIDENT'),
                                (0.1, 0.5, 'MODERATE'),
                                (0.0, 0.1, 'AMBIGUOUS')]:
            buck = [r for r in rows if lo <= r['prob_gap'] < hi]
            if buck:
                pct = np.mean([r['top1_agree'] for r in buck]) * 100
                cells.append(f"{pct:>5.1f}% (n={len(buck):>3})")
                if _label == 'CONFIDENT':
                    flips = sum(1 for r in buck if r['top1_agree'] == 0)
            else:
                cells.append("—")
        flips_str = green(f"{flips:>3}") if flips == 0 else red(f"{flips:>3}")
        print(f"  L{layer:>4}  {cells[0]:>14}  {cells[1]:>14}  "
              f"{cells[2]:>14}  {flips_str:>12}")


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
    n_layers = mdl.config.num_hidden_layers
    print(f"[SYS_INIT] Ready on {next(mdl.parameters()).device}, "
          f"{n_layers} layers\n", flush=True)
    return mdl, tok, n_layers


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
        layers = LAYERS_QUICK
    else:
        prompts = PROMPTS_FULL
        layers = LAYERS_FULL

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M7 LAYER SWEEP  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:        {args.model}")
    print(f"  Backend:      {BACKEND},  K={K_SAMPLES},  seed={args.seed}")
    print(f"  α sweep:      {ALPHA_SWEEP}")
    print(f"  Prompts:      {len(prompts)} × {args.tokens} tokens")
    print(f"  Layers:       {layers}")
    print(f"  Cells:        {len(layers)} layers × {len(prompts)} prompts × "
          f"{len(ALPHA_SWEEP)} α values")
    print(f"{'═'*78}")

    model, tok, n_layers = load_model(args.model)

    # Sanity check: requested layers exist
    bad = [L for L in layers if L >= n_layers]
    if bad:
        print(red(f"ERROR: layers {bad} exceed model n_layers={n_layers}"))
        sys.exit(1)

    all_records = []
    t_start = time.perf_counter()

    for layer_idx in layers:
        t_layer = time.perf_counter()
        print(f"\n{bold(f'── Layer L{layer_idx} ──')}")
        layer_records = []
        for prompt in prompts:
            t_prompt = time.perf_counter()
            print(f"  {cyan(prompt['id'])} [{prompt['domain']}]: ",
                  end='', flush=True)
            recs = measure_layer_on_prompt(
                model=model, tok=tok, prompt=prompt,
                tokens=args.tokens, alphas=ALPHA_SWEEP,
                layer_idx=layer_idx, base_seed=args.seed)
            elapsed = time.perf_counter() - t_prompt
            layer_records.extend(recs)
            r03 = [r for r in recs if r['alpha'] == 0.3]
            agree = np.mean([r['top1_agree'] for r in r03]) * 100 if r03 else 0
            kl = np.mean([r['kl'] for r in r03]) if r03 else 0
            print(f"α=0.3 top1={agree:>5.1f}%, KL={kl:.5f}  ({elapsed:.0f}s)")
        all_records.extend(layer_records)
        layer_elapsed = time.perf_counter() - t_layer
        print(f"  L{layer_idx} done in {layer_elapsed:.0f}s, "
              f"{len(layer_records)} rows")

        # Write per-layer CSV
        layer_csv = os.path.join(
            args.outdir, f"tasb_m7_layer_L{layer_idx:02d}_{ts}.csv")
        write_csv(layer_records, layer_csv)

    # Combined CSV
    combined_csv = os.path.join(
        args.outdir, f"tasb_m7_layer_sweep_{ts}.csv")
    write_csv(all_records, combined_csv)
    print(f"\n  Combined CSV: {combined_csv}  ({len(all_records)} rows)")

    summarize_layer_sweep(all_records, ALPHA_SWEEP, layers, alpha_focus=0.3)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═'*78}")
    print(bold(f"  M7 LAYER SWEEP COMPLETE  ({elapsed:.1f}s = {elapsed/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
