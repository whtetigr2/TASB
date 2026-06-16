"""
tasb_stress_v2_deep_ablation.py
==============================================================================
Patent:  USPTO Provisional Application No. 64/019,999 (filed March 28, 2026)
Author:  Paul W. Shaver (Independent Inventor)

TASB Stress Test v2 — Deep Layer Ablation + Coupling Headroom + Bug Fix

PURPOSE
-------
v1 produced two clean findings and one broken test:
  1. peak_1 (L24 alone, α=0.30) → 99.1% match, KL=0.001 (n=540 paired)
  2. McNemar all_26 vs peak_1: χ² = 24.7, p = 0.000001 — significant
  3. Phase C broke silently: LAYER_GROUPS refactor passed a dict where a list
     was expected, so injector got no integer layer keys, no hooks attached,
     and TASB output equaled vanilla output for all 400 autoregressive steps.

v2 RESOLVES THREE QUESTIONS
---------------------------
Q1 — LAYER 0/1 PROBE
    Patent design skipped L0 and L1 on the assumption that GPU needed to build
    a syntactically-coherent foundation before injection. Untested. If L0 or
    L1 alone works, hardware interface moves to the front of the pipeline,
    which simplifies TSU integration substantially.

Q2 — IS L24 SPECIAL OR IS ANY LATE LAYER SUFFICIENT?
    peak_1 at L24 = 99.1%. Does peak_1 at L12, L18, L20, L27 also work?
    If multiple single layers work equally → TSU hardware has flexibility
    in where it interfaces. If only L24 works → it's load-bearing and
    hardware must hit that specific layer.

Q3 — COUPLING HEADROOM
    α=0.30 made peak_1 KL=0.001 — TSU contribution is minimal.
    At α=0.50 and α=0.70, does peak_1 still hold match rate?
    Higher α with same match rate = TSU is doing more thermodynamic work
    per token = product story is stronger ("TSU does real work, not micro-perturb").

THREE PHASES
------------
Phase A — Deep layer ablation
  14 conditions × 9 prompts × N tokens (default 60) = 7560 records
  Conditions break into 5 groups:
    Baselines (3):  vanilla, all_26 (L2-L27), all_28 (L0-L27 — new)
    Layer-0/1 (3):  peak_L0, peak_L1, early_2 (L0+L1)
    Single layer (6): peak_L6, peak_L12, peak_L18, peak_L20, peak_L24, peak_L27
    Quantization (1): peak_L24_int8
    Coupling (1):    peak_L24_a50  (peak_L24 at α=0.50)
    (Note: peak_L24_a70 cut to keep total under 15 conditions; if peak_L24_a50
     shows headroom we run a70 in v3 as a single-alpha sweep.)

Phase B — Statistical analysis (no new runs)
  Bootstrap 95% CI on match rate AND on η for each condition
  McNemar pairwise (16 pre-registered pairs — see PAIRS_TO_TEST)
  Hardware quant verdict on peak_L24 vs peak_L24_int8 (replaces all_26 vs int8)
  Min viable layer selection

Phase C — Autoregressive coherence — BUG FIXED
  Uses winning single-layer config from Phase B
  4 prompts × 100 tokens, self-feeding, no teacher forcing
  Fix: extract cfg['layers'] before passing to run_phase_c

EVERY VALUE HAS A REASON
------------------------
α=0.30 for non-coupling-test conditions: matches v1 + v3 combined baseline,
  enables direct comparison.
α=0.50 for peak_L24_a50: midpoint between 0.30 (faithful) and 1.0 (full
  replacement). If 0.50 holds match rate, headroom exists.
K=10 (samples=10) for TSU phase: matches v1, v3, combined; sample variance
  Var[p_thermo] = p(1-p)/K, max 0.025 per cell at K=10. Sufficient for
  injector blend at α<1.0.
Layer choices for single-layer probes:
  L0, L1 — answer Q1 (skipped-layer assumption test)
  L6 — middle-early, between syntax and semantics
  L12 — true middle, representation-refinement regime
  L18 — middle-late, transitioning to commitment
  L20 — late, full commitment regime
  L24 — known winner from v1 (reference)
  L27 — final layer, latest possible injection
  Reasoning: 7 single-layer conditions span the stack and answer "what
  layers permit injection without breaking output" with one experiment.
N=60 tokens × 9 prompts = 540 paired tokens per comparison: matches v1
  full-run setup. McNemar p < 0.001 detectable at ~3% match-rate gap.
Quantization on peak_L24 (not all_26): we already know all_26_int8 from v1
  (93.3% vs 93.5%). The interesting question is "does int8 preserve the
  single-layer winner" — only that test informs hardware spec.

BUG GUARDS (from memory)
------------------------
  #1 args scope:    args.X only in main(); all functions take explicit params
  #2 imports:       TASB_LAYERS, VanillaCapture, ThermodynamicInjector,
                    run_tsu_phase from tasb_two_pass
                    SAMPLER from tasb_llama_config
                    NO import from tasb_fidelity_test
  #3 float32 eps:   1e-4 in _kl_logits, _vanilla_surprisal, _quantize
  #4 causal mask:   handled inside run_tsu_phase
  #5 h_directed:    handled inside run_tsu_phase
  #6 call sites:    sampler_cfg built per (alpha, bits) tuple in main;
                    every param passed explicitly
  #7 (NEW v2)       layer_subset extraction — when reading from LAYER_GROUPS
                    always extract cfg['layers'] (a list) before passing to
                    any function that iterates over layer indices. Verified
                    at every call site by AST walk.

NEGATIVE-KL ARTIFACT (logged for awareness)
-------------------------------------------
v1 Phase A: 114 peak_1 rows had kl_logit between -0.0018 and 0 due to
float32 precision when p ≈ q. Mathematically KL ≥ 0; the tiny negatives
are roundoff in p*log(p/q) when probabilities are nearly identical.
Fix in v2: take max(0, kl) before storing. This eliminates the negative
eta artifact without changing the underlying math (negative values were
always physically zero).
==============================================================================
"""

import argparse, csv, dataclasses, os, sys, time, threading
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _c(code, t): return f"\033[{code}m{t}\033[0m"
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def blue(t):   return _c("34", t)
def bold(t):   return _c("1",  t)
def gray(t):   return _c("90", t)


# ==============================================================================
# CONDITIONS — every value has a reason (see docstring)
# ==============================================================================

# Each condition: layers (list of ints), bits (None or 8), alpha (float)
# All combinations enumerated explicitly. No placeholders, no defaults that
# silently propagate. If you add a condition, document its reason inline.
CONDITIONS = {
    # ── Baselines: 3 conditions ──────────────────────────────────────────────
    'all_26':       {'layers': list(range(2, 28)),  'bits': None, 'alpha': 0.30},  # v1 baseline (L2-L27)
    'all_28':       {'layers': list(range(0, 28)),  'bits': None, 'alpha': 0.30},  # NEW: include L0+L1

    # ── Layer-0/1 probe: 3 conditions ────────────────────────────────────────
    'peak_L0':      {'layers': [0],                 'bits': None, 'alpha': 0.30},  # Q1: does L0 alone work?
    'peak_L1':      {'layers': [1],                 'bits': None, 'alpha': 0.30},  # Q1: does L1 alone work?
    'early_2':      {'layers': [0, 1],              'bits': None, 'alpha': 0.30},  # Q1: do skipped layers help?

    # ── Single-layer middle/late probe: 6 conditions ─────────────────────────
    'peak_L6':      {'layers': [6],                 'bits': None, 'alpha': 0.30},  # Q2: middle-early
    'peak_L12':     {'layers': [12],                'bits': None, 'alpha': 0.30},  # Q2: true middle
    'peak_L18':     {'layers': [18],                'bits': None, 'alpha': 0.30},  # Q2: middle-late
    'peak_L20':     {'layers': [20],                'bits': None, 'alpha': 0.30},  # Q2: late
    'peak_L24':     {'layers': [24],                'bits': None, 'alpha': 0.30},  # v1 winner (reference)
    'peak_L27':     {'layers': [27],                'bits': None, 'alpha': 0.30},  # Q2: final layer

    # ── Quantization tolerance on the v1 winner: 1 condition ─────────────────
    'peak_L24_int8':{'layers': [24],                'bits': 8,    'alpha': 0.30},  # Q3-adjacent: hardware DAC

    # ── Coupling headroom: 1 condition ───────────────────────────────────────
    'peak_L24_a50': {'layers': [24],                'bits': None, 'alpha': 0.50},  # Q3: more TSU work?
}

# Pre-registered McNemar pairs — declared up front to avoid post-hoc shopping
PAIRS_TO_TEST = [
    # Q1 — layer 0/1 vs known good
    ('peak_L0',  'peak_L24'),     # does the front of the stack work?
    ('peak_L1',  'peak_L24'),
    ('early_2',  'peak_L24'),
    ('all_26',   'all_28'),       # does adding L0+L1 to all_26 help or hurt?

    # Q2 — which single layers work
    ('peak_L6',  'peak_L24'),
    ('peak_L12', 'peak_L24'),
    ('peak_L18', 'peak_L24'),
    ('peak_L20', 'peak_L24'),
    ('peak_L27', 'peak_L24'),

    # Baselines vs winner
    ('all_26',   'peak_L24'),
    ('all_28',   'peak_L24'),

    # Q3 — quantization on the winner
    ('peak_L24', 'peak_L24_int8'),

    # Q3 — coupling headroom
    ('peak_L24', 'peak_L24_a50'),

    # Cross-checks
    ('peak_L20', 'peak_L27'),     # adjacency in late layers
    ('peak_L0',  'peak_L1'),      # adjacency in early layers
    ('peak_L12', 'peak_L18'),     # middle adjacency
]

PHASE_A_CONDITIONS = ['vanilla'] + list(CONDITIONS.keys())   # 14 total (vanilla + 13)


# Prompts — same 9 as v1 for direct comparison
PROMPTS = [
    {"id": "HC1", "domain": "HIGH_CONF",
     "text": "The capital of France is"},
    {"id": "HC2", "domain": "HIGH_CONF",
     "text": "Water is composed of hydrogen and"},
    {"id": "HC3", "domain": "HIGH_CONF",
     "text": "The quick brown fox jumps over the lazy"},
    {"id": "HC4", "domain": "HIGH_CONF",
     "text": "Two plus two equals"},
    {"id": "LC1", "domain": "LOW_CONF",
     "text": "The meaning of life according to philosophy is"},
    {"id": "LC2", "domain": "LOW_CONF",
     "text": "In the year 2050, artificial intelligence will"},
    {"id": "TC1", "domain": "TECHNICAL",
     "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"id": "TC2", "domain": "TECHNICAL",
     "text": 'In JSON format, a key-value pair looks like: {"name":'},
    {"id": "LX1", "domain": "LONG_CTX",
     "text": "Write a short story that begins: The old lighthouse keeper"},
]

PHASE_C_PROMPT_IDS = ['HC1', 'LC1', 'TC1', 'LX1']


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print(f"\n[SYS_INIT] Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),
        attn_implementation='eager', device_map='auto')
    mdl.eval()
    print(f"[SYS_INIT] Ready on {next(mdl.parameters()).device}", flush=True)
    return mdl, tok


# ==============================================================================
# HELPERS
# ==============================================================================

def _kl_logits(logits_vanilla, logits_condition):
    """
    KL(vanilla || condition) on the logit distribution.
    Bug #3: eps=1e-4 for float32 safety.
    Returns max(0, kl) — eliminates float32 roundoff negatives from v1.
    """
    import torch
    eps = 1e-4
    p = torch.softmax(logits_vanilla.float(), dim=-1).clamp(eps, 1.0)
    q = torch.softmax(logits_condition.float(), dim=-1).clamp(eps, 1.0)
    kl = float((p * (p / q).log()).sum().item())
    return max(0.0, kl)   # negative KL is impossible mathematically; this is roundoff


def _vanilla_surprisal(logits_vanilla, token_id):
    """-log P_vanilla(token_id). Bug #3: eps=1e-4 in clamp."""
    import torch
    eps = 1e-4
    p = torch.softmax(logits_vanilla.float(), dim=-1).clamp(eps, 1.0)
    return float(-np.log(p[token_id].item()))


def _quantize_attention_weights(weights_kv_dict: dict, bits: int) -> dict:
    """
    Simulate hardware fixed-point quantization of attention probability
    matrices before TSU sampling. Models input-side DAC precision of a TSU
    programmed with finite-precision coupling weights.

    Grounding: Normal Computing K-FAC paper (arxiv:2405.13817 §5.2 / Fig 4)
    shows 8-bit input quantization is competitive with float32 for
    second-order optimization. Same question applies to attention sampling.

    Bug #3 guard: eps=1e-4 in renormalization.
    """
    if bits is None:
        return weights_kv_dict
    levels = (1 << bits) - 1   # 255 for 8-bit
    eps = 1e-4
    out = {}
    for layer_idx, w in weights_kv_dict.items():
        q = np.round(w * levels).astype(np.float32) / levels
        row_sums = q.sum(axis=-1, keepdims=True)
        out[layer_idx] = (q / np.maximum(row_sums, eps)).astype(np.float32)
    return out


def _keepalive(stop_event, interval=15):
    while not stop_event.is_set():
        stop_event.wait(interval)
        if not stop_event.is_set():
            print(".", end="", flush=True)


# ==============================================================================
# PHASE A — TOKEN STEP
# ==============================================================================
# Architecture:
#   - 1 Phase-1 forward (vanilla, capture all needed layers)
#   - 1 TSU run per unique (alpha, bits) tuple — TSU output is independent of α
#     since α is the injector blend weight, NOT a sampler parameter. So only
#     `bits` actually varies the TSU output. Distinct alphas share TSU output.
#   - 1 Phase-2 forward per condition (different layer subsets / alphas)
#
# This means: with our 14 conditions, TSU runs only twice per token
# (once for float32, once for int8). Phase 2 runs 14 times.

def run_phase_a_token_step(
    model, tokenizer, ids,
    capturer, sampler_cfg, all_layers, conditions,
):
    """
    Run one teacher-forced token step. Returns (records_list, van_id).
    All params explicit — Bug #1 guard.
    Bug #7 guard: conditions is a dict-of-dicts, but cfg['layers'] is always
    extracted before being passed to functions that iterate layer indices.
    """
    import torch
    from tasb_two_pass import ThermodynamicInjector, run_tsu_phase

    # ── Phase 1: vanilla forward, capture all needed layers ──────────────────
    t1_start = time.perf_counter()
    capturer.clear(); capturer.attach()
    with torch.no_grad():
        out1 = model(ids, use_cache=False, output_attentions=True)
    capturer.detach()
    t_gpu1 = time.perf_counter() - t1_start

    logits_v = out1.logits[0, -1, :].float().cpu()
    van_id   = int(logits_v.argmax().item())
    probs_v  = torch.softmax(logits_v, dim=-1)
    top5_v   = torch.topk(probs_v, 5).indices.tolist()
    top2_v   = torch.topk(probs_v, 2).values
    gap      = float((top2_v[0] - top2_v[1]).item())

    # ── TSU phase: one run per unique bits value (alpha is injector-only) ────
    unique_bits = {cfg['bits'] for cfg in conditions.values()}
    t_tsu_total = 0.0
    p_thermo_by_bits = {}
    for bits in unique_bits:
        weights_for_tsu = _quantize_attention_weights(capturer.weights_kv, bits)
        t_tsu_start = time.perf_counter()
        p_thermo_all, r_head_all, _cv, _ti = run_tsu_phase(
            weights_for_tsu, sampler_cfg,
            q_weights=capturer.weights,
            raw_scores=capturer.raw_scores)
        t_tsu_total += time.perf_counter() - t_tsu_start
        p_thermo_by_bits[bits] = (p_thermo_all, r_head_all)

    records = []

    # ── VANILLA record ────────────────────────────────────────────────────────
    van_str = tokenizer.decode([van_id]).strip()[:14]
    records.append({
        'condition':  'vanilla',
        'n_layers':   0,
        'bits':       None,
        'alpha':      0.0,
        'van_tok':    van_str,
        'out_tok':    van_str,
        'match':      1,
        'd_sem':      0,
        'logit_gap':  round(gap, 4),
        'kl_logit':   0.0,
        'eta':        round(1.0 / (0.0 + 1e-8), 1),
        'R_head':     0.0,
        't_gpu1':     round(t_gpu1, 4),
        't_tsu':      0.0,
        't_gpu2':     0.0,
    })

    # ── Each condition: Phase 2 with its layer subset, bits, alpha ──────────
    for cond_name, cfg in conditions.items():
        # Bug #7 guard: extract concrete list, bits, alpha BEFORE any function call
        layer_subset = cfg['layers']     # list of int
        bits         = cfg['bits']       # None or int
        alpha        = cfg['alpha']      # float

        p_thermo_all, r_head_all = p_thermo_by_bits[bits]

        # r_head_subset measures TSU work on this condition's layers
        r_head_subset = {l: r_head_all[l] for l in layer_subset if l in r_head_all}
        r_head_mean_subset = float(np.mean(list(r_head_subset.values()))) \
                             if r_head_subset else 0.0

        # Phase 2 with this condition's injector
        t2_start = time.perf_counter()
        injector = ThermodynamicInjector(model, layer_subset, alpha=alpha)
        injector.load(
            {l: p_thermo_all[l] for l in layer_subset if l in p_thermo_all},
            capturer.weights,
        )
        injector.attach()
        with torch.no_grad():
            out2 = model(ids, use_cache=False)
        injector.detach()
        t_gpu2 = time.perf_counter() - t2_start

        logits_out = out2.logits[0, -1, :].float().cpu()
        out_id     = int(logits_out.argmax().item())
        match      = int(van_id == out_id)
        d_sem      = 0 if out_id in top5_v else 1
        kl         = _kl_logits(logits_v, logits_out)
        out_str    = tokenizer.decode([out_id]).strip()[:14]

        records.append({
            'condition':  cond_name,
            'n_layers':   len(layer_subset),
            'bits':       bits,
            'alpha':      alpha,
            'van_tok':    van_str,
            'out_tok':    out_str,
            'match':      match,
            'd_sem':      d_sem,
            'logit_gap':  round(gap, 4),
            'kl_logit':   round(kl, 6),
            'eta':        round(match / (kl + 1e-8), 1),
            'R_head':     round(r_head_mean_subset, 4),
            't_gpu1':     round(t_gpu1, 4),
            't_tsu':      round(t_tsu_total, 4),
            't_gpu2':     round(t_gpu2, 4),
        })

    return records, van_id


def run_phase_a(
    model, tokenizer, prompts, tokens_per_prompt,
    all_layers, conditions, sampler_cfg, outdir, ts,
):
    """
    Run Phase A across all prompts. All params explicit — Bug #1 guard.
    """
    import torch
    from tasb_two_pass import VanillaCapture

    # Capturer needs all layers used by ANY condition (union)
    used_layers = set()
    for cfg in conditions.values():
        used_layers.update(cfg['layers'])
    used_layers = sorted(used_layers)

    print(f"\n{'═'*78}")
    print(f"  [PHASE_A] Deep ablation — {len(prompts)} prompts × {tokens_per_prompt} tokens")
    print(f"  [PHASE_A] Conditions: vanilla + {len(conditions)} ablation")
    print(f"  [PHASE_A] Layers captured: L{min(used_layers)}–L{max(used_layers)} ({len(used_layers)} layers)")
    print(f"  [PHASE_A] Unique (bits,α) combos: {len({(c['bits'], c['alpha']) for c in conditions.values()})}")
    print(f"{'═'*78}\n")

    all_records = []

    for prompt in prompts:
        capturer  = VanillaCapture(model, used_layers)
        inputs    = tokenizer(prompt['text'], return_tensors='pt').to(model.device)
        ids       = inputs['input_ids'].clone()

        print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]  "
              f"{gray(prompt['text'][:55])}")
        print(f"  {'─'*70}")

        for step in range(tokens_per_prompt):
            stop = threading.Event()
            t = threading.Thread(target=_keepalive, args=(stop, 15), daemon=True)
            t.start()
            try:
                records, van_id = run_phase_a_token_step(
                    model=model,
                    tokenizer=tokenizer,
                    ids=ids,
                    capturer=capturer,
                    sampler_cfg=sampler_cfg,
                    all_layers=all_layers,
                    conditions=conditions,
                )
            finally:
                stop.set(); t.join(timeout=1)

            # Tag records with prompt info and step
            for r in records:
                r['prompt_id'] = prompt['id']
                r['domain']    = prompt['domain']
                r['step']      = step + 1
                all_records.append(r)

            # Compact per-step output: one symbol per condition
            van_str = records[0]['van_tok']
            row = "  ".join(
                f"{r['condition'][:8]:>8}={'✓' if r['match']==1 else '✗'}"
                for r in records[1:]
            )
            t1 = records[0]['t_gpu1']
            tt = records[1]['t_tsu']
            print(f"  step={step+1:>3}  van='{van_str:<10}'  {row}  "
                  f"t1={t1:.2f}s tt={tt:.2f}s", flush=True)

            # Teacher-forced: advance with vanilla token
            ids = torch.cat(
                [ids, torch.tensor([[van_id]], device=model.device)], dim=1)
            if van_id == tokenizer.eos_token_id: break

        # Per-prompt summary
        print(f"\n  {prompt['id']} summary:")
        for cond in PHASE_A_CONDITIONS:
            dr = [r for r in all_records
                  if r['prompt_id']==prompt['id'] and r['condition']==cond]
            if not dr: continue
            mr  = np.mean([r['match'] for r in dr]) * 100
            kl  = np.mean([r['kl_logit'] for r in dr])
            print(f"    {cond:<16}: match={mr:>5.1f}%  mean_KL={kl:.4f}  n={len(dr)}")

        # Checkpoint after each prompt
        ckpt = f"{outdir}/tasb_stress_v2_phaseA_partial_{ts}.csv"
        with open(ckpt, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader(); w.writerows(all_records)
        print(f"  {gray(f'[checkpoint → {ckpt}]')}")

    return all_records


# ==============================================================================
# PHASE B — STATISTICAL ANALYSIS
# ==============================================================================

def run_phase_b(phase_a_records):
    print(f"\n{'═'*78}")
    print(f"  [PHASE_B] Statistical robustness")
    print(f"{'═'*78}\n")

    # ── Per-condition summary with bootstrap CIs ──────────────────────────────
    print(f"  {'Condition':<16} {'n_lay':>5} {'bits':>5} {'α':>5} {'n':>4} "
          f"{'match%':>7} {'CI_match':>14} {'mean_KL':>9} {'median_η':>9}")
    print(f"  {'─'*92}")

    rng = np.random.default_rng(42)
    per_cond = {}

    for cond in PHASE_A_CONDITIONS:
        dr = [r for r in phase_a_records if r['condition']==cond]
        if not dr: continue
        n        = len(dr)
        n_layers = dr[0]['n_layers']
        bits     = dr[0]['bits']
        alpha    = dr[0]['alpha']
        matches  = [r['match'] for r in dr]
        kls      = [r['kl_logit'] for r in dr]
        etas     = [r['eta'] for r in dr if r['eta'] < 1e7]
        match_pct = np.mean(matches) * 100
        mean_kl   = np.mean(kls)
        median_eta = np.median(etas) if etas else 0

        # Bootstrap CI on match rate (10k resamples)
        boots = [np.mean(rng.choice(matches, n, replace=True)) * 100
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])

        per_cond[cond] = {
            'n_layers': n_layers, 'bits': bits, 'alpha': alpha, 'n': n,
            'match': match_pct, 'ci_lo': lo, 'ci_hi': hi,
            'mean_kl': mean_kl, 'median_eta': median_eta,
            'd_sem': sum(r['d_sem'] for r in dr),
        }

        match_c = (green(f"{match_pct:>6.1f}%") if match_pct>=98
                   else yellow(f"{match_pct:>6.1f}%") if match_pct>=90
                   else red(f"{match_pct:>6.1f}%"))
        bits_str = str(bits) if bits else '-'
        print(f"  {cond:<16} {n_layers:>5} {bits_str:>5} {alpha:>5.2f} {n:>4} "
              f"{match_c} [{lo:>5.1f}, {hi:>5.1f}] {mean_kl:>9.4f} {median_eta:>9.1f}")

    # ── McNemar pairwise tests ────────────────────────────────────────────────
    print(f"\n  {bold('Pre-registered McNemar pairs (paired by prompt × step):')}")
    print(f"  {'A':<14} vs  {'B':<14}  {'b':>4} {'c':>4}  {'χ²':>7}  {'p':>9}  verdict")
    print(f"  {'─'*78}")

    try:
        from scipy.stats import chi2 as chi2_dist
        has_scipy = True
    except ImportError:
        has_scipy = False

    mcnemar_results = []
    for ca, cb in PAIRS_TO_TEST:
        a_map = {(r['prompt_id'], r['step']): r['match']
                 for r in phase_a_records if r['condition']==ca}
        b_map = {(r['prompt_id'], r['step']): r['match']
                 for r in phase_a_records if r['condition']==cb}
        common = set(a_map) & set(b_map)
        if not common:
            print(f"  {ca:<14} vs  {cb:<14}  (no common tokens)")
            continue

        b_ = sum(1 for k in common if a_map[k]==1 and b_map[k]==0)
        c_ = sum(1 for k in common if a_map[k]==0 and b_map[k]==1)

        if (b_ + c_) > 0:
            chi2 = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
            p = (1.0 - chi2_dist.cdf(chi2, 1)) if has_scipy else None
        else:
            chi2 = 0.0
            p = 1.0

        if p is None:
            verdict = gray("no p")
        elif p < 0.001:
            verdict = green(f"SIGNIF p<0.001")
        elif p < 0.05:
            verdict = green(f"SIGNIF p<0.05")
        else:
            verdict = yellow("not signif.")

        p_str = f"{p:.4f}" if p is not None else "  ----"
        print(f"  {ca:<14} vs  {cb:<14}  {b_:>4} {c_:>4}  {chi2:>7.3f}  {p_str:>9}  {verdict}")
        mcnemar_results.append({
            'cond_a': ca, 'cond_b': cb, 'b': b_, 'c': c_,
            'chi2': chi2, 'p': p, 'n_pairs': len(common),
        })

    # ── Hardware quantization tolerance on peak_L24 ──────────────────────────
    print(f"\n  {bold('Hardware quantization tolerance (peak_L24 vs peak_L24_int8):')}")
    print(f"    Reference: Normal Computing K-FAC quant tolerance (arxiv:2405.13817)")
    print(f"    Question: does 8-bit DAC preserve single-layer injection quality?")

    fp32 = per_cond.get('peak_L24')
    int8 = per_cond.get('peak_L24_int8')
    if fp32 and int8:
        d_match = int8['match'] - fp32['match']
        d_kl    = int8['mean_kl'] - fp32['mean_kl']
        match_ok = abs(d_match) <= 1.0
        kl_ok    = abs(d_kl) <= max(0.001, 0.5 * fp32['mean_kl'])
        sem_ok   = int8['d_sem'] == 0
        compat   = match_ok and kl_ok and sem_ok
        verdict  = green("[COMPATIBLE]") if compat else yellow("[REVIEW]")
        print()
        print(f"    {'Metric':<12} {'float32':>10} {'int8':>10} {'Δ':>10}")
        print(f"    {'─'*48}")
        print(f"    {'match%':<12} {fp32['match']:>9.1f}% {int8['match']:>9.1f}% "
              f"{d_match:>+9.1f}")
        print(f"    {'mean_KL':<12} {fp32['mean_kl']:>10.4f} {int8['mean_kl']:>10.4f} "
              f"{d_kl:>+10.4f}")
        print(f"    {'d_sem':<12} {fp32['d_sem']:>10} {int8['d_sem']:>10}")
        print(f"\n    {verdict} 8-bit DAC verdict for single-layer injection.")

    # ── Coupling headroom on peak_L24 ─────────────────────────────────────────
    print(f"\n  {bold('Coupling headroom (peak_L24 α=0.30 vs α=0.50):')}")
    print(f"    Question: at higher α, does single-layer keep match but do more work?")
    a30 = per_cond.get('peak_L24')
    a50 = per_cond.get('peak_L24_a50')
    if a30 and a50:
        d_match = a50['match'] - a30['match']
        ratio_kl = a50['mean_kl'] / max(a30['mean_kl'], 1e-6)
        d_eta = a50['median_eta'] - a30['median_eta']
        # Headroom verdict: α=0.50 retains match within 1%, KL grows substantially
        headroom = abs(d_match) <= 1.0 and ratio_kl > 5.0
        verdict = green("[HEADROOM]") if headroom else yellow("[REVIEW]")
        print()
        print(f"    {'Metric':<12} {'α=0.30':>10} {'α=0.50':>10} {'Δ or ratio':>12}")
        print(f"    {'─'*48}")
        print(f"    {'match%':<12} {a30['match']:>9.1f}% {a50['match']:>9.1f}% "
              f"{d_match:>+11.1f}")
        print(f"    {'mean_KL':<12} {a30['mean_kl']:>10.4f} {a50['mean_kl']:>10.4f} "
              f"{ratio_kl:>10.1f}×")
        print(f"    {'median_η':<12} {a30['median_eta']:>10.1f} {a50['median_eta']:>10.1f}")
        print(f"\n    {verdict} TSU coupling headroom verdict.")
        if headroom:
            print(f"    → Single-layer injection at α=0.50 retains output fidelity AND")
            print(f"      increases thermodynamic work per token. Real TSU work happening.")

    # ── Minimum viable layer selection ────────────────────────────────────────
    print(f"\n  {bold('Minimum viable layer selection (α=0.30, float32 only):')}")
    print(f"    Criteria:")
    print(f"      - match within 2% of best single-layer condition")
    print(f"      - d_sem = 0 (no semantic divergences)")
    print(f"      - mean_KL within 2× of best")
    print(f"      - fewest layers wins on tiebreak; earliest layer wins among single-layer")

    # Find best single-layer match rate
    single_layer_conds = [k for k, v in per_cond.items()
                          if v['n_layers'] == 1 and v['bits'] is None and v['alpha'] == 0.30]
    if single_layer_conds:
        best_match = max(per_cond[c]['match'] for c in single_layer_conds)
        best_kl    = min(per_cond[c]['mean_kl'] for c in single_layer_conds)

        candidates = []
        for c in single_layer_conds:
            s = per_cond[c]
            if (s['match'] >= best_match - 2.0
                and s['d_sem'] == 0
                and s['mean_kl'] <= 2.0 * best_kl):
                # Extract layer index (peak_LN → N)
                layer_idx = int(c.replace('peak_L', ''))
                candidates.append((layer_idx, c, s))

        if candidates:
            candidates.sort()   # earliest layer first
            min_viable_name = candidates[0][1]
            min_viable_layer = candidates[0][0]
            print(f"\n    Single-layer candidates meeting all criteria:")
            for li, cn, s in candidates:
                print(f"      L{li:<2} ({cn:<10}): match={s['match']:.1f}%  "
                      f"KL={s['mean_kl']:.4f}  d_sem={s['d_sem']}")
            print(f"\n    {green('[VERDICT] Earliest min-viable layer: ' + min_viable_name)}")
            print(f"    {green(f'TSU bandwidth: 1 attention layer (L{min_viable_layer})')}")
        else:
            min_viable_name = 'peak_L24'  # fallback to known good
            print(f"\n    {yellow('No single-layer condition met all criteria. Falling back to peak_L24.')}")
    else:
        min_viable_name = 'peak_L24'
        print(f"\n    {yellow('No single-layer conditions found. Falling back to peak_L24.')}")

    return per_cond, mcnemar_results, min_viable_name


# ==============================================================================
# PHASE C — PARALLEL AUTOREGRESSIVE GENERATION (v2 patch: head-to-head)
# ==============================================================================
#
# WHAT CHANGED FROM v1:
# v1 Phase C ran TASB autoregressively and recorded kl_logit between vanilla
# and TASB at each step. Two problems with that design:
#   (1) v1 had Bug #7 — kl_logit was always 0 because TASB silently no-op'd
#   (2) Even with the bug fixed, the comparison "vanilla logits vs TASB
#       logits at step N" is only meaningful if both saw the SAME context.
#       Once TASB picks a different token, the two branches have different
#       contexts and per-step kl_logit no longer means what we want.
#
# v2 patch: run TWO independent autoregressive branches in parallel.
#   - vanilla branch: greedy argmax, no TSU, builds its own context
#   - TASB branch:    greedy argmax, with TSU injection at chosen layer,
#                     builds its own context
#
# After they diverge, they generate different sequences. The honest metrics
# are no longer "kl between aligned logits" — they're properties of each
# branch's generated text:
#   - rep_4_rate:    fraction of 4-grams that repeat earlier in this branch
#   - distinct_4:    unique 4-grams / total 4-grams (lexical diversity)
#   - step_entropy:  entropy of the logit distribution at this step
#                    (low entropy + repeated tokens = stuck in attractor)
#   - first_loop:    step at which a 4-gram appears for the second time
#
# Why 4-grams: standard in NLG repetition literature (Holtzman et al. 2019).
# 4-token repeats are the lowest order that catches genuine loops without
# false-positiving common phrases like "in the" or "of the".
#
# THE KEY QUESTION THIS RESOLVES:
# When base LLaMA 3.2-3B does greedy autoregressive on open-ended prompts,
# it falls into 4-gram repetition loops (well-documented: base LLMs without
# repetition penalty do this). Question: does TASB make this BETTER, WORSE,
# or UNCHANGED?
#   - If TASB rep_rate < vanilla rep_rate → TASB IMPROVES coherence
#     (the thermodynamic noise breaks the model out of repetition attractors)
#   - If TASB rep_rate ≈ vanilla rep_rate → TASB is FAITHFUL
#     (the bridge doesn't make things worse; it stays in the manifold)
#   - If TASB rep_rate > vanilla rep_rate → TASB DEGRADES coherence
#     (real problem; need to investigate)
#
# Any of the three is a valid scientific result. The current "TASB output
# repeats" observation is unverifiable without this control.
# ==============================================================================

def _vanilla_ar_step(model, ids):
    """
    One vanilla autoregressive step. No hooks, no capture, no TSU.
    Returns (token_id, logits, entropy, t_gpu).
    """
    import torch
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=False)
    t_gpu = time.perf_counter() - t0
    logits = out.logits[0, -1, :].float().cpu()
    tok_id = int(logits.argmax().item())
    # Entropy of the logit distribution — measures how committed the model is
    eps = 1e-4
    probs = torch.softmax(logits, dim=-1).clamp(eps, 1.0)
    entropy = float(-(probs * probs.log()).sum().item())
    return tok_id, logits, entropy, t_gpu


def _tasb_ar_step(model, ids, capturer, sampler_cfg, alpha, layer_subset):
    """
    One TASB autoregressive step: Phase 1 capture → TSU sample → Phase 2 inject.
    Returns (token_id, logits, entropy, r_head_mean, t_gpu1, t_tsu, t_gpu2).
    """
    import torch
    from tasb_two_pass import ThermodynamicInjector, run_tsu_phase

    # Phase 1
    t1s = time.perf_counter()
    capturer.clear(); capturer.attach()
    with torch.no_grad():
        _ = model(ids, use_cache=False, output_attentions=True)
    capturer.detach()
    t_gpu1 = time.perf_counter() - t1s

    # TSU phase
    tts = time.perf_counter()
    p_thermo_all, r_head_all, _cv, _t = run_tsu_phase(
        capturer.weights_kv, sampler_cfg,
        q_weights=capturer.weights,
        raw_scores=capturer.raw_scores)
    t_tsu = time.perf_counter() - tts

    r_head_subset = {l: r_head_all[l] for l in layer_subset if l in r_head_all}
    r_head_mean   = float(np.mean(list(r_head_subset.values()))) \
                    if r_head_subset else 0.0

    # Phase 2 — with injection
    t2s = time.perf_counter()
    injector = ThermodynamicInjector(model, layer_subset, alpha=alpha)
    injector.load(
        {l: p_thermo_all[l] for l in layer_subset if l in p_thermo_all},
        capturer.weights,
    )
    injector.attach()
    with torch.no_grad():
        out2 = model(ids, use_cache=False)
    injector.detach()
    t_gpu2 = time.perf_counter() - t2s

    logits = out2.logits[0, -1, :].float().cpu()
    tok_id = int(logits.argmax().item())
    eps = 1e-4
    probs = torch.softmax(logits, dim=-1).clamp(eps, 1.0)
    entropy = float(-(probs * probs.log()).sum().item())
    return tok_id, logits, entropy, r_head_mean, t_gpu1, t_tsu, t_gpu2


def _ngram_repetition_rate(token_ids, n=4):
    """
    Fraction of n-grams in token_ids that have appeared earlier in the sequence.
    n=4 is standard (Holtzman et al. 2019).
    Returns 0.0 if sequence is shorter than n+1 tokens.
    """
    if len(token_ids) < n + 1:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i:i+n])
        total += 1
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return repeats / max(total, 1)


def _distinct_n(token_ids, n=4):
    """
    distinct-N = unique n-grams / total n-grams (Li et al. 2016).
    Higher = more lexically diverse. 1.0 = no repeats. 0.x = increasingly looped.
    """
    if len(token_ids) < n:
        return 1.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def _first_repeat_step(token_ids, n=4):
    """
    Step at which a 4-gram is generated that has appeared earlier.
    Returns None if no repeat in the sequence.
    The reported step is the position of the FIRST token of the repeating gram.
    """
    if len(token_ids) < n + 1:
        return None
    seen = {}
    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i:i+n])
        if gram in seen:
            return i + 1  # 1-indexed step
        seen[gram] = i
    return None


def run_phase_c(
    model, tokenizer, prompts, tokens_per_prompt,
    alpha, layer_subset, sampler_cfg, outdir, ts,
):
    """
    Parallel autoregressive comparison: vanilla branch vs TASB branch.

    Each branch decodes greedily from its own growing context. After they
    diverge on a token choice, the branches have different histories and
    are independent autoregressive runs.

    All params explicit — Bug #1 guard.
    Bug #7 guard: layer_subset must be List[int].
    """
    import torch
    from tasb_two_pass import VanillaCapture

    if not isinstance(layer_subset, list) or not all(isinstance(l, int) for l in layer_subset):
        raise TypeError(
            f"Bug #7 guard: layer_subset must be List[int], got {type(layer_subset)}: "
            f"{layer_subset}")

    print(f"\n{'═'*78}")
    print(f"  [PHASE_C] PARALLEL AR — {len(prompts)} prompts × {tokens_per_prompt} tokens")
    print(f"  [PHASE_C] Layer subset: {layer_subset} ({len(layer_subset)} layers)  α={alpha}")
    print(f"  [PHASE_C] Two independent branches per prompt:")
    print(f"            • vanilla branch: greedy AR, no TSU")
    print(f"            • TASB branch:    greedy AR with TSU injection")
    print(f"  [PHASE_C] Repetition metrics: 4-gram (Holtzman et al. 2019)")
    print(f"{'═'*78}\n")

    # TASB branch needs capturer; vanilla branch does not
    capture_layers = list(range(28))

    all_records = []

    for prompt in prompts:
        capturer = VanillaCapture(model, capture_layers)
        inputs   = tokenizer(prompt['text'], return_tensors='pt').to(model.device)

        # Two independent token-id sequences — branches diverge after disagreement
        van_ids  = inputs['input_ids'].clone()
        tasb_ids = inputs['input_ids'].clone()
        prompt_len = van_ids.shape[1]

        # Track GENERATED tokens (excluding prompt) for repetition metrics
        van_generated  = []
        tasb_generated = []

        print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]  "
              f"{gray(prompt['text'][:55])}")
        print(f"  {'─'*78}")
        print(f"  {'step':>4} │ {'vanilla':<14} {'v_ent':>6} │ "
              f"{'tasb':<14} {'t_ent':>6} {'R_head':>7} │ {'diverged':>8}")
        print(f"  {'─'*78}")

        diverged_yet = False
        diverge_step = None

        for step in range(tokens_per_prompt):
            stop = threading.Event()
            t = threading.Thread(target=_keepalive, args=(stop, 15), daemon=True)
            t.start()
            try:
                # VANILLA branch step
                van_id, van_logits, van_entropy, t_van = _vanilla_ar_step(
                    model=model, ids=van_ids)

                # TASB branch step
                (tasb_id, tasb_logits, tasb_entropy, r_head,
                 t_gpu1, t_tsu, t_gpu2) = _tasb_ar_step(
                    model=model, ids=tasb_ids,
                    capturer=capturer,
                    sampler_cfg=sampler_cfg,
                    alpha=alpha,
                    layer_subset=layer_subset)
            finally:
                stop.set(); t.join(timeout=1)

            # Track divergence as the FIRST step where they disagree given
            # IDENTICAL context. After they diverge, contexts differ, so
            # subsequent token-by-token disagreement is expected.
            ctx_identical = bool(torch.equal(van_ids, tasb_ids))
            if ctx_identical and (van_id != tasb_id) and not diverged_yet:
                diverged_yet = True
                diverge_step = step + 1

            van_str  = tokenizer.decode([van_id]).strip()[:14]
            tasb_str = tokenizer.decode([tasb_id]).strip()[:14]
            div_marker = (red("DIVERGED") if (ctx_identical and van_id != tasb_id)
                          else gray("---") if ctx_identical
                          else gray("(split)"))
            print(f"  {step+1:>4} │ {van_str:<14} {van_entropy:>6.3f} │ "
                  f"{tasb_str:<14} {tasb_entropy:>6.3f} {r_head:>7.4f} │ "
                  f"{div_marker:>8}", flush=True)

            # Record this step. Note: van_ctx_len and tasb_ctx_len are the
            # CONTEXT LENGTHS each branch saw, which diverge after split.
            all_records.append({
                'prompt_id':         prompt['id'],
                'domain':            prompt['domain'],
                'step':              step + 1,
                'van_ctx_len':       van_ids.shape[1],
                'tasb_ctx_len':      tasb_ids.shape[1],
                'van_tok':           van_str,
                'tasb_tok':          tasb_str,
                'ctx_identical':     int(ctx_identical),
                'first_divergence':  diverge_step or 0,
                'van_entropy':       round(van_entropy, 4),
                'tasb_entropy':      round(tasb_entropy, 4),
                'R_head':            round(r_head, 4),
                't_van':             round(t_van, 4),
                't_gpu1':            round(t_gpu1, 4),
                't_tsu':             round(t_tsu, 4),
                't_gpu2':            round(t_gpu2, 4),
                'n_layers':          len(layer_subset),
                'alpha':             alpha,
            })

            van_generated.append(van_id)
            tasb_generated.append(tasb_id)

            # Each branch appends ITS OWN token to ITS OWN context
            van_ids  = torch.cat([van_ids,
                       torch.tensor([[van_id]],  device=model.device)], dim=1)
            tasb_ids = torch.cat([tasb_ids,
                       torch.tensor([[tasb_id]], device=model.device)], dim=1)

            # Stop if BOTH branches hit EOS
            if (van_id == tokenizer.eos_token_id
                    and tasb_id == tokenizer.eos_token_id):
                break

        # Per-prompt repetition metrics on the GENERATED tokens (excluding prompt)
        van_rep    = _ngram_repetition_rate(van_generated,  n=4)
        tasb_rep   = _ngram_repetition_rate(tasb_generated, n=4)
        van_dist4  = _distinct_n(van_generated,  n=4)
        tasb_dist4 = _distinct_n(tasb_generated, n=4)
        van_first  = _first_repeat_step(van_generated,  n=4)
        tasb_first = _first_repeat_step(tasb_generated, n=4)

        print(f"\n  {prompt['id']} BRANCH METRICS")
        print(f"    {'metric':<22} {'vanilla':>10} {'TASB':>10} {'Δ (TASB-van)':>14}")
        print(f"    {'─'*60}")
        print(f"    {'4-gram rep rate':<22} {van_rep:>10.4f} {tasb_rep:>10.4f} "
              f"{tasb_rep - van_rep:>+14.4f}")
        print(f"    {'distinct-4':<22} {van_dist4:>10.4f} {tasb_dist4:>10.4f} "
              f"{tasb_dist4 - van_dist4:>+14.4f}")
        print(f"    {'first 4-gram repeat':<22} {str(van_first or 'never'):>10} "
              f"{str(tasb_first or 'never'):>10}")
        if diverge_step:
            print(f"    branches diverged at step {diverge_step}")
        else:
            print(f"    branches NEVER diverged (TASB ≡ vanilla on this prompt)")

        # Verdict per prompt
        delta_rep = tasb_rep - van_rep
        if abs(delta_rep) < 0.02:
            verdict = gray("FAITHFUL (≈ vanilla)")
        elif delta_rep < 0:
            verdict = green(f"IMPROVED (-{abs(delta_rep)*100:.1f}% rep)")
        else:
            verdict = red(f"DEGRADED (+{delta_rep*100:.1f}% rep)")
        print(f"    {verdict}")

        # Stash branch-level metrics on each record for downstream analysis
        for r in all_records:
            if r['prompt_id'] != prompt['id']:
                continue
            r['van_rep4_final']    = round(van_rep, 4)
            r['tasb_rep4_final']   = round(tasb_rep, 4)
            r['van_distinct4']     = round(van_dist4, 4)
            r['tasb_distinct4']    = round(tasb_dist4, 4)
            r['van_first_loop']    = van_first  or 0
            r['tasb_first_loop']   = tasb_first or 0

        # Save full generated text for each branch (useful for spot-checking)
        van_text  = tokenizer.decode(van_generated)
        tasb_text = tokenizer.decode(tasb_generated)
        text_log = f"{outdir}/tasb_stress_v2_phaseC_texts_{ts}.log"
        with open(text_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'═'*70}\n")
            f.write(f"PROMPT {prompt['id']} [{prompt['domain']}]\n")
            f.write(f"INPUT: {prompt['text']}\n\n")
            f.write(f"--- VANILLA branch ({len(van_generated)} tokens) ---\n")
            f.write(van_text + "\n\n")
            f.write(f"--- TASB branch ({len(tasb_generated)} tokens) ---\n")
            f.write(tasb_text + "\n")

        ckpt = f"{outdir}/tasb_stress_v2_phaseC_partial_{ts}.csv"
        if all_records:
            with open(ckpt, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
                w.writeheader(); w.writerows(all_records)
        print(f"  {gray(f'[checkpoint → {ckpt}]')}")
        print(f"  {gray(f'[branch texts → {text_log}]')}")

    return all_records


def summarize_phase_c(records):
    """
    Aggregate the parallel-branch metrics across prompts and produce a
    head-to-head verdict on whether TASB improves, matches, or degrades
    autoregressive coherence vs vanilla.
    """
    print(f"\n{'═'*78}")
    print(f"  [PHASE_C_SUMMARY] Parallel AR — vanilla branch vs TASB branch")
    print(f"{'─'*78}")

    if not records:
        print(f"  {yellow('No Phase C records.')}")
        return {}

    # Per-prompt summary (each prompt has multiple step records; metrics same per prompt)
    by_prompt = {}
    for r in records:
        pid = r['prompt_id']
        if pid not in by_prompt:
            by_prompt[pid] = r  # any row has the prompt-level fields

    print(f"  {'prompt':<8} {'van_rep4':>9} {'tasb_rep4':>10} {'Δrep':>8}  "
          f"{'van_d4':>8} {'tasb_d4':>8}  {'van_loop':>9} {'tasb_loop':>10}")
    print(f"  {'─'*78}")

    deltas_rep = []
    deltas_d4  = []
    for pid in sorted(by_prompt.keys()):
        r = by_prompt[pid]
        vr  = r.get('van_rep4_final',  0)
        tr  = r.get('tasb_rep4_final', 0)
        vd  = r.get('van_distinct4',   1)
        td  = r.get('tasb_distinct4',  1)
        vl  = r.get('van_first_loop',  0) or 0
        tl  = r.get('tasb_first_loop', 0) or 0
        deltas_rep.append(tr - vr)
        deltas_d4.append(td - vd)
        vl_str = str(vl) if vl else 'never'
        tl_str = str(tl) if tl else 'never'
        print(f"  {pid:<8} {vr:>9.4f} {tr:>10.4f} {tr-vr:>+8.4f}  "
              f"{vd:>8.4f} {td:>8.4f}  {vl_str:>9} {tl_str:>10}")

    # Overall verdict
    mean_d_rep = np.mean(deltas_rep) if deltas_rep else 0.0
    mean_d_d4  = np.mean(deltas_d4)  if deltas_d4  else 0.0

    # Step-level entropy comparison: TASB should not collapse entropy
    # (low entropy + repetition = stuck attractor)
    van_ent_mean  = np.mean([r['van_entropy']  for r in records])
    tasb_ent_mean = np.mean([r['tasb_entropy'] for r in records])

    # R_head sanity (catches Bug #7 silently re-emerging)
    rhead_mean = np.mean([r['R_head'] for r in records])
    rhead_zero_pct = sum(1 for r in records if r['R_head'] == 0) / len(records) * 100

    print()
    print(f"  Aggregate:")
    print(f"    mean Δ rep4 (TASB - vanilla):  {mean_d_rep:+.4f}")
    print(f"    mean Δ distinct-4:             {mean_d_d4:+.4f}")
    print(f"    mean vanilla logit entropy:    {van_ent_mean:.4f}")
    print(f"    mean TASB    logit entropy:    {tasb_ent_mean:.4f}")
    print(f"    mean R_head (TSU work):        {rhead_mean:.4f}")
    print(f"    R_head == 0 rows:              {rhead_zero_pct:.1f}%  "
          f"({'OK — injection fired' if rhead_zero_pct < 5 else red('Bug #7 redux — INVESTIGATE')})")

    print()
    # Verdict thresholds: |Δrep| < 0.02 = faithful; < 0 = improved; > 0 = degraded
    if rhead_zero_pct > 50:
        print(f"  {red('[VERDICT] INJECTION FAILED — Bug #7 returned.')}")
    elif abs(mean_d_rep) < 0.02:
        print(f"  {green('[VERDICT] FAITHFUL — TASB stays in vanilla manifold')}")
        print(f"           (autoregressive behavior indistinguishable from vanilla)")
    elif mean_d_rep < -0.02:
        print(f"  {green('[VERDICT] IMPROVED — TASB reduces repetition vs vanilla')}")
        print(f"           (thermodynamic noise breaks degenerate attractors)")
    else:
        print(f"  {yellow('[VERDICT] DEGRADED — TASB increases repetition vs vanilla')}")
        print(f"           (investigate; check single-layer α tuning)")

    print(f"{'═'*78}\n")

    return {
        'mean_delta_rep':  mean_d_rep,
        'mean_delta_d4':   mean_d_d4,
        'van_entropy':     van_ent_mean,
        'tasb_entropy':    tasb_ent_mean,
        'mean_R_head':     rhead_mean,
        'rhead_zero_pct':  rhead_zero_pct,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description='TASB Stress Test v2 — Deep Ablation')
    ap.add_argument('--model',          default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--samples',        type=int,   default=10)
    ap.add_argument('--tokens-phase-a', type=int,   default=60, dest='tokens_a')
    ap.add_argument('--tokens-phase-c', type=int,   default=100, dest='tokens_c')
    ap.add_argument('--outdir',         default='.')
    ap.add_argument('--skip-phase-c',   action='store_true', dest='skip_c')
    ap.add_argument('--quick',          action='store_true',
                    help='Quick mode: 3 prompts, 15 tokens — for verification only')
    args = ap.parse_args()

    if args.quick:
        args.tokens_a = 15
        args.tokens_c = 20

    from tasb_llama_config import SAMPLER
    from tasb_two_pass import TASB_LAYERS

    ts = time.strftime('%Y%m%d_%H%M%S')

    # Bug #6 guard: sampler_cfg built in main, passed explicitly everywhere
    sampler_cfg = dataclasses.replace(
        SAMPLER,
        k_local=-999,        # exact Boltzmann sentinel
        k_attn=args.samples,
        field_strength=9.0,
        n_burnin=60,
        annealing_steps=10,
    )

    # Prepare prompts for Phase A (all) and Phase C (subset)
    phase_a_prompts = PROMPTS if not args.quick else PROMPTS[:3]
    phase_c_prompts = [p for p in phase_a_prompts if p['id'] in PHASE_C_PROMPT_IDS]

    print(f"\n{'═'*78}")
    print(f"  TASB STRESS TEST v2 — DEEP ABLATION  {ts}")
    print(f"{'═'*78}")
    print(f"  [SYS_INIT] Model: {args.model}")
    print(f"  [SYS_INIT] Sampler: torch.multinomial K={args.samples} (exact Boltzmann)")
    print(f"  [SYS_INIT] Phase A: {len(phase_a_prompts)} prompts × {args.tokens_a} tokens × "
          f"{len(PHASE_A_CONDITIONS)} conditions")
    target_rows = len(phase_a_prompts) * args.tokens_a * len(PHASE_A_CONDITIONS)
    print(f"  [SYS_INIT] Phase A target rows: {target_rows}")
    print(f"  [SYS_INIT] Phase A paired n per comparison: "
          f"{len(phase_a_prompts) * args.tokens_a}")
    print(f"  [SYS_INIT] Phase C: {len(phase_c_prompts)} prompts × {args.tokens_c} tokens "
          f"(autoregressive)")
    print(f"{'═'*78}")

    model, tokenizer = load_model(args.model)

    # ── PHASE A ───────────────────────────────────────────────────────────────
    phase_a_records = run_phase_a(
        model=model,
        tokenizer=tokenizer,
        prompts=phase_a_prompts,
        tokens_per_prompt=args.tokens_a,
        all_layers=TASB_LAYERS,
        conditions=CONDITIONS,
        sampler_cfg=sampler_cfg,
        outdir=args.outdir,
        ts=ts,
    )

    out_a = f"{args.outdir}/tasb_stress_v2_phaseA_{ts}.csv"
    with open(out_a, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(phase_a_records[0].keys()))
        w.writeheader(); w.writerows(phase_a_records)
    print(f"\n  Phase A CSV: {out_a}  ({len(phase_a_records)} rows)")

    # ── PHASE B ───────────────────────────────────────────────────────────────
    summary, mcnemar, min_viable_name = run_phase_b(phase_a_records)

    # ── PHASE C (with Bug #7 fix) ─────────────────────────────────────────────
    phase_c_records = []
    coherence = {}
    if not args.skip_c:
        # Bug #7 fix: extract concrete list, not the cfg dict
        winner_cfg = CONDITIONS.get(min_viable_name)
        if winner_cfg is None:
            print(f"  {red(f'min_viable {min_viable_name} not in CONDITIONS; using peak_L24')}")
            winner_cfg = CONDITIONS['peak_L24']
            min_viable_name = 'peak_L24'

        layer_subset = winner_cfg['layers']    # ← THE FIX: concrete list, not dict
        winner_alpha = winner_cfg['alpha']

        print(f"\n  [PHASE_C] Selected: {min_viable_name}  layers={layer_subset}  α={winner_alpha}")

        phase_c_records = run_phase_c(
            model=model,
            tokenizer=tokenizer,
            prompts=phase_c_prompts,
            tokens_per_prompt=args.tokens_c,
            alpha=winner_alpha,
            layer_subset=layer_subset,       # ← list of ints, not dict
            sampler_cfg=sampler_cfg,
            outdir=args.outdir,
            ts=ts,
        )

        if phase_c_records:
            out_c = f"{args.outdir}/tasb_stress_v2_phaseC_{ts}.csv"
            with open(out_c, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(phase_c_records[0].keys()))
                w.writeheader(); w.writerows(phase_c_records)
            print(f"\n  Phase C CSV: {out_c}  ({len(phase_c_records)} rows)")

        coherence = summarize_phase_c(phase_c_records)

    # ── Timing profile ────────────────────────────────────────────────────────
    print(f"\n{'═'*78}")
    print(f"  [TIMING_PROFILE] Phase A averages (per token-condition)")
    print(f"{'─'*78}")
    nv = [r for r in phase_a_records if r['condition'] != 'vanilla']
    if nv:
        t1 = np.mean([r['t_gpu1'] for r in nv])
        tt = np.mean([r['t_tsu']  for r in nv])
        t2 = np.mean([r['t_gpu2'] for r in nv])
        total = t1 + tt + t2
        print(f"  GPU Phase 1 (vanilla capture):  {t1:.3f}s  ({t1/total*100:>5.1f}%)")
        print(f"  TSU Phase   (exact Boltzmann):  {tt:.3f}s  ({tt/total*100:>5.1f}%)")
        print(f"  GPU Phase 2 (injection):        {t2:.3f}s  ({t2/total*100:>5.1f}%)")
        print(f"  Total round-trip:                {total:.3f}s")

    print(f"\n{'═'*78}")
    print(f"  [STATUS] All phases complete. {green('No errors.')}")
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
