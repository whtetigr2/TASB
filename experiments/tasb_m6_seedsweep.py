"""
tasb_m6_seedsweep.py — M6 seed sweep diagnostic
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Three diagnostic tests to disentangle THREE HYPOTHESES about M6 free-run
divergence:

  H1: top-p sampling is the chaos amplifier
      → unmodified vanilla run twice with different RNG seeds also diverges
        fast under top-p=0.9, temp=0.8

  H2: the bridge adds nontrivial extra perturbation
      → bridge-vs-vanilla diverges MORE than vanilla-vs-vanilla

  H3: the bridge causes degenerate loops specifically
      → bridge produces prompt-echo collapse that vanilla doesn't, even with
        bad RNG seeds

TESTS
-----
Test 1: VANILLA SEED SWEEP — establishes the baseline of stochastic chaos.
  3 prompts × 5 seeds × 40 tokens, NO bridge. All pairwise comparisons.

Test 2: BRIDGE SEED SWEEP @ α=0.3 — characterizes bridge variance at the
  M6 production design point.
  3 prompts × 5 seeds × 40 tokens, bridge α=0.3. All pairwise comparisons.

Test 3: ALPHA DOSE-RESPONSE — fixed RNG seed, sweep α to isolate the
  bridge's contribution from RNG noise.
  3 prompts × 1 seed × 6 α values × 40 tokens, paired against α=0 vanilla.

OUTPUT
------
1. CSV: tasb_m6_seedsweep_<ts>.csv (per-step metrics for every run)
2. Summary CSV: tasb_m6_seedsweep_summary_<ts>.csv (one row per pairing)
3. Markdown: tasb_m6_seedsweep_<ts>.md (side-by-side comparison)

USAGE
-----
    python tasb_m6_seedsweep.py             # full diagnostic, ~10-15 min
    python tasb_m6_seedsweep.py --quick     # 2 prompts × 3 seeds, ~5 min
==============================================================================
"""

import argparse
import csv
import os
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field

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
BACKEND   = 'exact'
K_SAMPLES = 10
TOP_P       = 0.9
TEMPERATURE = 0.8

# Five seeds for variance characterization in Tests 1 & 2
RNG_SEEDS_FULL = [42, 137, 271, 314, 1729]   # 5 seeds
RNG_SEEDS_QUICK = [42, 137, 271]              # 3 seeds for --quick

# Test 3: dose-response α values
ALPHA_DOSE_RESPONSE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# Test 2: fixed α for the bridge-only seed sweep
ALPHA_BRIDGE_SWEEP = 0.3

# Prompts: M5 carryover + fresh, picked to span the regime space
PROMPTS_FULL = [
    {"id": "M5_HC1", "domain": "FACTUAL",
     "text": "The capital of France is"},
    {"id": "FR_RS1", "domain": "REASONING",
     "text": "Explain why a compass points north in simple terms for a curious teenager."},
    {"id": "FR_DI1", "domain": "DIALOG",
     "text": "Two engineers are arguing over coffee about whether machines can be "
             "creative. Write the conversation."},
]
PROMPTS_QUICK = PROMPTS_FULL[:2]


# ── Metrics (carried from M6: log-space, no clamp) ────────────────────────

def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    return max(0.0, float((p * (log_p - log_q)).sum().item()))


def _entropy(logits: torch.Tensor) -> float:
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return float(-(p * log_p).sum().item())


def _ngram_repeat_rate(tokens: list[int], n: int) -> float:
    if len(tokens) < n + 1:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        total += 1
        if ng in seen:
            repeats += 1
        seen.add(ng)
    return repeats / total if total else 0.0


def _consecutive_repeat_rate(tokens: list[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    return sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1]) / (len(tokens) - 1)


def detect_loop(tokens: list[int]) -> tuple[bool, str]:
    """Multi-scale loop detection. Returns (is_looped, type_label).
    Thresholds carried from M5 recut."""
    if len(tokens) < 8:
        return False, "too_short"
    consec = _consecutive_repeat_rate(tokens) * 100
    g4     = _ngram_repeat_rate(tokens, 4) * 100
    g8     = _ngram_repeat_rate(tokens, 8) * 100
    if consec >= 40.0:
        return True, "consec_loop"
    if g4 >= 60.0 or g8 >= 50.0:
        return True, "cycle_loop"
    if g4 >= 30.0 or g8 >= 25.0:
        return False, "partial_cycling"
    return False, "varied"


# ── Seeded top-p sampling ─────────────────────────────────────────────────

def _seeded_uniform(rng_seed: int, prompt_id: str, step: int) -> float:
    """Deterministic Uniform[0,1) — unlike M6's CRN this is keyed by
    (rng_seed, prompt_id, step) so different rng_seeds produce different
    draw sequences. This is the variance source for Tests 1 and 2.
    """
    key = f"toppick|{rng_seed}|{prompt_id}|{step}".encode('utf-8')
    crc = zlib.crc32(key)
    return (crc & 0xFFFFFFFF) / float(1 << 32)


def _top_p_sample(logits: torch.Tensor, top_p: float, temperature: float,
                  u: float) -> int:
    """Top-p sampling with externally supplied uniform u. Same impl as M6."""
    scaled = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum_p = torch.cumsum(sorted_p, dim=-1)
    nucleus_mask = cum_p <= top_p
    nucleus_mask[0] = True
    first_exceed = (cum_p >= top_p).nonzero()
    if len(first_exceed) > 0:
        nucleus_mask[first_exceed[0].item()] = True
    nucleus_p = sorted_p * nucleus_mask.float()
    nucleus_p = nucleus_p / nucleus_p.sum()
    nucleus_cum = torch.cumsum(nucleus_p, dim=-1)
    pick_in_sorted = int((nucleus_cum >= u).nonzero()[0].item())
    return int(sorted_idx[pick_in_sorted].item())


# ── Single trajectory generation ──────────────────────────────────────────

@dataclass
class Trajectory:
    """A single generation trajectory.

    Fields:
        run_id: human-readable run identifier (e.g. "vanilla_s42" or "bridge_a0.3_s137")
        rng_seed: RNG seed used for top-p
        alpha: 0.0 for vanilla, else bridge α
        sampler_seed_base: base seed for the bridge sampler (only meaningful when alpha > 0)
        prompt_id: which prompt
        tokens: generated token IDs
        per_step: per-step diagnostics
        eos_step: -1 if no EOS, else 1-indexed step where EOS emitted
        loop_flag, loop_type: from detect_loop()
    """
    run_id: str
    rng_seed: int
    alpha: float
    sampler_seed_base: int
    prompt_id: str
    tokens: list[int] = field(default_factory=list)
    per_step: list[dict] = field(default_factory=list)
    eos_step: int = -1
    loop_flag: bool = False
    loop_type: str = ""


def generate_trajectory(model, tok, prompt: dict, n_tokens: int,
                        alpha: float, rng_seed: int,
                        sampler_seed_base: int) -> Trajectory:
    """Generate one trajectory. alpha=0 → pure vanilla; alpha>0 → bridge."""
    run_id = (f"vanilla_s{rng_seed}" if alpha == 0.0
              else f"bridge_a{alpha}_s{rng_seed}")
    traj = Trajectory(
        run_id=run_id, rng_seed=rng_seed, alpha=alpha,
        sampler_seed_base=sampler_seed_base, prompt_id=prompt['id'])

    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(n_tokens):
        u_t = _seeded_uniform(rng_seed, prompt['id'], step)
        sampler_seed = (sampler_seed_base + zlib.crc32(
            f"sampler|{rng_seed}|{prompt['id']}|{step}".encode('utf-8'))
            ) & 0x7FFFFFFF

        if alpha == 0.0:
            # Pure vanilla path — use bridge_forward(alpha=0) which gives
            # bit-exact vanilla logits per M5 invariant
            result = bridge_forward(
                model=model, tok=None, input_ids=ids,
                layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND, K=K_SAMPLES,
                seed=sampler_seed, return_intermediates=True)
            step_logits = result.vanilla_logits[0, -1, :]
        else:
            result = bridge_forward(
                model=model, tok=None, input_ids=ids,
                layer_idx=LAYER_IDX, alpha=alpha, backend=BACKEND,
                K=K_SAMPLES, seed=sampler_seed, return_intermediates=True)
            step_logits = result.logits[0, -1, :]

        token = _top_p_sample(step_logits, TOP_P, TEMPERATURE, u_t)
        H = _entropy(step_logits)
        top1 = int(step_logits.argmax().item())
        top1_prob = float(torch.softmax(
            step_logits.float(), dim=-1)[top1].item())

        traj.tokens.append(token)
        traj.per_step.append({
            'step':       step + 1,
            'u_t':        u_t,
            'token':      token,
            'top1':       top1,
            'top1_prob':  top1_prob,
            'entropy':    H,
        })

        if token == tok.eos_token_id:
            traj.eos_step = step + 1
            break

        ids = torch.cat(
            [ids, torch.tensor([[token]], device=model.device)], dim=1)

    traj.loop_flag, traj.loop_type = detect_loop(traj.tokens)
    return traj


# ── Pair comparison (post-hoc, no extra compute) ──────────────────────────

@dataclass
class PairComparison:
    """Compare two trajectories: divergence step, agreement %, etc."""
    label_a: str
    label_b: str
    prompt_id: str
    alpha_a: float
    alpha_b: float
    seed_a: int
    seed_b: int
    divergence_step: int = -1     # first step where tokens differ; -1 if never
    n_compared: int = 0           # min(len(a), len(b))
    n_agree: int = 0
    agree_pct: float = 0.0
    pre_div_n: int = 0            # steps before divergence
    post_div_n: int = 0           # steps after divergence (within n_compared)
    loop_a: bool = False
    loop_b: bool = False


def compare_pair(a: Trajectory, b: Trajectory) -> PairComparison:
    """Token-level comparison of two trajectories."""
    n = min(len(a.tokens), len(b.tokens))
    pair = PairComparison(
        label_a=a.run_id, label_b=b.run_id,
        prompt_id=a.prompt_id,
        alpha_a=a.alpha, alpha_b=b.alpha,
        seed_a=a.rng_seed, seed_b=b.rng_seed,
        n_compared=n,
        loop_a=a.loop_flag, loop_b=b.loop_flag,
    )
    agree = 0
    for i in range(n):
        if a.tokens[i] == b.tokens[i]:
            agree += 1
        elif pair.divergence_step == -1:
            pair.divergence_step = i + 1
    pair.n_agree = agree
    pair.agree_pct = 100.0 * agree / n if n > 0 else 0.0
    if pair.divergence_step > 0:
        pair.pre_div_n = pair.divergence_step - 1
        pair.post_div_n = n - (pair.divergence_step - 1)
    else:
        pair.pre_div_n = n
        pair.post_div_n = 0
    return pair


# ── Test 1: Vanilla seed sweep ────────────────────────────────────────────

def run_test1_vanilla_baseline(model, tok, prompts, seeds, n_tokens):
    """Vanilla-vs-vanilla pairwise comparison across RNG seeds.
    Establishes the BASELINE of stochastic chaos under top-p alone."""
    print(f"\n{bold('═══ TEST 1: VANILLA SEED SWEEP (baseline chaos) ═══')}")
    print(f"  {len(prompts)} prompts × {len(seeds)} seeds, vanilla only")
    print(f"  Pairwise: {len(seeds)*(len(seeds)-1)//2} pairs per prompt\n")

    trajectories = {}  # (prompt_id, seed) -> Trajectory
    for prompt in prompts:
        print(f"  {cyan(prompt['id'])}: ", end='', flush=True)
        for seed in seeds:
            t0 = time.perf_counter()
            traj = generate_trajectory(
                model, tok, prompt, n_tokens,
                alpha=0.0, rng_seed=seed, sampler_seed_base=seed)
            elapsed = time.perf_counter() - t0
            trajectories[(prompt['id'], seed)] = traj
            loop_marker = red("L") if traj.loop_flag else "."
            print(f"s{seed}({elapsed:.0f}s,{loop_marker})", end=' ', flush=True)
        print()

    # All pairwise comparisons within each prompt
    pairs = []
    for prompt in prompts:
        for i, sa in enumerate(seeds):
            for sb in seeds[i+1:]:
                a = trajectories[(prompt['id'], sa)]
                b = trajectories[(prompt['id'], sb)]
                pairs.append(compare_pair(a, b))
    return trajectories, pairs


# ── Test 2: Bridge seed sweep at fixed α ──────────────────────────────────

def run_test2_bridge_sweep(model, tok, prompts, seeds, n_tokens,
                           alpha=ALPHA_BRIDGE_SWEEP):
    """Bridge-vs-bridge pairwise comparison across RNG seeds at fixed α.
    Characterizes bridge variance at the production design point."""
    print(f"\n{bold(f'═══ TEST 2: BRIDGE SEED SWEEP @ α={alpha} ═══')}")
    print(f"  {len(prompts)} prompts × {len(seeds)} seeds, bridge α={alpha}")
    print(f"  Pairwise: {len(seeds)*(len(seeds)-1)//2} pairs per prompt\n")

    trajectories = {}
    for prompt in prompts:
        print(f"  {cyan(prompt['id'])}: ", end='', flush=True)
        for seed in seeds:
            t0 = time.perf_counter()
            traj = generate_trajectory(
                model, tok, prompt, n_tokens,
                alpha=alpha, rng_seed=seed, sampler_seed_base=seed)
            elapsed = time.perf_counter() - t0
            trajectories[(prompt['id'], seed)] = traj
            loop_marker = red("L") if traj.loop_flag else "."
            print(f"s{seed}({elapsed:.0f}s,{loop_marker})", end=' ', flush=True)
        print()

    pairs = []
    for prompt in prompts:
        for i, sa in enumerate(seeds):
            for sb in seeds[i+1:]:
                a = trajectories[(prompt['id'], sa)]
                b = trajectories[(prompt['id'], sb)]
                pairs.append(compare_pair(a, b))
    return trajectories, pairs


# ── Test 3: α dose-response at fixed seed ─────────────────────────────────

def run_test3_alpha_doseresponse(model, tok, prompts, alphas, fixed_seed,
                                 n_tokens):
    """Sweep α with a single fixed RNG seed, then compare each α run against
    the α=0 vanilla baseline at the same seed. Holds RNG variable constant
    so the divergence is purely the bridge's contribution."""
    print(f"\n{bold('═══ TEST 3: ALPHA DOSE-RESPONSE (fixed seed) ═══')}")
    print(f"  {len(prompts)} prompts × {len(alphas)} α values, fixed seed={fixed_seed}")
    print(f"  Each α compared against α=0 baseline at the same seed\n")

    trajectories = {}
    for prompt in prompts:
        print(f"  {cyan(prompt['id'])}: ", end='', flush=True)
        for alpha in alphas:
            t0 = time.perf_counter()
            traj = generate_trajectory(
                model, tok, prompt, n_tokens,
                alpha=alpha, rng_seed=fixed_seed, sampler_seed_base=fixed_seed)
            elapsed = time.perf_counter() - t0
            trajectories[(prompt['id'], alpha)] = traj
            loop_marker = red("L") if traj.loop_flag else "."
            print(f"α{alpha}({elapsed:.0f}s,{loop_marker})", end=' ', flush=True)
        print()

    # Each α compared against the α=0 baseline of the same prompt
    pairs = []
    for prompt in prompts:
        baseline = trajectories[(prompt['id'], 0.0)]
        for alpha in alphas:
            if alpha == 0.0:
                continue
            run = trajectories[(prompt['id'], alpha)]
            pairs.append(compare_pair(baseline, run))
    return trajectories, pairs


# ── Summary reports ───────────────────────────────────────────────────────

def summarize_pairs(pairs: list[PairComparison], label: str):
    """Aggregate pairwise stats grouped by prompt."""
    print(f"\n{bold(f'── {label} — pair summary ──')}")
    print(f"  {'prompt':<10}  {'pairs':>5}  {'mean div@':>10}  "
          f"{'min div@':>10}  {'mean agree%':>12}  {'loop_rate':>10}")
    print(f"  {'─'*70}")
    by_prompt = defaultdict(list)
    for p in pairs:
        by_prompt[p.prompt_id].append(p)
    for pid in sorted(by_prompt.keys()):
        ps = by_prompt[pid]
        # divergence step: treat "never" as n_compared (the upper bound)
        div_steps = [p.divergence_step if p.divergence_step > 0 else p.n_compared
                     for p in ps]
        mean_div = np.mean(div_steps)
        min_div  = min(div_steps)
        mean_agree = np.mean([p.agree_pct for p in ps])
        # loop rate: any trajectory in any pair that looped
        looped = sum(1 for p in ps if p.loop_a) + sum(1 for p in ps if p.loop_b)
        total_trajs = 2 * len(ps)
        loop_rate = 100.0 * looped / total_trajs if total_trajs else 0.0
        print(f"  {pid:<10}  {len(ps):>5}  {mean_div:>10.1f}  "
              f"{min_div:>10}  {mean_agree:>11.1f}%  {loop_rate:>9.1f}%")


def summarize_dose_response(pairs: list[PairComparison]):
    """Special summary for Test 3: show α effect at fixed seed."""
    print(f"\n{bold('── Test 3 — α effect at fixed seed (each compared to α=0) ──')}")
    print(f"  {'prompt':<10}  {'α':>5}  {'div@step':>10}  "
          f"{'agree%':>8}  {'loop':>6}")
    print(f"  {'─'*55}")
    by_prompt = defaultdict(list)
    for p in pairs:
        by_prompt[p.prompt_id].append(p)
    for pid in sorted(by_prompt.keys()):
        for p in sorted(by_prompt[pid], key=lambda x: x.alpha_b):
            div_s = str(p.divergence_step) if p.divergence_step > 0 else "never"
            loop_s = red("yes") if p.loop_b else "no"
            print(f"  {pid:<10}  {p.alpha_b:>5.2f}  {div_s:>10}  "
                  f"{p.agree_pct:>6.1f}%  {loop_s:>6}")


def write_csvs(test1_trajs, test1_pairs,
               test2_trajs, test2_pairs,
               test3_trajs, test3_pairs,
               outdir, ts):
    """Write per-step CSV + pair summary CSV."""
    # Per-step CSV (all trajectories from all tests)
    rows = []
    for test_label, trajs in [('test1_vanilla', test1_trajs),
                              ('test2_bridge', test2_trajs),
                              ('test3_dose',   test3_trajs)]:
        for key, traj in trajs.items():
            for s in traj.per_step:
                rows.append({
                    'test': test_label,
                    'run_id': traj.run_id,
                    'prompt_id': traj.prompt_id,
                    'alpha': traj.alpha,
                    'rng_seed': traj.rng_seed,
                    'loop_flag': int(traj.loop_flag),
                    'loop_type': traj.loop_type,
                    'eos_step': traj.eos_step,
                    **s,
                })
    csv_path = os.path.join(outdir, f"tasb_m6_seedsweep_{ts}.csv")
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Pair-summary CSV (one row per comparison)
    pair_rows = []
    for label, pairs in [('test1_vanilla', test1_pairs),
                         ('test2_bridge', test2_pairs),
                         ('test3_dose',   test3_pairs)]:
        for p in pairs:
            pair_rows.append({
                'test': label,
                'prompt_id': p.prompt_id,
                'label_a': p.label_a, 'label_b': p.label_b,
                'alpha_a': p.alpha_a, 'alpha_b': p.alpha_b,
                'seed_a': p.seed_a,   'seed_b': p.seed_b,
                'n_compared': p.n_compared,
                'n_agree': p.n_agree,
                'agree_pct': p.agree_pct,
                'divergence_step': p.divergence_step,
                'pre_div_n': p.pre_div_n,
                'post_div_n': p.post_div_n,
                'loop_a': int(p.loop_a),
                'loop_b': int(p.loop_b),
            })
    summary_csv = os.path.join(outdir, f"tasb_m6_seedsweep_summary_{ts}.csv")
    if pair_rows:
        fields = list(pair_rows[0].keys())
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(pair_rows)

    return csv_path, summary_csv


def write_markdown_report(test1_trajs, test2_trajs, test3_trajs,
                          tok, path: str):
    """Side-by-side markdown report for human review."""
    with open(path, 'w') as f:
        f.write("# M6 Seed Sweep Diagnostic\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Sampling:** top_p={TOP_P}, temperature={TEMPERATURE}\n\n")
        f.write(f"**Bridge:** L{LAYER_IDX}, backend={BACKEND}, K={K_SAMPLES}\n\n")
        f.write("Three tests to disentangle top-p chaos from bridge perturbation.\n\n")

        for test_label, trajs, header in [
            ('Test 1', test1_trajs, '## Test 1 — Vanilla seed sweep (baseline)'),
            ('Test 2', test2_trajs, f'## Test 2 — Bridge α={ALPHA_BRIDGE_SWEEP} seed sweep'),
            ('Test 3', test3_trajs, '## Test 3 — α dose-response (fixed seed)'),
        ]:
            f.write(f"\n{header}\n\n")
            # Group by prompt
            by_prompt = defaultdict(list)
            for key, traj in trajs.items():
                by_prompt[traj.prompt_id].append(traj)
            for pid in sorted(by_prompt.keys()):
                f.write(f"\n### {pid}\n\n")
                for traj in sorted(by_prompt[pid],
                                   key=lambda t: (t.alpha, t.rng_seed)):
                    loop_marker = " 🔁 LOOPED" if traj.loop_flag else ""
                    f.write(f"**{traj.run_id}** "
                            f"(α={traj.alpha}, seed={traj.rng_seed}, "
                            f"{len(traj.tokens)} tokens"
                            f"{', EOS@'+str(traj.eos_step) if traj.eos_step > 0 else ''}"
                            f"{loop_marker}):\n\n")
                    text = tok.decode(traj.tokens)
                    f.write(f"```\n{text}\n```\n\n")


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
    ap.add_argument('--fixed-seed', type=int, default=42,
                    help='Seed used for Test 3 (α dose-response)')
    args = ap.parse_args()

    if args.quick:
        prompts = PROMPTS_QUICK
        seeds   = RNG_SEEDS_QUICK
    else:
        prompts = PROMPTS_FULL
        seeds   = RNG_SEEDS_FULL

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M6 SEED SWEEP DIAGNOSTIC  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:        {args.model}")
    print(f"  Layer:        L{LAYER_IDX}, backend={BACKEND}, K={K_SAMPLES}")
    print(f"  Sampling:     top_p={TOP_P}, temperature={TEMPERATURE}")
    print(f"  Tokens/run:   {args.tokens}")
    print(f"  Prompts:      {len(prompts)} ({[p['id'] for p in prompts]})")
    print(f"  Seeds:        {seeds}")
    print(f"  Test 1:       vanilla seed sweep ({len(seeds)*(len(seeds)-1)//2} pairs/prompt)")
    print(f"  Test 2:       bridge α={ALPHA_BRIDGE_SWEEP} seed sweep ({len(seeds)*(len(seeds)-1)//2} pairs/prompt)")
    print(f"  Test 3:       α dose-response @ seed={args.fixed_seed}, "
          f"α∈{ALPHA_DOSE_RESPONSE}")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    t_start = time.perf_counter()

    test1_trajs, test1_pairs = run_test1_vanilla_baseline(
        model, tok, prompts, seeds, args.tokens)
    test2_trajs, test2_pairs = run_test2_bridge_sweep(
        model, tok, prompts, seeds, args.tokens, alpha=ALPHA_BRIDGE_SWEEP)
    test3_trajs, test3_pairs = run_test3_alpha_doseresponse(
        model, tok, prompts, ALPHA_DOSE_RESPONSE,
        args.fixed_seed, args.tokens)

    elapsed = time.perf_counter() - t_start

    # ── Summaries ──────────────────────────────────────────────────────
    summarize_pairs(test1_pairs, "Test 1: vanilla baseline chaos")
    summarize_pairs(test2_pairs, f"Test 2: bridge α={ALPHA_BRIDGE_SWEEP} variance")
    summarize_dose_response(test3_pairs)

    # Direct comparison — the key answer
    print(f"\n{'═'*78}")
    print(bold("  KEY COMPARISON: bridge variance vs vanilla baseline"))
    print(f"{'═'*78}")
    print(f"  If Test 2 numbers are CLOSE to Test 1, the bridge is mostly")
    print(f"  riding the top-p chaos floor.")
    print(f"  If Test 2 numbers are MUCH WORSE, the bridge adds nontrivial")
    print(f"  extra perturbation that needs explanation.\n")
    print(f"  {'prompt':<10}  {'metric':<22}  "
          f"{'Test 1 (vanilla)':>16}  {'Test 2 (bridge α=0.3)':>22}")
    print(f"  {'─'*78}")
    t1_by_prompt = defaultdict(list)
    t2_by_prompt = defaultdict(list)
    for p in test1_pairs: t1_by_prompt[p.prompt_id].append(p)
    for p in test2_pairs: t2_by_prompt[p.prompt_id].append(p)
    for pid in sorted(t1_by_prompt.keys()):
        t1 = t1_by_prompt[pid]
        t2 = t2_by_prompt[pid]
        t1_div = np.mean([p.divergence_step if p.divergence_step > 0
                          else p.n_compared for p in t1])
        t2_div = np.mean([p.divergence_step if p.divergence_step > 0
                          else p.n_compared for p in t2])
        t1_agree = np.mean([p.agree_pct for p in t1])
        t2_agree = np.mean([p.agree_pct for p in t2])
        t1_loop = sum(1 for p in t1 if p.loop_a or p.loop_b) / max(len(t1), 1)
        t2_loop = sum(1 for p in t2 if p.loop_a or p.loop_b) / max(len(t2), 1)
        print(f"  {pid:<10}  {'mean div@step':<22}  "
              f"{t1_div:>16.1f}  {t2_div:>22.1f}")
        print(f"  {'':<10}  {'mean agreement %':<22}  "
              f"{t1_agree:>15.1f}%  {t2_agree:>21.1f}%")
        print(f"  {'':<10}  {'pair-loop rate %':<22}  "
              f"{t1_loop*100:>15.1f}%  {t2_loop*100:>21.1f}%")
        print()

    # ── Write outputs ──────────────────────────────────────────────────
    csv_path, summary_csv = write_csvs(
        test1_trajs, test1_pairs, test2_trajs, test2_pairs,
        test3_trajs, test3_pairs, args.outdir, ts)
    md_path = os.path.join(args.outdir, f"tasb_m6_seedsweep_{ts}.md")
    write_markdown_report(test1_trajs, test2_trajs, test3_trajs, tok, md_path)
    print(f"\n  Per-step CSV:   {csv_path}")
    print(f"  Pair-summary CSV: {summary_csv}")
    print(f"  Markdown report:  {md_path}")

    print(f"\n{'═'*78}")
    print(bold(f"  SEED SWEEP COMPLETE  ({elapsed:.1f}s = {elapsed/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
