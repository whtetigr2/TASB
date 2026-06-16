"""
tasb_m5_faithfulness.py — Stage 1 M5: matched-baseline faithfulness core test
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

This is the first measurement on the corrected (post-RoPE-fix) TASB stack.
Implements Codex's Milestone 5:

  "Does TSU perturbation preserve or predictably alter the next-token
   distribution relative to a matched baseline?"

WHAT IT MEASURES
----------------
For a battery of prompts, at each teacher-forced step, runs the bridge at
multiple α values and records per-step:

  Distributional (matched-baseline, primary):
    kl_logit:       KL(vanilla || bridge) — primary fidelity metric
    js_div:         Jensen-Shannon divergence (symmetric)

  Decoding-projection (secondary):
    top1_agree:     vanilla argmax == bridge argmax
    top5_agree:     bridge top-1 in vanilla top-5

  Confidence diagnostics:
    prob_gap:       vanilla top-1 prob − top-2 prob (position ambiguity)
    vanilla_entropy:    Shannon entropy of vanilla distribution (nats)

The "matched baseline" framing: vanilla and bridge see identical input at
every step (teacher-forced advance by vanilla's next token). Both run on the
same canonical captured object via bridge_forward. The differences arise
purely from the bridge's substitution at the target layer.

α SWEEP
-------
α ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}. Seven points characterize the
faithfulness curve from "no perturbation" to "pure p_thermo".

  α=0.0: bit-exact vanilla. Mean KL must be ~0. (sanity)
  α=0.3: production design point. Codex's M5 expects close to vanilla.
  α=1.0: pure p_thermo. Worst-case perturbation bound.

If KL grows smoothly and monotonically with α, the bridge "predictably
alters" the distribution. If it spikes or oscillates, the bridge is chaotic
in some α regime.

WHAT THIS FILE DOES NOT DO
--------------------------
- No autoregressive generation. Each step uses vanilla's next token to
  advance (teacher-forced). Free generation is M6.
- No top-p / temperature sampling on output. Raw logits → analytical
  distributions. Decoding-strategy effects are M6.
- No multi-layer injection. Stage 1 is one layer (L18 winner from v2).
- No backend sweep. Uses 'exact' only. Backend comparison was M4
  (test_sampler_v2 T4 already validated exact vs gumbel at K=5000).

BUG GUARDS
----------
- #1 args scope: all params explicit, no `args.X` inside inner functions.
- #3 fp32 eps: PATCH 2026-05-28 — KL/JS/entropy now use log_softmax with
  NO clamp. (Previously clamped to eps=1e-4 which, on a 128k-vocab,
  added ~12 units of artificial mass and suppressed KL by ~15-17×.
  The clamp is gone; see _kl_divergence below.)
- #6 params propagate: explicit through every call site.
- #7 layer_subset: bridge_forward already type-guards.
==============================================================================
"""

import argparse
import csv
import math
import os
import sys
import time
import zlib
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PATCH 2026-05-30 (post-Gemini-second-review, P2): reconfigure stdout to
# UTF-8 so box-drawing and Greek characters (═ α ✓) print correctly on
# Windows consoles using cp1252. No-op on Linux/macOS where UTF-8 is
# already default. Wrapped in try/except so older Pythons or unusual
# streams don't crash on this reconfigure call.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

from tasb_pipeline_v2 import bridge_forward


# ── Color helpers (TTY only) ──────────────────────────────────────────────
def _c(code, t):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def gray(t):   return _c("90", t)


# ── Configuration ─────────────────────────────────────────────────────────
LAYER_IDX = 18           # peak_L18 from v2 (carries as starting point)
BACKEND   = 'exact'      # M4-validated production backend
K_SAMPLES = 10           # production K (M4 showed convergence well past this)

# α sweep: 7 points characterize the perturbation curve
ALPHA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Standard 9-prompt battery (same as legacy v2 for direct comparability)
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


# ── Distributional metrics ────────────────────────────────────────────────
#
# PATCH 2026-05-28 (post-Gemini-review):
#   Old implementations clamped softmax outputs to eps=1e-4 before computing
#   KL/JS/entropy. With a 128k-token LLaMA vocab, the clamp adds ~12 units of
#   artificial mass (128000 * 1e-4) before renormalization. That mass is
#   identical in p and q, so KL was suppressed by ~15-17x in confident
#   distributions, contaminating the headline number.
#
#   New implementations use F.log_softmax which keeps small probabilities
#   stable in log space — no clamp, no renormalization, no contamination.
#
# Verified on synthetic LLaMA-shape distributions:
#   confident position:  buggy 1.23e-05  vs correct 2.05e-04  (16.6x off)
#   ambiguous position:  buggy 9.79e-07  vs correct 1.43e-05  (14.6x off)

import torch.nn.functional as F


def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) in nats, computed in log space (no clamp).

    p_logits, q_logits: 1D tensors of raw (pre-softmax) logits.
    """
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    # KL(p || q) = sum p * (log p - log q)
    kl = float((p * (log_p - log_q)).sum().item())
    return max(0.0, kl)   # tiny negatives are fp32 roundoff


def _js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Jensen-Shannon divergence (symmetric KL) in nats. Log-space, no clamp."""
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    # log(m) is stable since m = 0.5(p+q) and p,q sum to 1, so m >= 0
    log_m = (m + 1e-30).log()
    kl_pm = float((p * (log_p - log_m)).sum().item())
    kl_qm = float((q * (log_q - log_m)).sum().item())
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def _entropy(logits: torch.Tensor) -> float:
    """Shannon entropy in nats. Log-space, no clamp."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return float(-(p * log_p).sum().item())


def _prob_gap(logits: torch.Tensor) -> float:
    """Top-1 minus top-2 in PROBABILITY space. Range [0, 1].
    Renamed from _logit_gap (which was misleadingly named — it computed
    probability gap, not logit margin)."""
    p = torch.softmax(logits.float(), dim=-1)
    top2 = torch.topk(p, 2).values
    return float((top2[0] - top2[1]).item())


def _logit_margin(logits: torch.Tensor) -> float:
    """Top-1 minus top-2 in LOGIT space (the real logit margin)."""
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


def _top_probs(logits: torch.Tensor) -> tuple[float, float]:
    """Return (top1_prob, top2_prob) for the row."""
    p = torch.softmax(logits.float(), dim=-1)
    top2 = torch.topk(p, 2).values
    return float(top2[0].item()), float(top2[1].item())


# ── Single-token-step measurement ─────────────────────────────────────────

def measure_token_step(model, tok, input_ids: torch.Tensor,
                       alphas: list[float], layer_idx: int, K: int,
                       backend: str, seed: int) -> tuple[list[dict], int]:
    """Run one teacher-forced step at all α values.

    Returns (records, vanilla_next_token_id).
    Each record is one (α, metrics) measurement at this step.

    Uses the SAME seed across α values per step so the sampler draws the
    same p_thermo for the same captured object — α is the only thing varying.

    PATCH 2026-05-28 (post-Gemini-review):
      - α=0 is now MEASURED, not hardcoded. We compare bridge_forward's
        α=0 logits against its captured vanilla_logits per step and record
        the actual max_abs_diff. The alpha_zero_identity invariant is now
        regression-tested every step, not assumed.
      - Renamed logit_gap → prob_gap. Added logit_margin, top1_prob, top2_prob.
      - KL/JS now use log_softmax (no clamp). See _kl_divergence patch above.
    """
    # Run the α=0 path FIRST. bridge_forward at α=0 should produce bit-exact
    # vanilla logits (Stage 1 invariant #4, alpha_zero_identity). We measure
    # this per step rather than assuming it.
    base_result = bridge_forward(
        model=model, tok=None, input_ids=input_ids,
        layer_idx=layer_idx, alpha=0.0, backend=backend, K=K,
        seed=seed, return_intermediates=True)

    vanilla_logits_full = base_result.vanilla_logits[0]   # (S, vocab)
    bridge_alpha0_full  = base_result.logits[0]            # (S, vocab) at α=0
    vanilla_logits = vanilla_logits_full[-1, :]
    bridge_alpha0  = bridge_alpha0_full[-1, :]

    # α=0 identity regression test (per step)
    alpha0_max_abs_diff = float(
        (vanilla_logits.float() - bridge_alpha0.float()).abs().max().item())

    vanilla_top1 = int(vanilla_logits.argmax().item())
    vanilla_top5 = torch.topk(vanilla_logits, 5).indices.tolist()
    vanilla_prob_gap   = _prob_gap(vanilla_logits)
    vanilla_logit_marg = _logit_margin(vanilla_logits)
    vanilla_top1_p, vanilla_top2_p = _top_probs(vanilla_logits)
    vanilla_H = _entropy(vanilla_logits)

    # α=0 row — MEASURED, not assumed
    bridge_top1_a0 = int(bridge_alpha0.argmax().item())
    kl_a0 = _kl_divergence(vanilla_logits, bridge_alpha0)
    js_a0 = _js_divergence(vanilla_logits, bridge_alpha0)
    records = [{
        'alpha':               0.0,
        'kl_logit':            kl_a0,
        'js_div':              js_a0,
        'top1_agree':          int(bridge_top1_a0 == vanilla_top1),
        'top5_agree':          int(bridge_top1_a0 in vanilla_top5),
        'vanilla_top1':        vanilla_top1,
        'bridge_top1':         bridge_top1_a0,
        'prob_gap':            vanilla_prob_gap,
        'logit_margin':        vanilla_logit_marg,
        'top1_prob':           vanilla_top1_p,
        'top2_prob':           vanilla_top2_p,
        'vanilla_entropy':     vanilla_H,
        'alpha0_max_abs_diff': alpha0_max_abs_diff,
    }]

    # Now sweep over non-zero α
    for alpha in alphas:
        if alpha == 0.0:
            continue
        result = bridge_forward(
            model=model, tok=None, input_ids=input_ids,
            layer_idx=layer_idx, alpha=alpha, backend=backend, K=K,
            seed=seed, return_intermediates=True)
        bridge_logits = result.logits[0, -1, :]
        bridge_top1 = int(bridge_logits.argmax().item())
        kl = _kl_divergence(vanilla_logits, bridge_logits)
        js = _js_divergence(vanilla_logits, bridge_logits)
        records.append({
            'alpha':               alpha,
            # PATCH 2026-05-30 (post-Gemini-third-review, P3):
            # Store raw float precision. Rounding belongs at display time,
            # not write time — preserves headroom for downstream analysis.
            'kl_logit':            kl,
            'js_div':              js,
            'top1_agree':          int(bridge_top1 == vanilla_top1),
            'top5_agree':          int(bridge_top1 in vanilla_top5),
            'vanilla_top1':        vanilla_top1,
            'bridge_top1':         bridge_top1,
            'prob_gap':            vanilla_prob_gap,
            'logit_margin':        vanilla_logit_marg,
            'top1_prob':           vanilla_top1_p,
            'top2_prob':           vanilla_top2_p,
            'vanilla_entropy':     vanilla_H,
            'alpha0_max_abs_diff': alpha0_max_abs_diff,   # carries through
        })

    return records, vanilla_top1


# ── Main test loop ────────────────────────────────────────────────────────

def run_faithfulness(model, tok, prompts: list[dict],
                     tokens_per_prompt: int, alphas: list[float],
                     layer_idx: int, K: int, backend: str,
                     base_seed: int, outdir: str, ts: str) -> list[dict]:
    """Run the full faithfulness measurement.

    All params explicit (Bug #1, #6). Returns a list of records, each one
    a (prompt_id, step, alpha) measurement row.
    """
    print(f"\n{'═'*78}")
    print(f"  M5 FAITHFULNESS CORE TEST")
    print(f"  {len(prompts)} prompts × {tokens_per_prompt} tokens × "
          f"{len(alphas)} α values = "
          f"{len(prompts) * tokens_per_prompt * len(alphas)} rows")
    print(f"  Layer L{layer_idx}, backend={backend}, K={K}")
    print(f"  α sweep: {alphas}")
    print(f"{'═'*78}\n")

    all_records = []

    for prompt in prompts:
        inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
        ids = inputs['input_ids'].clone()

        print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]  "
              f"{gray(prompt['text'][:60])}")
        print(f"  {'─'*70}")

        for step in range(tokens_per_prompt):
            # PATCH 2026-05-30 (post-Gemini-second-review, P1):
            # Use zlib.crc32 (deterministic) instead of hash() (salted per
            # process via PYTHONHASHSEED). Previously --seed 42 did NOT
            # reproduce across separate python invocations because hash()
            # is process-salted by default. zlib.crc32 gives stable
            # SAMPLER SEEDS across invocations.
            #
            # Scope of reproducibility (post-Gemini-third-review, P3):
            # zlib.crc32 makes sampler draws deterministic given a fixed
            # base_seed. End-to-end CSV reproducibility requires the same
            # environment (PyTorch, CUDA, transformers, model quant, GPU).
            # Same machine + same env → numerically identical results.
            # Different machine or different env → same statistical pattern
            # but not bit-identical numbers. Do not claim "bit-identical
            # CSV across environments" externally.
            key = f"{prompt['id']}|{step}".encode('utf-8')
            seed = (base_seed + zlib.crc32(key)) & 0x7FFFFFFF

            records, van_id = measure_token_step(
                model=model, tok=tok, input_ids=ids,
                alphas=alphas, layer_idx=layer_idx, K=K,
                backend=backend, seed=seed)

            # Attach metadata
            for r in records:
                r['prompt_id'] = prompt['id']
                r['domain']    = prompt['domain']
                r['step']      = step + 1
                all_records.append(r)

            # Compact per-step log
            van_str = tok.decode([van_id]).strip()[:14]
            kl_strs = []
            for alpha in alphas:
                rec = next(r for r in records if r['alpha'] == alpha)
                tag = 'OK' if rec['top1_agree'] else 'X'
                color = green if rec['top1_agree'] else red
                # KL printed in scientific notation since true KL values are tiny
                kl_strs.append(color(f"α{alpha:.1f}={rec['kl_logit']:.2e}{tag}"))
            print(f"  step={step+1:>3}  van='{van_str:<10}'  " +
                  "  ".join(kl_strs), flush=True)

            # Teacher-forced advance by VANILLA token (matched baseline)
            ids = torch.cat(
                [ids, torch.tensor([[van_id]], device=model.device)],
                dim=1)
            if van_id == tok.eos_token_id:
                break

        # Per-prompt summary
        print(f"\n  {prompt['id']} summary:")
        for alpha in alphas:
            rows = [r for r in all_records
                    if r['prompt_id'] == prompt['id'] and r['alpha'] == alpha]
            if not rows:
                continue
            agree = np.mean([r['top1_agree'] for r in rows]) * 100
            kl    = np.mean([r['kl_logit']   for r in rows])
            print(f"    α={alpha:.1f}: top1={agree:>5.1f}%, "
                  f"mean_KL={kl:.6f}, n={len(rows)}")

        # Checkpoint
        ckpt = os.path.join(outdir, f"tasb_m5_faithfulness_partial_{ts}.csv")
        _write_csv(all_records, ckpt)
        print(f"  {gray(f'[checkpoint → {ckpt}]')}")

    return all_records


# ── Aggregate analysis ────────────────────────────────────────────────────

def _cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]],
                          n_boots: int = 5000,
                          seed: int = 42) -> tuple[float, float]:
    """Cluster-bootstrap on PROMPT MEANS (not pooled rows).

    PATCH 2026-05-28 (post-Gemini-review): rows within a prompt are
    correlated by the teacher-forced trajectory and shared context. Row-level
    bootstrap underestimates CI width. Cluster-bootstrap resamples whole
    prompts with replacement, then computes the statistic.

    PATCH 2026-05-30 (post-Gemini-second-review, P2/P3): now correctly
    computes per-prompt mean FIRST, then bootstraps the mean of those
    prompt means. Previous version pooled rows from sampled prompts and
    averaged the pool — which is statistically fine when prompts have
    equal length (current case: 60 tokens each), but biased when prompts
    have variable length (e.g. future EOS-terminated runs). Mean-of-means
    is the correct non-parametric estimator for clustered observations
    regardless of cluster size.
    """
    rng = np.random.default_rng(seed)
    clusters = list(values_by_cluster.keys())
    if not clusters:
        return 0.0, 0.0
    # Compute each cluster's mean once
    cluster_means = {k: np.mean(v) for k, v in values_by_cluster.items() if v}
    if not cluster_means:
        return 0.0, 0.0
    cluster_keys = list(cluster_means.keys())
    boots = []
    for _ in range(n_boots):
        sampled_keys = rng.choice(cluster_keys, len(cluster_keys), replace=True)
        # Mean of the resampled prompt means — each prompt weighted equally
        # regardless of its number of rows.
        boots.append(np.mean([cluster_means[k] for k in sampled_keys]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_ci(values: list[float], n_boots: int = 5000,
                  seed: int = 42) -> tuple[float, float]:
    """Row-level bootstrap (LEGACY — kept for backward compat / sanity check).

    POST-GEMINI-REVIEW NOTE: do NOT use this for headline numbers. Rows
    from the same prompt are correlated, so row-level bootstrap produces
    overly-narrow CIs. Use _cluster_bootstrap_ci for the real CIs and
    treat row-level results as exploratory only.
    """
    rng = np.random.default_rng(seed)
    if not values:
        return 0.0, 0.0
    boots = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(n_boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(records: list[dict], alphas: list[float]) -> dict:
    """Aggregate stats per α with cluster-bootstrap CIs by prompt.

    PATCH 2026-05-28 (post-Gemini-review): primary CIs are now cluster-
    bootstrap (resample whole prompts), since rows within a prompt are
    correlated. Row-level CIs are still computed and shown as 'rowCI' for
    comparison — but they are exploratory only.

    PATCH 2026-05-30 (post-Gemini-third-review, P2):
    Headline point estimates are now PROMPT-MEAN (matching the CI's estimand).
    Row-mean is shown alongside as a sanity check. For equal-length prompts
    these agree; for variable-length (e.g. future EOS-terminated runs) they
    differ, and prompt-mean is the right one to cite alongside cluster CIs.
    """
    print(f"\n{'═'*78}")
    print(f"  AGGREGATE FAITHFULNESS  (cluster estimator + CI = primary)")
    print(f"{'═'*78}\n")

    print(f"  {'α':>5} {'n':>5}  {'top1% (cluster)':>16} {'top1 95%CI':>16}  "
          f"{'top1% (row)':>12} {'top5%':>7}  "
          f"{'KL (cluster)':>12} {'KL 95%CI':>20}  {'KL (row)':>10}")
    print(f"  {'─'*135}")

    summary = {}
    for alpha in alphas:
        rows = [r for r in records if r['alpha'] == alpha]
        if not rows:
            continue
        agrees = [r['top1_agree'] for r in rows]
        top5s  = [r['top5_agree'] for r in rows]
        kls    = [r['kl_logit']   for r in rows]
        jss    = [r['js_div']     for r in rows]

        # Row-weighted (sanity) point estimates
        top1_row = np.mean(agrees) * 100
        top5_row = np.mean(top5s) * 100
        kl_row   = np.mean(kls)
        js_row   = np.mean(jss)

        # Group by prompt for cluster estimator + CI
        agrees_by_prompt = defaultdict(list)
        top5s_by_prompt  = defaultdict(list)
        kls_by_prompt    = defaultdict(list)
        jss_by_prompt    = defaultdict(list)
        for r in rows:
            agrees_by_prompt[r['prompt_id']].append(r['top1_agree'])
            top5s_by_prompt[r['prompt_id']].append(r['top5_agree'])
            kls_by_prompt[r['prompt_id']].append(r['kl_logit'])
            jss_by_prompt[r['prompt_id']].append(r['js_div'])

        # Cluster point estimates (mean of per-prompt means) — match the CI
        top1_cluster = float(np.mean([np.mean(v) for v in
                                      agrees_by_prompt.values()])) * 100
        top5_cluster = float(np.mean([np.mean(v) for v in
                                      top5s_by_prompt.values()])) * 100
        kl_cluster   = float(np.mean([np.mean(v) for v in
                                      kls_by_prompt.values()]))
        js_cluster   = float(np.mean([np.mean(v) for v in
                                      jss_by_prompt.values()]))

        # Cluster CIs (mean-of-prompt-means bootstrap)
        top1_clo, top1_chi = _cluster_bootstrap_ci(agrees_by_prompt)
        kl_clo, kl_chi     = _cluster_bootstrap_ci(kls_by_prompt)

        # Row-level CIs (exploratory only)
        top1_rlo, top1_rhi = _bootstrap_ci(agrees)
        kl_rlo, kl_rhi     = _bootstrap_ci(kls)

        # Color the headline (cluster) top1 pct
        top1_color = (green(f"{top1_cluster:>14.1f}%") if top1_cluster >= 95
                      else yellow(f"{top1_cluster:>14.1f}%") if top1_cluster >= 80
                      else red(f"{top1_cluster:>14.1f}%"))

        print(f"  {alpha:>5.1f} {len(rows):>5}  {top1_color}  "
              f"[{top1_clo*100:>5.1f}, {top1_chi*100:>5.1f}]  "
              f"{top1_row:>11.1f}%  "
              f"{top5_cluster:>6.1f}%  "
              f"{kl_cluster:>12.6f}  "
              f"[{kl_clo:>7.5f}, {kl_chi:>7.5f}]  "
              f"{kl_row:>10.6f}")

        summary[alpha] = {
            'n': len(rows),
            # Cluster (headline) point estimates — match the CI's estimand
            'top1_pct':         top1_cluster,
            'top5_pct':         top5_cluster,
            'mean_kl':          kl_cluster,
            'mean_js':          js_cluster,
            # Row-weighted (sanity) point estimates
            'top1_pct_row':     top1_row,
            'top5_pct_row':     top5_row,
            'mean_kl_row':      kl_row,
            'mean_js_row':      js_row,
            # CIs
            'top1_cluster_lo': top1_clo * 100, 'top1_cluster_hi': top1_chi * 100,
            'top1_row_lo':     top1_rlo * 100, 'top1_row_hi':     top1_rhi * 100,
            'kl_cluster_lo':   kl_clo,         'kl_cluster_hi':   kl_chi,
            'kl_row_lo':       kl_rlo,         'kl_row_hi':       kl_rhi,
        }

    # α=0 identity check (now measured, not assumed)
    # PATCH 2026-05-28 (post-Gemini-review): we now have alpha0_max_abs_diff
    # per row from the actual α=0 forward — use it.
    print()
    if 0.0 in summary:
        s = summary[0.0]
        a0_diffs = [r['alpha0_max_abs_diff'] for r in records
                    if r['alpha'] == 0.0]
        worst_diff = max(a0_diffs) if a0_diffs else float('inf')
        # Use cluster estimates for the sanity check (matches headline)
        if (s['top1_pct'] == 100.0
                and s['mean_kl'] < 1e-7
                and worst_diff < 1e-5):
            print(green(f"  ✓ α=0 identity: top1=100%, mean_KL={s['mean_kl']:.2e}, "
                        f"worst per-step max_abs_diff={worst_diff:.2e}"))
            print(green(f"    (alpha_zero_identity invariant holds end-to-end, "
                        f"measured per step)"))
        else:
            print(red(f"  ✗ α=0 identity FAILED:"))
            print(red(f"    top1={s['top1_pct']:.1f}%, "
                      f"mean_KL={s['mean_kl']:.2e}, "
                      f"worst per-step max_abs_diff={worst_diff:.2e}"))
            print(red(f"    The alpha_zero_identity invariant broke. Investigate."))

    # Monotonicity check on mean KL (using cluster estimates — headline)
    nonzero_alphas = sorted([a for a in summary if a > 0])
    kls_in_order = [summary[a]['mean_kl'] for a in nonzero_alphas]
    monotonic = all(kls_in_order[i] <= kls_in_order[i+1] + 1e-7
                    for i in range(len(kls_in_order)-1))
    if monotonic:
        print(green(f"  ✓ KL monotonic in α: "
                    f"{[f'{k:.6f}' for k in kls_in_order]}"))
    else:
        print(yellow(f"  ~ KL not strictly monotonic: "
                     f"{[f'{k:.6f}' for k in kls_in_order]}"))
        print(yellow(f"    (small inversions can be Monte Carlo noise at K=10; "
                     f"investigate if large)"))

    # Production design point readout — cluster estimator + cluster CI
    if 0.3 in summary:
        s = summary[0.3]
        print(f"\n  {bold('Production design point (α=0.3):')}")
        print(f"    Top-1 agreement: {s['top1_pct']:.1f}% (cluster) "
              f"cluster-CI [{s['top1_cluster_lo']:.1f}, {s['top1_cluster_hi']:.1f}]")
        print(f"                     {s['top1_pct_row']:.1f}% (row)     "
              f"row-CI    [{s['top1_row_lo']:.1f}, {s['top1_row_hi']:.1f}] "
              f"— exploratory only")
        print(f"    Mean KL:         {s['mean_kl']:.6f} (cluster) "
              f"cluster-CI [{s['kl_cluster_lo']:.6f}, {s['kl_cluster_hi']:.6f}] nats/token")
        print(f"                     {s['mean_kl_row']:.6f} (row)")

    return summary


# ── CSV I/O ───────────────────────────────────────────────────────────────

def _write_csv(records: list[dict], path: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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
    print(f"[SYS_INIT] Ready on {next(mdl.parameters()).device}\n",
          flush=True)
    return mdl, tok


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='TASB M5 Faithfulness Core Test')
    ap.add_argument('--model',  default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--tokens', type=int, default=60,
                    help='Tokens per prompt (60 full, 15 quick)')
    ap.add_argument('--outdir', default='results',
                    help='Output directory for CSV')
    ap.add_argument('--quick',  action='store_true',
                    help='Quick mode: 3 prompts × 15 tokens (~3 min)')
    ap.add_argument('--seed',   type=int, default=42,
                    help='Base seed for reproducible sampling')
    args = ap.parse_args()

    if args.quick:
        args.tokens = 15
        prompts = PROMPTS[:3]
    else:
        prompts = PROMPTS

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(f"  TASB M5 FAITHFULNESS  {ts}")
    print(f"{'═'*78}")
    print(f"  [CFG] Model:    {args.model}")
    print(f"  [CFG] Layer:    L{LAYER_IDX}  backend={BACKEND}  K={K_SAMPLES}")
    print(f"  [CFG] α sweep:  {ALPHA_SWEEP}")
    print(f"  [CFG] Prompts:  {len(prompts)} × {args.tokens} tokens")
    print(f"  [CFG] Rows:     {len(prompts) * args.tokens * len(ALPHA_SWEEP)}")
    print(f"  [CFG] Seed:     {args.seed}")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    t_start = time.perf_counter()
    records = run_faithfulness(
        model=model, tok=tok,
        prompts=prompts, tokens_per_prompt=args.tokens,
        alphas=ALPHA_SWEEP,
        layer_idx=LAYER_IDX, K=K_SAMPLES, backend=BACKEND,
        base_seed=args.seed,
        outdir=args.outdir, ts=ts)
    t_elapsed = time.perf_counter() - t_start

    out_csv = os.path.join(args.outdir, f"tasb_m5_faithfulness_{ts}.csv")
    _write_csv(records, out_csv)
    print(f"\n  CSV: {out_csv}  ({len(records)} rows)")

    analyze(records, ALPHA_SWEEP)

    print(f"\n{'═'*78}")
    print(f"  M5 COMPLETE  ({t_elapsed:.1f}s = {t_elapsed/60:.1f}min)")
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
