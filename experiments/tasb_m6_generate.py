"""
tasb_m6_generate.py — M6: production-realism extension
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

After M5 closed (per-step faithfulness on matched, teacher-forced contexts),
M6 answers Codex's next question:

  "When TASB is allowed to generate freely, does the whole trajectory remain
   useful / coherent / stable?"

WHAT IT DOES (TWO MODES)
------------------------
1. SHADOW (matched-context faithfulness under realistic sampling):
   Vanilla generates a trajectory under top-p sampling. At each step,
   the bridge runs on the SAME context (vanilla's trajectory) and we
   measure per-step distributional divergence — KL, JS, top-1/top-5/top-10
   agreement, rank of vanilla's sampled token under bridge's distribution,
   and probability of vanilla's sampled token under both distributions.

   This is the CLEAN FAITHFULNESS METRIC for production-realism. Because
   contexts are matched (both models see the same input at every step),
   KL is interpretable.

2. FREE-RUN (independent trajectories with drift diagnostics):
   Vanilla and bridge each generate their own trajectories under top-p
   sampling. Once they diverge at any step, every subsequent step has
   them operating on different contexts. KL between them is then a
   TRAJECTORY-DRIFT DIAGNOSTIC, not a faithfulness metric. Cite it as
   "drift," not "faithfulness."

   What this mode IS for: the actual side-by-side text outputs, the
   divergence-step measurement, the entropy trend, the n-gram repetition
   under sampling (vs. M5's greedy looping), and the qualitative readout
   a human can compare.

COMMON RANDOM NUMBERS (CRN)
---------------------------
Codex flagged this and he's right. To separate DISTRIBUTION DRIFT from
SAMPLING RANDOMNESS, both vanilla and bridge use the SAME random draw at
every step. Concretely: at step t we draw a single u_t ~ Uniform[0,1) from
a seeded RNG keyed by (prompt_id, step), then both samplers apply that
u_t to their respective distributions via inverse-CDF sampling.

If both distributions are identical → both pick the same token (by CRN).
If they differ → token difference comes purely from distribution difference.

Standard torch.multinomial doesn't expose the underlying draw, so top-p
sampling is implemented manually with cumsum + seeded uniform.

PROMPT SET (Codex-approved 2026-05-30)
--------------------------------------
3 M5 carryovers (factual / code / open-ended) for direct comparability,
3 fresh (reasoning / technical / dialog) for quality showcase.

α SWEEP
-------
α ∈ {0.0, 0.1, 0.3, 0.5}. Smaller than M5 because:
  - α=0 is the bit-exact identity sanity (must match vanilla every step)
  - α=0.1, 0.3 are the realistic production blending range
  - α=0.5 is a stress upper bound
  - α=1.0 omitted (already characterized in M5; under sampling without
    teacher-forced safety net it produces visibly degraded text and we
    don't need that data point twice)

SAMPLING PARAMS (Codex-approved)
--------------------------------
top_p=0.9, temperature=0.8 — standard production defaults.

OUTPUT
------
1. CSV per (prompt, step, α, mode) with all metrics.
2. Markdown report with side-by-side text outputs for human review.
3. Console summary.

USAGE
-----
    python tasb_m6_generate.py             # full M6 (~35-45 min)
    python tasb_m6_generate.py --quick     # 3 prompts × 30 tokens (~10 min)
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

# UTF-8 stdout for Windows console compat (carried over from M5 patches)
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

ALPHA_SWEEP = [0.0, 0.1, 0.3, 0.5]
TOP_P       = 0.9
TEMPERATURE = 0.8

# Codex-approved prompt battery (3 M5 carryover + 3 fresh)
PROMPTS = [
    {"id": "M5_HC1", "domain": "FACTUAL",
     "text": "The capital of France is"},
    {"id": "M5_TC1", "domain": "CODE",
     "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"id": "M5_LX1", "domain": "OPEN_ENDED",
     "text": "Write a short story that begins: The old lighthouse keeper"},
    {"id": "FR_RS1", "domain": "REASONING",
     "text": "Explain why a compass points north in simple terms for a curious teenager."},
    {"id": "FR_TC1", "domain": "TECHNICAL",
     "text": "Write a Python function that checks whether a string is a palindrome, "
             "then explain how it works."},
    {"id": "FR_DI1", "domain": "DIALOG",
     "text": "Two engineers are arguing over coffee about whether machines can be "
             "creative. Write the conversation."},
]


# ── Distributional metrics (carried from M5: log-space, no clamp) ─────────

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


def _entropy(logits: torch.Tensor) -> float:
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return float(-(p * log_p).sum().item())


def _vanilla_top1_in_q_topk(p_logits: torch.Tensor, q_logits: torch.Tensor,
                             k: int) -> bool:
    """True if argmax(p) is in top-k of q. NOT a set-overlap metric.
    PATCH 2026-05-30 (post-Codex-M6-review, #2): renamed from
    _topk_agreement which was misleading. This measures one direction:
    'did vanilla's top-1 prediction make it into the bridge's top-k?'
    """
    p_top = int(p_logits.argmax().item())
    q_topk = torch.topk(q_logits, k).indices.tolist()
    return p_top in q_topk


def _topk_set_jaccard(p_logits: torch.Tensor, q_logits: torch.Tensor,
                      k: int) -> float:
    """Jaccard similarity of the top-k sets of p and q.
    PATCH 2026-05-30 (post-Codex-M6-review, #2): the actual set-overlap
    metric Codex asked for. Returns |intersection| / |union| ∈ [0, 1].
    A value of 1.0 means the bridge's top-k set EXACTLY matches vanilla's
    top-k set. 0.0 means they share no tokens. Independent of which is in
    top-1 — measures the candidate-pool agreement, not the argmax.
    """
    p_topk = set(torch.topk(p_logits, k).indices.tolist())
    q_topk = set(torch.topk(q_logits, k).indices.tolist())
    inter = len(p_topk & q_topk)
    union = len(p_topk | q_topk)
    return inter / union if union > 0 else 0.0


def _rank_of_token(token_id: int, logits: torch.Tensor) -> int:
    """Rank of `token_id` in `logits` (1-indexed; 1 = top-1)."""
    sorted_ids = torch.argsort(logits, descending=True).tolist()
    return sorted_ids.index(token_id) + 1


def _prob_of_token(token_id: int, logits: torch.Tensor) -> float:
    """Probability assigned to token_id under softmax(logits)."""
    p = torch.softmax(logits.float(), dim=-1)
    return float(p[token_id].item())


# ── CRN-seeded top-p sampling ──────────────────────────────────────────────

def _step_uniform(prompt_id: str, step: int, base_seed: int) -> float:
    """Deterministic Uniform[0,1) draw, keyed by (prompt_id, step).

    This is the CRN draw — both vanilla and bridge use this same u_t for
    inverse-CDF sampling at step t, so token differences come from
    distribution differences, not RNG noise.
    """
    key = f"toppick|{prompt_id}|{step}".encode('utf-8')
    crc = zlib.crc32(key) ^ base_seed
    # crc32 is 32-bit unsigned; divide to get [0, 1)
    return (crc & 0xFFFFFFFF) / float(1 << 32)


def _top_p_sample(logits: torch.Tensor, top_p: float, temperature: float,
                  u: float) -> int:
    """Sample one token via top-p (nucleus) sampling, using a given uniform u.

    Standard top-p: rank tokens by probability, keep the smallest prefix
    whose cumulative probability ≥ top_p, renormalize, sample.

    The `u` parameter is the CRN draw — same u across vanilla and bridge
    means inverse-CDF sampling picks the same token if distributions agree.
    """
    scaled = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum_p = torch.cumsum(sorted_p, dim=-1)

    # Find the smallest prefix whose cumsum >= top_p
    # nucleus_mask[i] = True if sorted_idx[i] is in the kept nucleus
    nucleus_mask = cum_p <= top_p
    # Always keep at least the top-1
    nucleus_mask[0] = True
    # Also include the first position that crosses top_p (so cumulative
    # mass is >= top_p, not just <)
    first_exceed = (cum_p >= top_p).nonzero()
    if len(first_exceed) > 0:
        nucleus_mask[first_exceed[0].item()] = True

    # Renormalize the nucleus
    nucleus_p = sorted_p * nucleus_mask.float()
    nucleus_p = nucleus_p / nucleus_p.sum()

    # Inverse-CDF sampling using the supplied u
    nucleus_cum = torch.cumsum(nucleus_p, dim=-1)
    # Find smallest i such that nucleus_cum[i] >= u
    pick_in_sorted = int((nucleus_cum >= u).nonzero()[0].item())
    return int(sorted_idx[pick_in_sorted].item())


# ── Mode 1: SHADOW (matched-context faithfulness) ──────────────────────────

def run_shadow(model, tok, prompt: dict, tokens: int,
               alphas: list[float], base_seed: int) -> list[dict]:
    """Vanilla generates under top-p. Bridge runs on the SAME context at
    every step. Per-step distributional metrics.

    This is the clean faithfulness metric under realistic sampling — both
    distributions are computed on identical input at every step.
    """
    records = []
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    ids = inputs['input_ids'].clone()

    for step in range(tokens):
        u_t = _step_uniform(prompt['id'], step, base_seed)
        sampler_seed = (base_seed + zlib.crc32(
            f"sampler|{prompt['id']}|{step}".encode('utf-8'))) & 0x7FFFFFFF

        # Run α=0 once to get vanilla logits at this context
        base = bridge_forward(
            model=model, tok=None, input_ids=ids,
            layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND, K=K_SAMPLES,
            seed=sampler_seed, return_intermediates=True)
        vanilla_logits = base.vanilla_logits[0, -1, :]

        # Vanilla samples its next token via CRN top-p
        vanilla_token = _top_p_sample(vanilla_logits, TOP_P, TEMPERATURE, u_t)

        vanilla_top1 = int(vanilla_logits.argmax().item())
        vanilla_H    = _entropy(vanilla_logits)

        # For each α, compute bridge distribution at THE SAME context
        for alpha in alphas:
            if alpha == 0.0:
                # PATCH 2026-05-30 (post-Codex-M6-review, #1):
                # MEASURE the α=0 identity, don't just assume it. We already
                # have `base` (the bridge_forward(alpha=0) result), and its
                # .logits should be bit-exact equal to .vanilla_logits. Record
                # the actual max_abs_diff so the CSV proves the identity per
                # step rather than asserting it by construction.
                bridge_logits = base.logits[0, -1, :]
                alpha0_max_abs_diff = float(
                    (vanilla_logits.float() - bridge_logits.float())
                    .abs().max().item())
            else:
                result = bridge_forward(
                    model=model, tok=None, input_ids=ids,
                    layer_idx=LAYER_IDX, alpha=alpha, backend=BACKEND,
                    K=K_SAMPLES, seed=sampler_seed, return_intermediates=True)
                bridge_logits = result.logits[0, -1, :]
                alpha0_max_abs_diff = -1.0   # sentinel: only meaningful at α=0

            bridge_token_topp = _top_p_sample(
                bridge_logits, TOP_P, TEMPERATURE, u_t)
            bridge_top1 = int(bridge_logits.argmax().item())

            kl = _kl(vanilla_logits, bridge_logits)
            js = _js(vanilla_logits, bridge_logits)
            bridge_H = _entropy(bridge_logits)

            records.append({
                'mode':              'shadow',
                'prompt_id':         prompt['id'],
                'domain':            prompt['domain'],
                'step':              step + 1,
                'alpha':             alpha,
                'context_len':       int(ids.shape[1]),
                'u_t':               round(u_t, 6),
                'sampler_seed':      sampler_seed,
                'vanilla_token':     vanilla_token,
                'vanilla_top1':      vanilla_top1,
                'vanilla_top1_prob': _prob_of_token(vanilla_top1, vanilla_logits),
                'vanilla_entropy':   vanilla_H,
                'bridge_token_topp': bridge_token_topp,
                'bridge_top1':       bridge_top1,
                'bridge_entropy':    bridge_H,
                'top1_agree':        int(bridge_top1 == vanilla_top1),
                # PATCH 2026-05-30 (Codex #2): renamed for clarity. These
                # measure "did vanilla's top-1 land in bridge's top-k?",
                # NOT set overlap.
                'vanilla_top1_in_bridge_top5':
                    int(_vanilla_top1_in_q_topk(vanilla_logits, bridge_logits, 5)),
                'vanilla_top1_in_bridge_top10':
                    int(_vanilla_top1_in_q_topk(vanilla_logits, bridge_logits, 10)),
                # PATCH 2026-05-30 (Codex #2): real top-k set overlap (Jaccard)
                'top5_jaccard':
                    _topk_set_jaccard(vanilla_logits, bridge_logits, 5),
                'top10_jaccard':
                    _topk_set_jaccard(vanilla_logits, bridge_logits, 10),
                'rank_of_vanilla_token_under_bridge':
                                     _rank_of_token(vanilla_token, bridge_logits),
                'prob_of_vanilla_token_under_vanilla':
                                     _prob_of_token(vanilla_token, vanilla_logits),
                'prob_of_vanilla_token_under_bridge':
                                     _prob_of_token(vanilla_token, bridge_logits),
                'topp_token_agree':  int(bridge_token_topp == vanilla_token),
                'kl':                kl,
                'js':                js,
                # PATCH 2026-05-30 (Codex #1): measured α=0 identity, -1 elsewhere
                'alpha0_max_abs_diff': alpha0_max_abs_diff,
            })

        # Advance vanilla's context (shadow follows vanilla's trajectory)
        ids = torch.cat(
            [ids, torch.tensor([[vanilla_token]], device=model.device)],
            dim=1)
        if vanilla_token == tok.eos_token_id:
            break

    return records


# ── Mode 2: FREE-RUN (independent trajectories) ────────────────────────────

@dataclass
class FreeRunResult:
    prompt_id: str
    alpha: float
    vanilla_text: str = ""
    bridge_text: str = ""
    vanilla_tokens: list[int] = field(default_factory=list)
    bridge_tokens: list[int] = field(default_factory=list)
    divergence_step: int = -1     # first step where vanilla_token != bridge_token
    # PATCH 2026-05-30 (Codex #1): track EOS per branch so we can surface
    # early termination cleanly in the report
    eos_step_vanilla: int = -1    # -1 if no EOS in the captured range
    eos_step_bridge: int = -1
    per_step: list[dict] = field(default_factory=list)


def run_free(model, tok, prompt: dict, tokens: int,
             alpha: float, base_seed: int) -> FreeRunResult:
    """Vanilla and bridge each generate their own trajectories.

    CRN: same u_t shared between them at each step. Once they diverge,
    their contexts are different and KL is no longer a faithfulness metric
    — it's a drift diagnostic.
    """
    inputs = tok(prompt['text'], return_tensors='pt').to(model.device)
    vanilla_ids = inputs['input_ids'].clone()
    bridge_ids  = inputs['input_ids'].clone()

    result = FreeRunResult(prompt_id=prompt['id'], alpha=alpha)

    # PATCH 2026-05-30 (post-Codex-M6-review #1): per-branch EOS tracking.
    # Old code only broke when BOTH branches hit EOS, so a branch that emitted
    # EOS at step 22 would keep generating from a post-EOS context for 8 more
    # steps, contaminating the KL/drift averages with undefined-regime data.
    # Now we stop the moment EITHER branch hits EOS.
    vanilla_done = False
    bridge_done = False
    result.eos_step_vanilla = -1
    result.eos_step_bridge = -1

    for step in range(tokens):
        u_t = _step_uniform(prompt['id'], step, base_seed)
        sampler_seed = (base_seed + zlib.crc32(
            f"sampler|{prompt['id']}|{step}".encode('utf-8'))) & 0x7FFFFFFF

        # Vanilla pass on vanilla's own context
        van_base = bridge_forward(
            model=model, tok=None, input_ids=vanilla_ids,
            layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND, K=K_SAMPLES,
            seed=sampler_seed, return_intermediates=True)
        vanilla_logits = van_base.vanilla_logits[0, -1, :]
        vanilla_token = _top_p_sample(vanilla_logits, TOP_P, TEMPERATURE, u_t)
        vanilla_H = _entropy(vanilla_logits)

        # Bridge pass on bridge's own context
        if alpha == 0.0:
            # α=0 is identity. If contexts still match, we can shortcut for
            # speed (M5 has the rigorous identity proof). If contexts somehow
            # already differ, we run the real forward.
            if torch.equal(vanilla_ids, bridge_ids):
                bridge_logits = vanilla_logits   # constructed identity (M5-validated)
            else:
                br_base = bridge_forward(
                    model=model, tok=None, input_ids=bridge_ids,
                    layer_idx=LAYER_IDX, alpha=0.0, backend=BACKEND,
                    K=K_SAMPLES, seed=sampler_seed,
                    return_intermediates=True)
                bridge_logits = br_base.vanilla_logits[0, -1, :]
        else:
            br = bridge_forward(
                model=model, tok=None, input_ids=bridge_ids,
                layer_idx=LAYER_IDX, alpha=alpha, backend=BACKEND,
                K=K_SAMPLES, seed=sampler_seed, return_intermediates=True)
            bridge_logits = br.logits[0, -1, :]
        bridge_token = _top_p_sample(bridge_logits, TOP_P, TEMPERATURE, u_t)
        bridge_H = _entropy(bridge_logits)

        # Per-step diagnostic (NOTE: KL here is drift when contexts diverge)
        contexts_match = torch.equal(vanilla_ids, bridge_ids)
        kl_drift = _kl(vanilla_logits, bridge_logits)
        js_drift = _js(vanilla_logits, bridge_logits)

        result.per_step.append({
            'step':            step + 1,
            'u_t':             u_t,
            'contexts_match':  int(contexts_match),
            'vanilla_token':   vanilla_token,
            'bridge_token':    bridge_token,
            'vanilla_entropy': vanilla_H,
            'bridge_entropy':  bridge_H,
            'kl_drift':        kl_drift,     # drift when contexts differ
            'js_drift':        js_drift,
            'prob_of_vanilla_token_under_vanilla':
                               _prob_of_token(vanilla_token, vanilla_logits),
            'prob_of_bridge_token_under_bridge':
                               _prob_of_token(bridge_token, bridge_logits),
            'tokens_agree':    int(vanilla_token == bridge_token),
        })

        if vanilla_token != bridge_token and result.divergence_step == -1:
            result.divergence_step = step + 1

        result.vanilla_tokens.append(vanilla_token)
        result.bridge_tokens.append(bridge_token)

        # Check EOS per branch and record the step it happened
        if vanilla_token == tok.eos_token_id and not vanilla_done:
            vanilla_done = True
            result.eos_step_vanilla = step + 1
        if bridge_token == tok.eos_token_id and not bridge_done:
            bridge_done = True
            result.eos_step_bridge = step + 1

        # PATCH 2026-05-30 (Codex #1): stop the moment EITHER branch hits EOS.
        # Continuing past EOS feeds the EOS token back into context, which is
        # an undefined regime and contaminates KL/agreement averages.
        if vanilla_done or bridge_done:
            break

        # Advance each branch on its own choice (only reached if neither EOS'd)
        vanilla_ids = torch.cat(
            [vanilla_ids, torch.tensor([[vanilla_token]], device=model.device)],
            dim=1)
        bridge_ids = torch.cat(
            [bridge_ids, torch.tensor([[bridge_token]], device=model.device)],
            dim=1)

    result.vanilla_text = tok.decode(result.vanilla_tokens)
    result.bridge_text  = tok.decode(result.bridge_tokens)
    return result


# ── n-gram repetition (carried from M5 recut) ─────────────────────────────

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


# ── Aggregate + report ────────────────────────────────────────────────────

def shadow_summary(records: list[dict], alphas: list[float]):
    """Print shadow-mode aggregate. This is the clean faithfulness readout.
    PATCH 2026-05-30 (Codex #2): columns renamed for clarity, Jaccard added.
    """
    print(f"\n{'═'*78}")
    print(bold("  SHADOW MODE — MATCHED-CONTEXT FAITHFULNESS (clean metric)"))
    print(f"{'═'*78}\n")
    print(f"  {'α':>5}  {'n':>5}  {'top1%':>7}  {'v_in_b5%':>9}  {'v_in_b10%':>10}  "
          f"{'top5_J':>8}  {'top10_J':>8}  {'topp%':>7}  {'mean_KL':>10}  "
          f"{'α0_diff':>10}")
    print(f"  {'─'*100}")
    for alpha in alphas:
        rows = [r for r in records if r['alpha'] == alpha]
        if not rows:
            continue
        n = len(rows)
        top1 = np.mean([r['top1_agree']                  for r in rows]) * 100
        v5   = np.mean([r['vanilla_top1_in_bridge_top5'] for r in rows]) * 100
        v10  = np.mean([r['vanilla_top1_in_bridge_top10'] for r in rows]) * 100
        j5   = np.mean([r['top5_jaccard']                 for r in rows])
        j10  = np.mean([r['top10_jaccard']                for r in rows])
        topp = np.mean([r['topp_token_agree']             for r in rows]) * 100
        mkl  = np.mean([r['kl']                           for r in rows])
        if alpha == 0.0:
            # Worst-case α=0 max_abs_diff across rows (should be 0 or ε)
            a0   = max(r['alpha0_max_abs_diff'] for r in rows)
            a0_str = f"{a0:.2e}"
        else:
            a0_str = "—"
        color = green if top1 >= 95 else yellow if top1 >= 80 else red
        print(f"  {alpha:>5.2f}  {n:>5}  {color(f'{top1:>5.1f}%')}  "
              f"{v5:>7.1f}%  {v10:>8.1f}%  "
              f"{j5:>8.3f}  {j10:>8.3f}  {topp:>5.1f}%  "
              f"{mkl:>10.6f}  {a0_str:>10}")

    # Loud check on α=0 identity (Codex #1)
    a0_rows = [r for r in records if r['alpha'] == 0.0]
    if a0_rows:
        worst = max(r['alpha0_max_abs_diff'] for r in a0_rows)
        print()
        if worst < 1e-5:
            print(green(f"  ✓ α=0 identity confirmed in shadow CSV: "
                        f"worst max_abs_diff = {worst:.2e}"))
            print(green(f"    (alpha_zero_identity holds end-to-end under "
                        f"top-p sampling contexts)"))
        else:
            print(red(f"  ✗ α=0 identity FAILED: worst max_abs_diff = {worst:.2e}"))
            print(red(f"    The alpha_zero invariant broke. Investigate."))


def free_run_summary(free_results: list[FreeRunResult]):
    """Print free-run trajectory drift diagnostics.
    PATCH 2026-05-30 (Codex #3): KL split into pre-divergence
    (same-context, faithfulness-interpretable) and post-divergence
    (different-context, drift only).
    """
    print(f"\n{'═'*78}")
    print(bold("  FREE-RUN MODE — TRAJECTORY DRIFT"))
    print(yellow(f"  NOTE: post-divergence KL is a DRIFT DIAGNOSTIC, not faithfulness."))
    print(yellow(f"  Pre-divergence steps (vanilla and bridge see same context) "
                 f"DO yield interpretable KL."))
    print(f"{'═'*78}\n")
    print(f"  {'prompt':<10} {'α':>5}  {'steps':>5}  {'agree%':>7}  "
          f"{'div@':>5}  {'eos_v':>5}  {'eos_b':>5}  "
          f"{'pre_n':>5}  {'KL_pre (faithfulness)':>22}  "
          f"{'post_n':>6}  {'KL_post (drift)':>16}  "
          f"{'8gr_v':>7}  {'8gr_b':>7}")
    print(f"  {'─'*135}")
    for r in free_results:
        steps = r.per_step
        n = len(steps)
        if n == 0:
            continue
        agree = np.mean([s['tokens_agree'] for s in steps]) * 100
        rep_v = _ngram_repeat_rate(r.vanilla_tokens, 8) * 100
        rep_b = _ngram_repeat_rate(r.bridge_tokens, 8) * 100

        # Pre/post split using the per-step contexts_match flag
        pre_steps  = [s for s in steps if s['contexts_match'] == 1]
        post_steps = [s for s in steps if s['contexts_match'] == 0]
        pre_n  = len(pre_steps)
        post_n = len(post_steps)
        kl_pre  = np.mean([s['kl_drift'] for s in pre_steps])  if pre_steps  else float('nan')
        kl_post = np.mean([s['kl_drift'] for s in post_steps]) if post_steps else float('nan')

        div = r.divergence_step
        div_s = f"{div}" if div > 0 else "—"
        eos_v_s = f"{r.eos_step_vanilla}" if r.eos_step_vanilla > 0 else "—"
        eos_b_s = f"{r.eos_step_bridge}"  if r.eos_step_bridge > 0  else "—"
        pre_kl_s  = f"{kl_pre:.6f}"  if pre_n  > 0 else "—"
        post_kl_s = f"{kl_post:.6f}" if post_n > 0 else "—"

        print(f"  {r.prompt_id:<10} {r.alpha:>5.2f}  {n:>5}  "
              f"{agree:>6.1f}%  {div_s:>5}  {eos_v_s:>5}  {eos_b_s:>5}  "
              f"{pre_n:>5}  {pre_kl_s:>22}  "
              f"{post_n:>6}  {post_kl_s:>16}  "
              f"{rep_v:>6.1f}%  {rep_b:>6.1f}%")


def write_markdown_report(free_results: list[FreeRunResult],
                          path: str, alphas: list[float]):
    """Write the side-by-side text report for human review."""
    with open(path, 'w') as f:
        f.write("# M6 Free-Run Generation — Side-by-Side\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Sampling:** top_p={TOP_P}, temperature={TEMPERATURE}\n\n")
        f.write(f"**Bridge:** L{LAYER_IDX}, backend={BACKEND}, K={K_SAMPLES}\n\n")
        f.write(f"**CRN:** same u_t draw shared between vanilla and bridge at each step.\n\n")
        f.write("> ⚠ Free-run KL is a *drift diagnostic*, not a faithfulness metric. "
                "Once trajectories diverge at any step, vanilla and bridge are answering "
                "different contexts. For clean per-step faithfulness under top-p sampling, "
                "see the matched-context shadow CSV.\n\n")

        # Group by prompt
        by_prompt = defaultdict(list)
        for r in free_results:
            by_prompt[r.prompt_id].append(r)

        for pid in sorted(by_prompt.keys()):
            prompt_text = next(p['text'] for p in PROMPTS if p['id'] == pid)
            domain = next(p['domain'] for p in PROMPTS if p['id'] == pid)
            f.write(f"\n## {pid} ({domain})\n\n")
            f.write(f"**Prompt:** `{prompt_text}`\n\n")
            for r in sorted(by_prompt[pid], key=lambda x: x.alpha):
                f.write(f"### α = {r.alpha}\n\n")
                # PATCH 2026-05-30 (Codex #3): split KL into pre/post divergence
                pre_steps  = [s for s in r.per_step if s['contexts_match'] == 1]
                post_steps = [s for s in r.per_step if s['contexts_match'] == 0]
                if r.divergence_step > 0:
                    f.write(f"- Divergence at step {r.divergence_step} "
                            f"(out of {len(r.per_step)})\n")
                else:
                    f.write(f"- Trajectories never diverged "
                            f"({len(r.per_step)} steps)\n")
                # PATCH 2026-05-30 (Codex #1): surface per-branch EOS
                if r.eos_step_vanilla > 0:
                    f.write(f"- Vanilla branch emitted EOS at step "
                            f"{r.eos_step_vanilla} (loop terminated)\n")
                if r.eos_step_bridge > 0:
                    f.write(f"- Bridge branch emitted EOS at step "
                            f"{r.eos_step_bridge} (loop terminated)\n")
                f.write("\n")
                if pre_steps:
                    kl_pre = float(np.mean([s['kl_drift'] for s in pre_steps]))
                    f.write(f"- **Pre-divergence** (n={len(pre_steps)}, "
                            f"same context — faithfulness-interpretable): "
                            f"mean KL = {kl_pre:.6f}\n")
                if post_steps:
                    kl_post = float(np.mean([s['kl_drift'] for s in post_steps]))
                    f.write(f"- **Post-divergence** (n={len(post_steps)}, "
                            f"different contexts — drift diagnostic only): "
                            f"mean KL = {kl_post:.6f}\n")
                f.write("\n**Vanilla output:**\n\n")
                f.write(f"```\n{r.vanilla_text}\n```\n\n")
                f.write("**Bridge output:**\n\n")
                f.write(f"```\n{r.bridge_text}\n```\n\n")
                f.write("---\n")


def write_shadow_csv(records: list[dict], path: str):
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def write_free_csv(free_results: list[FreeRunResult], path: str):
    """Write per-step free-run rows. PATCH 2026-05-30 (Codex M6 smoke-test
    feedback): eos_step_vanilla and eos_step_bridge are denormalized onto
    every row so post-hoc analysis can filter/join without re-reading the
    FreeRunResult container. -1 sentinel means "no EOS in the captured range."
    """
    rows = []
    for r in free_results:
        for s in r.per_step:
            row = {
                'prompt_id':        r.prompt_id,
                'alpha':            r.alpha,
                'divergence_step':  r.divergence_step,
                'eos_step_vanilla': r.eos_step_vanilla,
                'eos_step_bridge':  r.eos_step_bridge,
            }
            row.update(s)
            rows.append(row)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument('--tokens', type=int, default=60)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--quick',  action='store_true')
    ap.add_argument('--seed',   type=int, default=42)
    ap.add_argument('--mode',   choices=['shadow', 'free', 'both'], default='both')
    args = ap.parse_args()

    if args.quick:
        args.tokens = 30
        prompts = PROMPTS[:3]
    else:
        prompts = PROMPTS

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'═'*78}")
    print(bold(f"  TASB M6 PRODUCTION-REALISM  {ts}"))
    print(f"{'═'*78}")
    print(f"  Model:       {args.model}")
    print(f"  Layer:       L{LAYER_IDX}, backend={BACKEND}, K={K_SAMPLES}")
    print(f"  Sampling:    top_p={TOP_P}, temperature={TEMPERATURE}")
    print(f"  α sweep:     {ALPHA_SWEEP}")
    print(f"  Prompts:     {len(prompts)} × {args.tokens} tokens")
    print(f"  Mode:        {args.mode}")
    print(f"  Seed:        {args.seed}")
    print(f"  CRN:         enabled (vanilla and bridge share u_t per step)")
    print(f"{'═'*78}")

    model, tok = load_model(args.model)

    t0 = time.perf_counter()

    # ── SHADOW MODE ─────────────────────────────────────────────────────
    shadow_records = []
    if args.mode in ('shadow', 'both'):
        print(f"\n{bold('── Running SHADOW mode ──')}")
        for prompt in prompts:
            print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]: "
                  f"{gray(prompt['text'][:50])}...")
            recs = run_shadow(
                model=model, tok=tok, prompt=prompt,
                tokens=args.tokens, alphas=ALPHA_SWEEP,
                base_seed=args.seed)
            shadow_records.extend(recs)
            # quick per-prompt α=0.3 readout
            r03 = [r for r in recs if r['alpha'] == 0.3]
            if r03:
                t1 = np.mean([r['top1_agree']       for r in r03]) * 100
                tp = np.mean([r['topp_token_agree'] for r in r03]) * 100
                kl = np.mean([r['kl']               for r in r03])
                print(f"    α=0.3: top1={t1:.1f}%, "
                      f"topp_agree={tp:.1f}%, mean_KL={kl:.5f}")

        shadow_csv = os.path.join(
            args.outdir, f"tasb_m6_shadow_{ts}.csv")
        write_shadow_csv(shadow_records, shadow_csv)
        print(f"\n  Shadow CSV: {shadow_csv}  ({len(shadow_records)} rows)")
        shadow_summary(shadow_records, ALPHA_SWEEP)

    # ── FREE-RUN MODE ───────────────────────────────────────────────────
    free_results = []
    if args.mode in ('free', 'both'):
        print(f"\n{bold('── Running FREE-RUN mode ──')}")
        for prompt in prompts:
            print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]")
            for alpha in ALPHA_SWEEP:
                t_start = time.perf_counter()
                result = run_free(
                    model=model, tok=tok, prompt=prompt,
                    tokens=args.tokens, alpha=alpha, base_seed=args.seed)
                t_elapsed = time.perf_counter() - t_start
                free_results.append(result)
                div = result.divergence_step
                div_str = f"step {div}" if div > 0 else "never"
                eos_bits = []
                if result.eos_step_vanilla > 0:
                    eos_bits.append(f"V-EOS@{result.eos_step_vanilla}")
                if result.eos_step_bridge > 0:
                    eos_bits.append(f"B-EOS@{result.eos_step_bridge}")
                eos_str = (", " + ", ".join(eos_bits)) if eos_bits else ""
                print(f"    α={alpha}: divergence={div_str}{eos_str}, "
                      f"{t_elapsed:.1f}s")

        free_csv = os.path.join(args.outdir, f"tasb_m6_free_{ts}.csv")
        free_md  = os.path.join(args.outdir, f"tasb_m6_free_{ts}.md")
        write_free_csv(free_results, free_csv)
        write_markdown_report(free_results, free_md, ALPHA_SWEEP)
        print(f"\n  Free-run CSV: {free_csv}")
        print(f"  Free-run MD:  {free_md}")
        free_run_summary(free_results)

    t_total = time.perf_counter() - t0
    print(f"\n{'═'*78}")
    print(bold(f"  M6 COMPLETE  ({t_total:.1f}s = {t_total/60:.1f}min)"))
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
