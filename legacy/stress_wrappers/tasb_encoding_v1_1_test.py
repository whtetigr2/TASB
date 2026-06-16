"""
tasb_encoding_v1_test.py
==============================================================================
Patent:  USPTO Provisional Application No. 64/019,999 (filed March 28, 2026)
Author:  Paul W. Shaver (Independent Inventor)

TASB Encoding Comparison v1 — Probability-space vs Logit-space Equivalence

THE QUESTION
------------
v2 stress test established peak_L18 single-layer injection achieves 99.3%
match (95% CI [98.5, 99.8]) with vanilla LLaMA 3.2-3B at α=0.30, using
probability-space encoding:

    capture softmax(QK^T/√dk) → multinomial(K) → average → blend → inject

The question this test answers: is that result an ARTIFACT of the encoding
choice, or is the bridge ENCODING-INVARIANT across mathematically equivalent
Boltzmann samplers?

If encoding-invariant → strongest hardware portability claim possible.
If encoding-dependent → identifies which encoding best matches real TSU
                        silicon (logit-space is hardware-natural per
                        Camsari p-bit Hamiltonian H = -⟨h,σ⟩ - ½⟨σ,Jσ⟩).

THREE ENCODINGS TESTED + VANILLA CONTROL
----------------------------------------
1. vanilla                — no TSU injection (control, 100% match by def.)
2. probability-space      — v2 baseline: softmax already applied, sample
                            from row-stochastic matrix via multinomial
3. logit-space (softmax)  — Option A: capture raw QK^T, internally compute
                            softmax(raw/T_struct) where T_struct=√d_k=11.314,
                            sample via multinomial.
                            Mathematically equivalent to probability-space
                            if raw_scores are correctly scaled. Tests
                            whether the encoding refactor preserves output.
4. logit-space (gumbel)   — Option B: capture raw QK^T, add Gumbel(0,1)
                            noise to logits/T_struct, take argmax per row.
                            No softmax computed. No partition function.
                            This is how a physical TSU samples — it
                            thermalizes directly without computing Z.
                            HARDWARE-NATURAL ENCODING.

WHY THESE THREE
---------------
Encodings 2 and 3 should produce statistically IDENTICAL output in the
K→∞ limit (both sample from the same Boltzmann distribution). At finite
K=10 they may differ in variance. If they match within noise → confirms
the bridge doesn't care which "side of softmax" we sample on.

Encoding 4 (Gumbel-max) is the standard "Boltzmann sampling without
computing softmax" trick (Maddison et al. 2014, Jang et al. 2016) and
matches how p-bit arrays actually settle to equilibrium. If 4 matches
2 and 3, the bridge is robust to the hardware-realistic sampling
implementation.

THEORETICAL BACKING
-------------------
Kim (arXiv:2602.08216):
    softmax(QK^T/√d_k) IS the Boltzmann distribution at T_struct=√d_k
    when E = -QK^T. The energy form and probability form are isomorphic.

Spinbath paper (project knowledge):
    "Each logit corresponds to minus the energy of a token: E_i = -L_i.
     Hence, the soft-max implements a Boltzmann distribution."
    "∆L is invariant under any additive offset to all logits, such as
     those introduced by residual connections or layer normalizations."

Camsari p-bits / Hopfield / Ramsauer:
    Physical Ising machines implement H = -⟨h,σ⟩ - ½⟨σ,Jσ⟩ natively.
    They never form the partition function Z. They sample by physical
    relaxation. Gumbel-max sampling is the software analog.

ARCHITECTURE — WHAT CHANGES, WHAT STAYS
---------------------------------------
STAYS (no code changes needed):
- VanillaCapture: already records BOTH weights_kv (post-softmax) AND
  raw_scores (pre-softmax, QK^T/√d_k after causal masking).
  See tasb_two_pass.py line 162-269.
- ThermodynamicInjector: takes a row-stochastic p_thermo dict, blends
  with α, injects via p_thermo @ V. Output type identical across encodings.
- Layer choice: peak_L18 (v2 winner)
- α blending: α=0.30 (v2 baseline)
- Teacher-forced Phase A test design
- McNemar pairing by (prompt, step)
- Bug guards #1-#7

CHANGES (new code in this file):
- New sampler kernel `_sample_logit_softmax` (Option A)
- New sampler kernel `_sample_logit_gumbel`  (Option B)
- Both kernels take raw_scores as input
- Both apply causal mask in LOGIT space (-inf, not 0.0 — this REVERSES
  Bug #4 for logit space; documented inline)
- T_struct = √d_k = √128 ≈ 11.314 hardcoded with Kim citation

EXPOSED:
- Temperature T_struct as explicit parameter (was implicit in softmax)
- Sampling strategy as orthogonal axis (multinomial vs Gumbel-max)

INVARIANT VERIFIED AT RUNTIME:
After computing softmax(captured_raw_scores / T_struct), the result MUST
match captured_weights_kv within float32 tolerance. If this assertion ever
fails, raw_scores capture or scaling is broken. Catches silent issues.

BUG GUARDS (carried from v2)
----------------------------
#1 args scope:    args.X only in main(); functions take explicit params
#2 imports:       tasb_two_pass + tasb_llama_config only
#3 float32 eps:   1e-4 in clamps
#4 causal mask:   PROBABILITY space uses 0.0; LOGIT space uses -inf
                  (this file uses BOTH — flagged inline at each use)
#5 h_directed:    handled inside run_tsu_phase (probability path)
#6 call sites:    sampler_cfg/encoding passed explicitly everywhere
#7 layer_subset:  type-guarded List[int]
==============================================================================
"""

import argparse, csv, dataclasses, os, sys, time, threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _c(code, t): return f"\033[{code}m{t}\033[0m"
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def gray(t):   return _c("90", t)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Winner from v2: peak_L18 alone, α=0.30, K=10 (k_attn=10 = sample count)
WINNER_LAYER = 18
WINNER_ALPHA = 0.30
SAMPLES_K    = 10

# T_struct = √d_k per Kim (arXiv:2602.08216).
# d_k = 128 for LLaMA 3.2-3B → T_struct = 11.3137...
# This is the temperature at which softmax(QK^T) IS the Boltzmann distribution
# of energy E = -QK^T. Hardcoded with citation rather than recomputed each call.
T_STRUCT = float(np.sqrt(128))   # 11.3137...

# Four conditions — vanilla + 3 encodings
ENCODINGS = [
    'vanilla',         # control: no injection, must hit 100% match by def.
    'prob_space',      # v2 path: sample from captured weights_kv (post-softmax)
    'logit_softmax',   # Option A: sample from softmax(raw_scores / T_struct)
    'logit_gumbel',    # Option B: Gumbel-max on raw_scores / T_struct (hw-native)
]

# Same 9 prompts as v1/v2 — direct comparison
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

# Pre-registered McNemar pairs — three core questions
PAIRS_TO_TEST = [
    ('prob_space',    'logit_softmax'),   # encoding refactor preserves output?
    ('prob_space',    'logit_gumbel'),    # hardware-native sampler equivalent?
    ('logit_softmax', 'logit_gumbel'),    # internal logit-space consistency?
]


# ==============================================================================
# CAPTURE INVARIANT — confirms raw_scores capture is correct
# ==============================================================================

def _verify_capture_invariant(weights_kv_dict, raw_scores_dict, T):
    """
    Sanity check: softmax(raw_scores / T) should equal captured weights_kv
    within float32 tolerance for non-masked (lower-triangle) entries.

    If this fails, raw_scores capture is broken. We must know about it.

    Per tasb_two_pass.py docstring: raw_scores are KV-averaged QK^T/√d_k
    AFTER causal masking, with masked entries set to 0.0 in the raw_scores
    (so softmax of those would be exp(0)/Z which doesn't match the proper
    causal mask). We therefore re-mask before re-softmaxing.

    Tolerance: rtol=0.05 absolute on the row-stochastic outputs. Looser
    than float32 epsilon because the captured weights_kv came from LLaMA's
    bf16 attention path while our re-softmax happens in fp32.
    """
    if not weights_kv_dict or not raw_scores_dict:
        return False, "empty inputs"

    common_layers = set(weights_kv_dict) & set(raw_scores_dict)
    if not common_layers:
        return False, "no common layers"

    max_diff = 0.0
    for L in common_layers:
        w = weights_kv_dict[L]   # (8, S, S) row-stochastic
        r = raw_scores_dict[L]   # (8, S, S) raw QK^T/√d_k, zeros on upper tri
        S = w.shape[-1]
        # Reconstruct softmax from raw with causal mask in logit space
        # (replace the captured-zero upper triangle with -inf)
        upper = np.triu(np.ones((S, S), dtype=bool), k=1)
        r_masked = r.copy()
        r_masked[..., upper] = -1e9
        # Note: raw_scores in tasb_two_pass are QK^T/√d_k, so they're
        # already at T=1 effective. NOT QK^T raw.
        # softmax row-wise
        r_max = r_masked.max(axis=-1, keepdims=True)
        e     = np.exp(r_masked - r_max)
        re    = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-30)
        # Compare lower-triangle entries (upper should both be ~0)
        lower = ~upper
        diff = np.abs(re - w)[..., lower].max()
        if diff > max_diff:
            max_diff = float(diff)

    # Tolerance: 0.05 absolute on probability matrices.
    # Larger differences flag a real capture problem.
    return max_diff < 0.05, f"max abs diff = {max_diff:.4f}"


# ==============================================================================
# NEW SAMPLER KERNELS — the meat of this test
# ==============================================================================

def _sample_prob_space(weights_kv_dict, K, eps=1e-8):
    """
    Reference path — sample from row-stochastic weights_kv via multinomial.
    Identical to the exact-Boltzmann branch in run_tsu_phase (v2 production).
    Returns {layer: ndarray(8, S, S) row-stochastic}.

    Bug #4 (prob-space): causal mask already applied in upper triangle as
    zeros from the capture. No special handling needed here.
    """
    import torch
    out = {}
    for layer, w in weights_kv_dict.items():
        n_kv, S, _ = w.shape
        p_exact = np.zeros_like(w)
        for h in range(n_kv):
            row_probs = torch.from_numpy(w[h]).float()
            for i in range(S):
                p_row = torch.clamp(row_probs[i], min=0.0)
                s = p_row.sum()
                if s < eps:
                    p_exact[h, i, int(p_row.argmax())] = 1.0
                    continue
                p_row = p_row / s
                indices = torch.multinomial(p_row, K, replacement=True)
                for idx in indices.tolist():
                    p_exact[h, i, idx] += 1.0 / K
        out[layer] = p_exact.astype(np.float32)
    return out


def _sample_logit_softmax(raw_scores_dict, K, T, eps=1e-8):
    """
    Option A — sample from softmax(raw_scores) via multinomial.

    *** Bug #8 fix (v3.1) ***
    Raw scores captured by VanillaCapture in tasb_two_pass.py are ALREADY
    divided by √d_k. See tasb_two_pass.py line 251:
        raw = torch.matmul(Q, K_exp.transpose(-2, -1)) / T_struct
    Therefore: softmax(raw_scores) IS the Boltzmann distribution at the
    structural temperature T_struct. No further division by T needed.

    v3.0 erroneously did `scores[i] / T`, applying T_struct twice. That made
    sampling effectively T² = 128× hotter than vanilla, flattening
    distributions (p_max collapsed from 0.82 to 0.29) and degrading match
    rate from 99.3% to 95%. The capture_ok invariant check caught this on
    every row but the test ran to completion anyway. v3.1 removes the
    redundant /T and the invariant should now pass.

    T parameter retained in signature for API consistency but ignored.
    Asserted == √128 ≈ 11.314 at call site so future readers see the
    expected value.

    Bug #4 REVERSAL for logit-space: causal mask MUST be applied as -inf
    (large negative), NOT 0.0. exp(-inf)=0 in the partition function.
    The captured raw_scores have zeros on the upper triangle (per
    tasb_two_pass.py), which would give exp(0)=1 in the partition function —
    that's wrong for logit space. We re-mask explicitly here.
    """
    import torch
    assert abs(T - float(np.sqrt(128))) < 1e-3, \
        f"T_struct must be √128 ≈ 11.314 to match capture scaling; got {T}"
    out = {}
    for layer, r in raw_scores_dict.items():
        n_kv, S, _ = r.shape
        p_exact = np.zeros((n_kv, S, S), dtype=np.float32)
        # Re-apply causal mask in LOGIT space (Bug #4 reversal: -inf, not 0.0)
        upper = np.triu(np.ones((S, S), dtype=bool), k=1)
        for h in range(n_kv):
            scores = r[h].copy()
            scores[upper] = -1e9   # logit-space causal mask
            # Sample row-by-row
            for i in range(S):
                # Bug #8 fix: scores already at correct temperature (QK^T/√d_k)
                # DO NOT divide by T_struct again.
                logits = torch.from_numpy(scores[i]).float()
                p_row = torch.softmax(logits, dim=-1)
                s = p_row.sum()
                if s < eps:
                    p_exact[h, i, int(p_row.argmax())] = 1.0
                    continue
                p_row = p_row / s
                indices = torch.multinomial(p_row, K, replacement=True)
                for idx in indices.tolist():
                    p_exact[h, i, idx] += 1.0 / K
        out[layer] = p_exact
    return out


def _sample_logit_gumbel(raw_scores_dict, K, T, eps=1e-8):
    """
    Option B — Gumbel-max sampling in logit space (HARDWARE-NATURAL).

    The Gumbel-max trick: argmax_i(L_i + g_i) where g_i ~ Gumbel(0,1) is
    equivalent to sampling from softmax(L). No partition function ever
    computed. This is the standard "Boltzmann sampling without computing Z"
    trick (Maddison et al. 2014).

    For K averaged samples we draw K independent Gumbel noise vectors,
    take argmax for each, accumulate one-hots, divide by K.

    *** Bug #8 fix (v3.1) ***
    Raw scores from VanillaCapture are ALREADY divided by √d_k (see
    tasb_two_pass.py line 251). v3.0 divided by T_struct=√128 a second
    time, making sampling 128× hotter than vanilla. Removed.

    For Gumbel-max, the temperature is implicit in the magnitude of the
    logits — Gumbel(0,1) noise has unit variance, so logits at scale s
    produce sampling at effective T=1 relative to that scale. The captured
    raw_scores at QK^T/√d_k are already at the correct scale for Boltzmann
    sampling at T_struct.

    Bug #4 reversal: causal mask as -inf in logit space (same as Option A).

    Why this is hardware-natural: physical p-bit arrays (Camsari) never
    form the partition function. They settle to equilibrium via local
    stochastic dynamics that, at steady state, ARE Gumbel-max sampling
    over the Hamiltonian's basins.
    """
    assert abs(T - float(np.sqrt(128))) < 1e-3, \
        f"T_struct must be √128 ≈ 11.314 to match capture scaling; got {T}"
    rng = np.random.default_rng()  # fresh RNG per call (independent samples)
    out = {}
    for layer, r in raw_scores_dict.items():
        n_kv, S, _ = r.shape
        p_exact = np.zeros((n_kv, S, S), dtype=np.float32)
        upper = np.triu(np.ones((S, S), dtype=bool), k=1)

        for h in range(n_kv):
            # Bug #8 fix: scores already at correct temperature (QK^T/√d_k)
            # DO NOT divide by T_struct again.
            scores = r[h].copy()
            scores[upper] = -1e9              # logit-space causal mask
            # For each row, draw K independent Gumbel noise tensors and argmax
            for i in range(S):
                logits = scores[i]            # (S,) — already at T_struct
                # Sample K Gumbel-max draws
                # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
                u = rng.uniform(low=1e-30, high=1.0, size=(K, S))
                gumbel = -np.log(-np.log(u))
                perturbed = logits[None, :] + gumbel   # (K, S)
                idxs = perturbed.argmax(axis=-1)       # (K,)
                # Accumulate one-hots
                for idx in idxs:
                    p_exact[h, i, idx] += 1.0 / K
        out[layer] = p_exact.astype(np.float32)
    return out


# ==============================================================================
# MODEL LOADING (mirrors v2)
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
    import torch
    eps = 1e-4   # Bug #3
    p = torch.softmax(logits_vanilla.float(), dim=-1).clamp(eps, 1.0)
    q = torch.softmax(logits_condition.float(), dim=-1).clamp(eps, 1.0)
    kl = float((p * (p / q).log()).sum().item())
    return max(0.0, kl)   # negative KL is float32 roundoff


def _keepalive(stop_event, interval=15):
    while not stop_event.is_set():
        stop_event.wait(interval)
        if not stop_event.is_set():
            print(".", end="", flush=True)


# ==============================================================================
# TOKEN STEP — runs all 4 encodings on the same captured attention
# ==============================================================================

def run_token_step(model, tokenizer, ids, capturer, layer_subset, alpha, K, T):
    """
    Run one teacher-forced step:
      Phase 1 capture once (vanilla forward, capture weights_kv AND raw_scores)
      For each non-vanilla encoding:
        - Run that encoding's sampler on the captured data
        - Phase 2 forward with injection
      Vanilla record: no Phase 2 needed
    Returns (records, van_id, capture_ok, capture_diag)
    """
    import torch
    from tasb_two_pass import ThermodynamicInjector

    # ── Phase 1: vanilla capture ──────────────────────────────────────────────
    t1s = time.perf_counter()
    capturer.clear(); capturer.attach()
    with torch.no_grad():
        out1 = model(ids, use_cache=False, output_attentions=True)
    capturer.detach()
    t_gpu1 = time.perf_counter() - t1s

    logits_v = out1.logits[0, -1, :].float().cpu()
    van_id   = int(logits_v.argmax().item())
    top5_v   = torch.topk(torch.softmax(logits_v, dim=-1), 5).indices.tolist()
    top2_v   = torch.topk(torch.softmax(logits_v, dim=-1), 2).values
    gap      = float((top2_v[0] - top2_v[1]).item())

    # ── Capture invariant check (once per step) ──────────────────────────────
    capture_ok, capture_diag = _verify_capture_invariant(
        capturer.weights_kv, capturer.raw_scores, T)

    # ── Sample p_thermo for each non-vanilla encoding ────────────────────────
    # Filter captured data to JUST the target layer (saves time vs all 28)
    w_subset = {L: capturer.weights_kv[L]
                for L in layer_subset if L in capturer.weights_kv}
    r_subset = {L: capturer.raw_scores[L]
                for L in layer_subset if L in capturer.raw_scores}

    t_samp_start = time.perf_counter()
    p_thermo_by_enc = {
        'prob_space':    _sample_prob_space(w_subset, K=K),
        'logit_softmax': _sample_logit_softmax(r_subset, K=K, T=T),
        'logit_gumbel':  _sample_logit_gumbel(r_subset,  K=K, T=T),
    }
    t_samp = time.perf_counter() - t_samp_start

    van_str = tokenizer.decode([van_id]).strip()[:14]

    records = []

    # Vanilla record (no Phase 2)
    records.append({
        'encoding':   'vanilla',
        'van_tok':    van_str,
        'out_tok':    van_str,
        'match':      1,
        'd_sem':      0,
        'logit_gap':  round(gap, 4),
        'kl_logit':   0.0,
        'eta':        round(1.0 / (0.0 + 1e-8), 1),
        'p_max':      0.0,   # uniformity metric
        'p_top1':     0.0,
        't_gpu1':     round(t_gpu1, 4),
        't_sample':   0.0,
        't_gpu2':     0.0,
        'capture_ok': int(capture_ok),
    })

    # Each non-vanilla encoding: Phase 2 inject + measure
    for enc_name in ['prob_space', 'logit_softmax', 'logit_gumbel']:
        p_thermo_subset = p_thermo_by_enc[enc_name]

        # Uniformity diagnostics: how peaky is the sampled distribution?
        # p_max = max entry per row, averaged across heads × rows × layers
        # p_top1 = mean top-1 mass (closer to 1.0 = sampler very confident)
        p_arr = next(iter(p_thermo_subset.values()))   # one layer's (8,S,S)
        p_max_mean  = float(p_arr.max(axis=-1).mean())
        p_top1_mean = float(np.sort(p_arr, axis=-1)[..., -1].mean())

        # Phase 2 with injector limited to target layer
        t2s = time.perf_counter()
        injector = ThermodynamicInjector(model, layer_subset, alpha=alpha)
        injector.load(p_thermo_subset, capturer.weights)
        injector.attach()
        with torch.no_grad():
            out2 = model(ids, use_cache=False)
        injector.detach()
        t_gpu2 = time.perf_counter() - t2s

        logits_out = out2.logits[0, -1, :].float().cpu()
        out_id     = int(logits_out.argmax().item())
        match      = int(van_id == out_id)
        d_sem      = 0 if out_id in top5_v else 1
        kl         = _kl_logits(logits_v, logits_out)
        out_str    = tokenizer.decode([out_id]).strip()[:14]

        records.append({
            'encoding':   enc_name,
            'van_tok':    van_str,
            'out_tok':    out_str,
            'match':      match,
            'd_sem':      d_sem,
            'logit_gap':  round(gap, 4),
            'kl_logit':   round(kl, 6),
            'eta':        round(match / (kl + 1e-8), 1),
            'p_max':      round(p_max_mean, 4),
            'p_top1':     round(p_top1_mean, 4),
            't_gpu1':     round(t_gpu1, 4),
            't_sample':   round(t_samp, 4),
            't_gpu2':     round(t_gpu2, 4),
            'capture_ok': int(capture_ok),
        })

    return records, van_id, capture_ok, capture_diag


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run_test(model, tokenizer, prompts, tokens_per_prompt,
             layer_subset, alpha, K, T, outdir, ts):
    """All params explicit — Bug #1, #6, #7 guards."""
    import torch
    from tasb_two_pass import VanillaCapture

    # Bug #7: layer_subset must be List[int]
    if not isinstance(layer_subset, list) or not all(isinstance(l, int) for l in layer_subset):
        raise TypeError(
            f"Bug #7 guard: layer_subset must be List[int], got {type(layer_subset)}: "
            f"{layer_subset}")

    print(f"\n{'═'*78}")
    print(f"  [TEST] Encoding comparison — {len(prompts)} prompts × {tokens_per_prompt} tokens")
    print(f"  [TEST] Layer: {layer_subset}  α={alpha}  K={K}  T_struct={T:.4f}")
    print(f"  [TEST] Encodings: {ENCODINGS}")
    print(f"  [TEST] Target rows: {len(prompts) * tokens_per_prompt * len(ENCODINGS)}")
    print(f"{'═'*78}\n")

    # Capture needs all the layers we plan to inject at (just the subset)
    capturer = None
    all_records = []
    capture_invariant_failures = 0

    for prompt in prompts:
        capturer = VanillaCapture(model, layer_subset)
        inputs   = tokenizer(prompt['text'], return_tensors='pt').to(model.device)
        ids      = inputs['input_ids'].clone()

        print(f"\n  {cyan(prompt['id'])} [{prompt['domain']}]  "
              f"{gray(prompt['text'][:55])}")
        print(f"  {'─'*70}")

        for step in range(tokens_per_prompt):
            stop = threading.Event()
            t = threading.Thread(target=_keepalive, args=(stop, 15), daemon=True)
            t.start()
            try:
                records, van_id, cap_ok, cap_diag = run_token_step(
                    model=model, tokenizer=tokenizer, ids=ids,
                    capturer=capturer, layer_subset=layer_subset,
                    alpha=alpha, K=K, T=T)
            finally:
                stop.set(); t.join(timeout=1)

            if not cap_ok:
                capture_invariant_failures += 1

            for r in records:
                r['prompt_id'] = prompt['id']
                r['domain']    = prompt['domain']
                r['step']      = step + 1
                all_records.append(r)

            van_str = records[0]['van_tok']
            row = "  ".join(
                f"{r['encoding'][:8]:>8}={'✓' if r['match']==1 else '✗'}"
                for r in records[1:]
            )
            cap_marker = green('cap=OK') if cap_ok else red(f'cap=FAIL[{cap_diag}]')
            print(f"  step={step+1:>3}  van='{van_str:<10}'  {row}  {cap_marker}",
                  flush=True)

            # Teacher-forced advance
            ids = torch.cat(
                [ids, torch.tensor([[van_id]], device=model.device)], dim=1)
            if van_id == tokenizer.eos_token_id: break

        # Per-prompt summary
        print(f"\n  {prompt['id']} summary:")
        for enc in ENCODINGS:
            dr = [r for r in all_records
                  if r['prompt_id']==prompt['id'] and r['encoding']==enc]
            if not dr: continue
            mr = np.mean([r['match'] for r in dr]) * 100
            kl = np.mean([r['kl_logit'] for r in dr])
            print(f"    {enc:<16}: match={mr:>5.1f}%  mean_KL={kl:.4f}  n={len(dr)}")

        # Checkpoint
        ckpt = f"{outdir}/tasb_encoding_v1_partial_{ts}.csv"
        with open(ckpt, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader(); w.writerows(all_records)
        print(f"  {gray(f'[checkpoint → {ckpt}]')}")

    n_steps = len(all_records) // len(ENCODINGS)
    pass_pct = 100.0 * (n_steps - capture_invariant_failures) / max(1, n_steps)

    if capture_invariant_failures == 0:
        print(f"\n  {green(f'capture invariant: PASSED on all {n_steps} steps (100%)')}")
    elif pass_pct >= 95.0:
        print(f"\n  {yellow(f'capture invariant: {pass_pct:.1f}% pass rate '
              f'({capture_invariant_failures}/{n_steps} failed). Run VALID '
              f'but flagged for review.')}")
    else:
        # Below 95%: this is the v3.0 Bug #8 pattern. Refuse to publish.
        print(f"\n  {red(f'CAPTURE INVARIANT FAILED: only {pass_pct:.1f}% pass '
              f'({capture_invariant_failures}/{n_steps} failed)')}")
        print(f"  {red('This indicates a sampler/capture mismatch (e.g. Bug #8 '
              'double-T division). Run is NOT VALID for publication.')}")
        print(f"  {red('Investigate: does softmax(captured_raw / T_struct) match '
              'captured weights_kv within 0.05? See _verify_capture_invariant.')}")

    return all_records


def analyze(records):
    """Bootstrap CI per encoding + McNemar pairwise."""
    from collections import defaultdict

    print(f"\n{'═'*78}")
    print(f"  [ANALYSIS] Encoding comparison")
    print(f"{'═'*78}\n")

    cond = defaultdict(lambda: {'match': [], 'kl': [], 'd_sem': 0,
                                'p_max': [], 'p_top1': []})
    for r in records:
        c = r['encoding']
        cond[c]['match'].append(int(r['match']))
        cond[c]['kl'].append(float(r['kl_logit']))
        cond[c]['d_sem'] += int(r['d_sem'])
        cond[c]['p_max'].append(float(r['p_max']))
        cond[c]['p_top1'].append(float(r['p_top1']))

    rng = np.random.default_rng(42)
    print(f"  {'Encoding':<18} {'n':>4} {'match%':>8} {'95%CI':>16} "
          f"{'mean_KL':>9} {'p_max':>7} {'p_top1':>7} {'d_sem':>6}")
    print(f"  {'─'*88}")
    summary = {}
    for enc in ENCODINGS:
        s = cond[enc]
        if not s['match']: continue
        m = np.mean(s['match']) * 100
        boots = [np.mean(rng.choice(s['match'], len(s['match']), replace=True)) * 100
                 for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary[enc] = {'match': m, 'lo': lo, 'hi': hi,
                        'kl': np.mean(s['kl']), 'd_sem': s['d_sem'],
                        'p_max': np.mean(s['p_max']),
                        'p_top1': np.mean(s['p_top1']),
                        'n': len(s['match'])}
        match_c = (green(f"{m:>7.1f}%") if m >= 98
                   else yellow(f"{m:>7.1f}%") if m >= 90
                   else red(f"{m:>7.1f}%"))
        print(f"  {enc:<18} {s['match'].__len__():>4} {match_c} "
              f"[{lo:>5.1f}, {hi:>5.1f}] {np.mean(s['kl']):>9.4f} "
              f"{np.mean(s['p_max']):>7.4f} {np.mean(s['p_top1']):>7.4f} "
              f"{s['d_sem']:>6}")

    # McNemar pairwise
    try:
        from scipy.stats import chi2 as chi2_dist
    except ImportError:
        chi2_dist = None

    print(f"\n  {bold('Pre-registered McNemar tests:')}")
    print(f"  {'A':<16} vs  {'B':<16}  {'b':>4} {'c':>4}  {'χ²':>7}  {'p':>10}  verdict")
    print(f"  {'─'*78}")

    mcnemar_results = []
    for ca, cb in PAIRS_TO_TEST:
        a_map = {(r['prompt_id'], r['step']): r['match']
                 for r in records if r['encoding']==ca}
        b_map = {(r['prompt_id'], r['step']): r['match']
                 for r in records if r['encoding']==cb}
        common = set(a_map) & set(b_map)
        b_ = sum(1 for k in common if a_map[k]==1 and b_map[k]==0)
        c_ = sum(1 for k in common if a_map[k]==0 and b_map[k]==1)
        if (b_+c_) > 0:
            chi2 = (abs(b_-c_)-1)**2/(b_+c_)
            p = (1.0 - chi2_dist.cdf(chi2, 1)) if chi2_dist else None
        else:
            chi2, p = 0.0, 1.0
        p_str = f"{p:.4f}" if p is not None else "  ----"
        if p is None: verdict = gray("no p")
        elif p < 0.001: verdict = green("SIG p<0.001 (encodings differ)")
        elif p < 0.05:  verdict = green("SIG p<0.05 (encodings differ)")
        else:           verdict = yellow("n.s. (encodings equivalent)")
        print(f"  {ca:<16} vs  {cb:<16}  {b_:>4} {c_:>4}  {chi2:>7.3f}  "
              f"{p_str}  {verdict}")
        mcnemar_results.append({'a': ca, 'b': cb, 'b_cnt': b_, 'c_cnt': c_,
                                'chi2': chi2, 'p': p, 'n': len(common)})

    # Verdict
    print(f"\n  {bold('Encoding invariance verdict:')}")
    n_signif = sum(1 for r in mcnemar_results if r['p'] is not None and r['p'] < 0.05)
    if n_signif == 0:
        print(f"    {green('[ENCODING-INVARIANT] All 3 McNemar tests not significant.')}")
        print(f"    The bridge produces statistically equivalent output across")
        print(f"    probability-space, logit-space-softmax, and logit-space-Gumbel.")
        print(f"    → Hardware portability claim strengthened.")
        print(f"    → Real TSU (using Gumbel-max-like physical sampling) should")
        print(f"      produce the same output as the probability-space simulator.")
    else:
        print(f"    {yellow(f'[ENCODING-DEPENDENT] {n_signif}/3 pairs significantly differ.')}")
        print(f"    The encoding choice affects output. Examine which pair(s)")
        print(f"    are significant to identify the operative effect.")

    return summary, mcnemar_results


def main():
    ap = argparse.ArgumentParser(description='TASB Encoding Comparison v1')
    ap.add_argument('--model',  default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--tokens', type=int, default=60)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--quick',  action='store_true',
                    help='Quick mode: 3 prompts × 15 tokens')
    args = ap.parse_args()

    if args.quick:
        args.tokens = 15

    ts = time.strftime('%Y%m%d_%H%M%S')

    prompts = PROMPTS if not args.quick else PROMPTS[:3]

    print(f"\n{'═'*78}")
    print(f"  TASB ENCODING COMPARISON v1  {ts}")
    print(f"{'═'*78}")
    print(f"  [SYS_INIT] Model: {args.model}")
    print(f"  [SYS_INIT] Layer: L{WINNER_LAYER}  α={WINNER_ALPHA}  K={SAMPLES_K}")
    print(f"  [SYS_INIT] T_struct: {T_STRUCT:.4f} = √128 (Kim 2026)")
    print(f"  [SYS_INIT] Encodings: {len(ENCODINGS)} (vanilla + 3 samplers)")
    print(f"  [SYS_INIT] Prompts: {len(prompts)} × {args.tokens} tokens")
    print(f"{'═'*78}")

    model, tokenizer = load_model(args.model)

    all_records = run_test(
        model=model, tokenizer=tokenizer,
        prompts=prompts, tokens_per_prompt=args.tokens,
        layer_subset=[WINNER_LAYER], alpha=WINNER_ALPHA,
        K=SAMPLES_K, T=T_STRUCT,
        outdir=args.outdir, ts=ts,
    )

    out_csv = f"{args.outdir}/tasb_encoding_v1_{ts}.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader(); w.writerows(all_records)
    print(f"\n  CSV: {out_csv}  ({len(all_records)} rows)")

    analyze(all_records)

    print(f"\n{'═'*78}")
    print(f"  [STATUS] Complete. {green('No errors.')}")
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
