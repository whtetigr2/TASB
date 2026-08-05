"""
sampler.py — Boltzmann sampler backends for thermobridge
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver.

Consumes a LayerCapture and produces a per-Q-head p_thermo tensor that the
injector blends with vanilla attention weights.

BACKENDS
--------
exact  — Analytical softmax → K multinomial draws per row → one-hot
         accumulate / K. As K → ∞, p_thermo → softmax. Production path.

gumbel — Gumbel-max trick: add independent Gumbel(0,1) noise K times,
         argmax each, accumulate. Mathematically equivalent to `exact`
         (Maddison et al. 2014). Hardware-natural — how p-bit arrays settle.

rbm    — Gibbs-style RBM sampling on the energy landscape. Research
         backend; retained for comparison. Not the production claim.

thrml  — Extropic THRML block-Gibbs Boltzmann sampler. Requires
         `pip install thrml`. Hardware path: drops in as chip.sample()
         on a real TSU.

INPUT
-----
LayerCapture with q_post_rope (B, n_q, Sq, head_dim), k_post_rope
(B, n_kv, Sk, head_dim), attention_mask (B, 1, Sq, Sk) or None, scaling
float. Sq == Sk during full-sequence reprocessing (prefill); Sq == 1 < Sk
during a single-token KV-cached decode step.

OUTPUT
------
p_thermo: (B, n_q, Sq, Sk) row-stochastic tensor. Same shape as captured
attn_weights — drop-in replacement for the injector.

INVARIANTS
----------
- All backends produce row-stochastic output (rows sum to 1.0 ± float32 eps).
- All backends produce identical shape (B, n_q, Sq, Sk).
- Mask positions (upper-triangle of causal) receive zero mass.
- Score computation is internal to the sampler — double-scaling is
  structurally impossible.
- Mask is applied in logit space (additive HF sentinel ~-3.4e38), never
  used to zero in probability space.
==============================================================================
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from transformers.models.llama.modeling_llama import repeat_kv as _repeat_kv

from thermobridge.capture import LayerCapture


# ── Backend enum ──────────────────────────────────────────────────────────
Backend = Literal['exact', 'gumbel', 'rbm', 'thrml']
VALID_BACKENDS: set[str] = {'exact', 'gumbel', 'rbm', 'thrml'}


@dataclass
class SamplerConfig:
    backend: Backend = 'exact'
    K: int = 10              # samples per row
    rbm_steps: int = 50      # Gibbs steps for rbm backend
    rbm_field: float = 0.0   # external field for rbm
    seed: int | None = None  # for reproducible sampling

    def __post_init__(self):
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {VALID_BACKENDS}, got {self.backend!r}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")


# ── Top-level sample function ─────────────────────────────────────────────

def sample(capture: LayerCapture, config: SamplerConfig) -> torch.Tensor:
    """Sample p_thermo from a LayerCapture.

    Returns a torch.Tensor of shape (B, n_q, Sq, Sk) on the same device as
    the input capture, dtype float32 (caller can cast as needed).
    (2026-07-04 fix: previously derived a single S from logits.shape and
    used it for both query-row count and key-column count, which silently
    broke under KV-cached decode where Sq=1 != Sk. Fixed throughout this
    file to use Sq/Sk separately. Ported from the equivalent fix in
    Active_Dev/TASB/tasb_sampler_v2.py.)

    The output is row-stochastic per Q head — each (B, h, i, :) row sums
    to 1.0 modulo float precision.
    """
    if not isinstance(capture, LayerCapture):
        raise TypeError(
            f"sample() expects a LayerCapture, got {type(capture).__name__}")

    # Compute the per-Q-head scaled logits with mask added (canonical form)
    logits = _compute_logits(capture)
    # logits shape: (B, n_q, Sq, Sk), fp32

    if config.backend == 'exact':
        return _sample_exact(logits, config)
    elif config.backend == 'gumbel':
        return _sample_gumbel(logits, config)
    elif config.backend == 'rbm':
        return _sample_rbm(logits, config)
    elif config.backend == 'thrml':
        from thermobridge.backends.thrml import thrml_sample
        return thrml_sample(
            capture,
            K=config.K,
            seed=config.seed,
            n_warmup=0,
            steps_per_sample=1,
        )
    else:
        # SamplerConfig __post_init__ should have caught this; defensive
        raise ValueError(f"unknown backend: {config.backend}")


# ── Logit construction (shared by all backends) ───────────────────────────

def _compute_logits(capture: LayerCapture) -> torch.Tensor:
    """Compute the per-Q-head scaled logits: Q @ repeat_kv(K).T * scaling + mask.

    This is the canonical form. After this point, all backends agree on
    the input distribution; they differ only in HOW they sample from it.
    Output is fp32 regardless of capture dtype.
    """
    Q = capture.q_post_rope.to(torch.float32)
    K = capture.k_post_rope.to(torch.float32)

    n_q = Q.shape[1]
    n_kv = K.shape[1]
    if n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} not divisible by n_kv={n_kv}")
    kv_groups = n_q // n_kv

    K_rep = _repeat_kv(K, kv_groups)

    # matmul × scaling
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * capture.scaling

    # Additive mask in logit space — HF sentinel ~-3.4e38, never prob-space zeroing
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.to(torch.float32)

    return logits


# ── Exact backend ─────────────────────────────────────────────────────────

def _sample_exact(logits: torch.Tensor,
                  config: SamplerConfig) -> torch.Tensor:
    """K multinomial draws per row from the analytical Boltzmann distribution.

    As K → ∞, the empirical distribution converges to softmax(logits).
    """
    K = config.K
    B, n_q, Sq, Sk = logits.shape

    # Row-wise softmax (canonical Boltzmann at T_struct embedded in scaling)
    probs = torch.softmax(logits, dim=-1)   # (B, n_q, Sq, Sk)

    # Flatten the batch/head/row dims for multinomial: (B*n_q*Sq, Sk)
    probs_2d = probs.reshape(-1, Sk)

    # Defensive: replace any all-zero rows with uniform.
    # Mask positions can produce -inf, which softmax handles, but if an entire
    # row is masked we'd get NaN. Detect and handle.
    row_sums = probs_2d.sum(dim=-1, keepdim=True)
    bad_rows = (row_sums == 0) | torch.isnan(row_sums)
    if bad_rows.any():
        # Replace with delta on position 0 (corresponds to BOS in standard
        # autoregressive setup, never masked)
        probs_2d = torch.where(
            bad_rows.expand_as(probs_2d),
            torch.zeros_like(probs_2d).scatter_(
                -1, torch.zeros(probs_2d.shape[0], 1, dtype=torch.long,
                                device=probs_2d.device), 1.0),
            probs_2d)

    # Renormalize defensively (multinomial requires exact-1 sum within fp32 tol)
    probs_2d = probs_2d / probs_2d.sum(dim=-1, keepdim=True)

    # Set generator for reproducibility if seed provided
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=probs_2d.device)
        generator.manual_seed(config.seed)

    # K multinomial samples per row
    samples = torch.multinomial(
        probs_2d, K, replacement=True, generator=generator)
    # samples shape: (B*n_q*S, K)

    # Accumulate one-hots / K
    p_thermo_2d = torch.zeros_like(probs_2d)
    p_thermo_2d.scatter_add_(
        -1, samples, torch.ones_like(samples, dtype=torch.float32) / K)

    return p_thermo_2d.reshape(B, n_q, Sq, Sk)


# ── Gumbel backend ────────────────────────────────────────────────────────

def _sample_gumbel(logits: torch.Tensor,
                   config: SamplerConfig) -> torch.Tensor:
    """K Gumbel-max samples per row.

    For each of K draws: add independent Gumbel(0,1) noise to logits,
    take argmax. Equivalent to sampling from softmax(logits) but never
    forms the partition function. This is how physical p-bit arrays
    settle to Boltzmann.
    """
    K = config.K
    B, n_q, Sq, Sk = logits.shape

    # Scoped generator — no global RNG leakage across backend/layer calls
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(config.seed)

    # Vectorized when memory allows (S ≤ 256 at K=10 is ~3.3 MB); loop otherwise.
    if Sk <= 256 or K * Sk <= 65536:
        # Vectorized path
        logits_exp = logits.unsqueeze(-2).expand(B, n_q, Sq, K, Sk)
        u = torch.rand(logits_exp.shape, dtype=logits_exp.dtype,
                       device=logits_exp.device, generator=generator).clamp(min=1e-30)
        gumbel = -torch.log(-torch.log(u))
        perturbed = logits_exp + gumbel
        idxs = perturbed.argmax(dim=-1)  # (B, n_q, Sq, K)
        p_thermo = torch.zeros(B, n_q, Sq, Sk, dtype=torch.float32,
                                device=logits.device)
        p_thermo_flat = p_thermo.reshape(-1, Sk)
        idxs_flat = idxs.reshape(-1, K)
        ones = torch.ones_like(idxs_flat, dtype=torch.float32) / K
        p_thermo_flat.scatter_add_(-1, idxs_flat, ones)
        return p_thermo_flat.reshape(B, n_q, Sq, Sk)
    else:
        # Loop over K to bound memory
        p_thermo = torch.zeros(B, n_q, Sq, Sk, dtype=torch.float32,
                                device=logits.device)
        for _ in range(K):
            u = torch.rand(logits.shape, dtype=logits.dtype,
                           device=logits.device, generator=generator).clamp(min=1e-30)
            gumbel = -torch.log(-torch.log(u))
            idxs = (logits + gumbel).argmax(dim=-1, keepdim=True)
            p_thermo.scatter_add_(-1, idxs, torch.ones_like(
                idxs, dtype=torch.float32) / K)
        return p_thermo


# ── RBM backend ───────────────────────────────────────────────────────────

def _sample_rbm(logits: torch.Tensor,
                config: SamplerConfig) -> torch.Tensor:
    """Gibbs-style Metropolis relaxation on the energy landscape.

    Energy E_i = -logit_i. At equilibrium, p_i ∝ exp(-E_i) = exp(logit_i),
    matching softmax. Runs a single-site Metropolis Markov chain of
    `rbm_steps` local updates per sample (uniform proposal over positions;
    Metropolis acceptance min(1, exp(logit_proposal - logit_current))), then
    takes the final chain state as one draw. Repeated K times per row (K
    independent chains), accumulated. This is a genuinely different sampling
    PATH than `exact`'s direct multinomial draw — a finite-step stochastic
    relaxation process, not an instantaneous analytical draw — which is the
    point: at rbm_steps -> infinity this chain's stationary distribution is
    exactly softmax(logits) (standard Metropolis-Hastings detailed-balance
    result for a symmetric/uniform proposal), so agreement with
    `exact`/`gumbel` at large rbm_steps is a genuine, independent cross-check
    that a physically-realizable relaxation process reaches the
    analytically-predicted equilibrium — not a restatement of it. At small
    `rbm_steps` relative to S, imperfect mixing (and a resulting KL gap
    against `exact`/`gumbel`) is expected, not a bug — this backend is
    explicitly slower-but-independent, per the module docstring.

    NOTE (2026-07-03): prior to this fix, this function ignored `rbm_steps`
    entirely and fell through to the same softmax+multinomial draw as
    `_sample_exact` whenever `rbm_field == 0.0` (the default, and what
    `per_head_fidelity.py`'s T1.C test uses) — producing byte-identical
    output to `exact` rather than an independent sample. Found by
    [[FIND-035-tasb-validation-metrics-scrub]] in the SCIN Ecosystem vault.

    Each chain is initialized at position 0, which is never masked under
    this file's standing causal-mask assumption (same convention
    `_sample_exact` uses for its degenerate-row fallback). Given that valid
    start, a transition into any masked position (logit ~ -3.4e38) has
    Metropolis acceptance probability that underflows to exactly 0.0 in
    float32, so the chain provably never visits a masked position — masking
    is respected exactly, not just approximately.

    This is a research backend. Production uses `exact` or `gumbel`.
    """
    K = config.K
    steps = config.rbm_steps
    B, n_q, Sq, Sk = logits.shape
    device = logits.device

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)

    logits_2d = logits.reshape(-1, Sk)  # (N, Sk), N = B*n_q*Sq

    if config.rbm_field != 0.0:
        # Additive external-field bias in logit space (equivalent to the
        # previous probability-space reweighting: exp(logit)*exp(field*offset)
        # == exp(logit + field*offset)), now applied before the relaxation
        # chain runs rather than as a separate post-hoc step.
        offset = torch.arange(Sk, dtype=torch.float32, device=device) / Sk
        logits_2d = logits_2d + config.rbm_field * offset

    N = logits_2d.shape[0]

    # Every chain starts at position 0 — never masked under the causal-mask
    # assumption this file makes throughout (mirrors _sample_exact's
    # degenerate-row fallback, which uses the same position for the same
    # reason).
    state = torch.zeros(N, K, dtype=torch.long, device=device)

    for _ in range(steps):
        proposal = torch.randint(0, Sk, (N, K), generator=generator, device=device)

        cur_logit = torch.gather(logits_2d, 1, state)
        prop_logit = torch.gather(logits_2d, 1, proposal)

        # Symmetric/uniform proposal -> Metropolis acceptance reduces to the
        # target-density ratio. Clamp the exponent so a masked-position
        # comparison overflows cleanly to inf (then clamps to 1.0) instead
        # of relying on unclamped exp() behavior at extreme magnitudes.
        accept_prob = torch.exp(torch.clamp(prop_logit - cur_logit, max=80.0))
        accept_prob = torch.clamp(accept_prob, max=1.0)

        u = torch.rand(N, K, generator=generator, device=device)
        accept = u < accept_prob
        state = torch.where(accept, proposal, state)

    p_thermo_2d = torch.zeros_like(logits_2d)
    p_thermo_2d.scatter_add_(
        -1, state, torch.ones_like(state, dtype=torch.float32) / K)
    return p_thermo_2d.reshape(B, n_q, Sq, Sk)


# ── Self-test on import ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("sampler.py self-checks:")
    print(f"  VALID_BACKENDS: {sorted(VALID_BACKENDS)}")
    print(f"  default config: {SamplerConfig()}")

    B, n_q, n_kv, S, head_dim = 1, 4, 2, 6, 8
    Q = torch.randn(B, n_q, S, head_dim, dtype=torch.float32)
    K = torch.randn(B, n_kv, S, head_dim, dtype=torch.float32)
    mask = torch.zeros(B, 1, S, S, dtype=torch.float32)
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(causal, -3.39e38)

    cap = LayerCapture(
        layer_idx=0,
        q_post_rope=Q, k_post_rope=K, attention_mask=mask,
        scaling=1.0/math.sqrt(head_dim),
        attn_weights=torch.zeros(B, n_q, S, S),
        seq_len=S, dtype=torch.float32,
    )

    for backend in ['exact', 'gumbel', 'rbm']:
        cfg = SamplerConfig(backend=backend, K=100, seed=42)
        p = sample(cap, cfg)
        assert p.shape == (B, n_q, S, S), f"shape mismatch for {backend}: {p.shape}"
        row_sums = p.sum(dim=-1)
        causal_b = causal.unsqueeze(0).unsqueeze(0).expand(B, n_q, S, S)
        upper_mass = p[causal_b].sum().item()
        valid_rows = ~causal_b.all(dim=-1).reshape(B, n_q, S)
        rs_valid = row_sums[valid_rows]
        max_dev = (rs_valid - 1.0).abs().max().item()
        print(f"  {backend:>6}: shape OK, row sum max dev {max_dev:.2e}, "
              f"upper mass {upper_mass:.2e}")
        assert max_dev < 1e-5, f"{backend} not row-stochastic"
        assert upper_mass < 1e-10, (
            f"{backend} put mass on masked positions: {upper_mass}")

    print("  All backends produce row-stochastic, mask-respecting output")
