"""
tasb_thrml_warmup_diagnostic_v1.py — Falsifying diagnostic for n_warmup/steps_per_sample
==============================================================================
PURPOSE:
    TASB's CategoricalEBMFactor for attention has theta[i,:] = J[i,:], with
    NO dependence on the current state of any node (single-site field term,
    no pairwise interactions). If true, every Gibbs step is an i.i.d. draw
    from softmax(J[i,:]) regardless of init_state or step number — meaning
    n_warmup=50 + steps_per_sample=2 (148 total Gibbs steps per K=50 samples)
    should be statistically indistinguishable from n_warmup=0,
    steps_per_sample=1 (50 total Gibbs steps), just slower.

WHAT THIS SCRIPT DOES:
    1. Builds a realistic-sized J matrix (S=60, matching M5/M7 seq lengths)
       with structure similar to real attention logits (causal mask +
       a mix of sharp and diffuse rows).
    2. Calls thrml_sample-equivalent logic directly (no model load needed —
       this tests the THRML Gibbs machinery in isolation, not the full
       capture/inject pipeline) with:
         (a) CURRENT config:  n_warmup=50, steps_per_sample=2  (148 steps)
         (b) PROPOSED config: n_warmup=0,  steps_per_sample=1  (50 steps)
       using the SAME seed for both.
    3. Compares:
         - p_thermo (a) vs p_thermo (b): should agree within MC noise
         - wall-clock (a) vs (b): should drop ~3x if warmup/extra-steps
           were pure overhead (148 -> 50 steps)
         - both vs analytic softmax(J): sanity check that both configs
           are actually converging to the right distribution

NOTE ON SCOPE:
    This is a standalone diagnostic using thrml's primitives directly
    (same primitives tasb_sampler_thrml.py uses), built from a single J
    matrix rather than a live model capture. This isolates the Gibbs-
    sampling-config question from the per-token-reconstruction/JIT
    hypothesis (which requires the live runtime loop to test). If this
    script shows (a) ~= (b) in distribution but NOT faster, that's still
    useful — it means the per-step overhead is dominated by something
    OTHER than Gibbs step count (pointing back at JIT/reconstruction).

USAGE:
    stdbuf -oL -eL python3 tasb_thrml_warmup_diagnostic_v1.py | tee -a thrml_warmup_diag.log
==============================================================================
"""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from thrml import (
    CategoricalNode,
    Block,
    BlockGibbsSpec,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states,
)
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
)


# ---------------------------------------------------------------------------
# Build a realistic-ish J matrix (S=60, causal mask, mixed sharpness)
# ---------------------------------------------------------------------------

def build_test_J(S: int = 60, seed: int = 0) -> jnp.ndarray:
    """
    Build an (S,S) logit matrix resembling post-RoPE attention scores:
    - causal mask (upper triangle -> -inf)
    - row-dependent sharpness: some rows near-uniform, some near-deterministic
    - random base logits scaled to plausible attention-logit magnitude (~O(5))
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=1.0, size=(S, S)).astype(np.float32)

    # Make some rows sharp (one dominant key), some diffuse (near-uniform)
    for i in range(S):
        if i % 3 == 0 and i > 0:
            # sharp row: boost a random earlier position
            j = rng.integers(0, i)
            base[i, j] += 8.0
        else:
            base[i, : i + 1] *= 0.5  # diffuse-ish

    J = jnp.asarray(base)

    # causal mask: position i can only attend to j <= i
    causal = jnp.triu(jnp.ones((S, S), dtype=bool), k=1)
    J = jnp.where(causal, jnp.finfo(jnp.float32).min, J)

    return J


# ---------------------------------------------------------------------------
# Single-call wrapper mirroring tasb_sampler_thrml.thrml_sample's THRML usage
# ---------------------------------------------------------------------------

def run_thrml(J: jnp.ndarray, K: int, seed: int, n_warmup: int,
               steps_per_sample: int) -> tuple[jnp.ndarray, float]:
    """
    Returns (p_empirical, wall_clock_seconds).
    p_empirical: (S, S) row-stochastic empirical distribution from K draws.
    """
    S = J.shape[0]

    nodes = [CategoricalNode() for _ in range(S)]
    free_block = Block(nodes)

    factor = CategoricalEBMFactor([free_block], J)
    conditional = CategoricalGibbsConditional(S)

    gibbs_spec = BlockGibbsSpec(
        free_super_blocks=[free_block],
        clamped_blocks=[],
    )
    program = FactorSamplingProgram(
        gibbs_spec=gibbs_spec,
        samplers=[conditional],
        factors=[factor],
        other_interaction_groups=[],
    )

    key = jax.random.key(seed)
    key, sk1, sk2 = jax.random.split(key, 3)
    init_state = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]
    schedule = SamplingSchedule(
        n_warmup=n_warmup, n_samples=K, steps_per_sample=steps_per_sample,
    )

    # First call includes JIT compile time; do one throwaway call to warm
    # the JIT cache so the timed call reflects steady-state cost, matching
    # how the live runtime would behave after the first token.
    _ = sample_states(sk2, program, schedule, init_state, [], [free_block])

    key, sk1b, sk2b = jax.random.split(key, 3)
    init_state_b = [jax.random.randint(sk1b, (S,), 0, S, dtype=jnp.uint8)]

    t0 = time.perf_counter()
    samples = sample_states(sk2b, program, schedule, init_state_b, [], [free_block])
    samples[0].block_until_ready()
    t1 = time.perf_counter()

    one_hot = jax.nn.one_hot(samples[0].astype(jnp.int32), S, dtype=jnp.float32)
    p_empirical = one_hot.mean(axis=0)

    return p_empirical, (t1 - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("TASB THRML warmup/steps_per_sample diagnostic")
    print("=" * 78)
    print()
    print(f"  JAX devices: {jax.devices()}")
    print()

    S = 60
    K = 50
    SEED = 42

    J = build_test_J(S=S, seed=0)
    p_analytic = jax.nn.softmax(J, axis=-1)

    print(f"  S={S}, K={K}, seed={SEED}")
    print()

    configs = {
        "CURRENT  (n_warmup=50, steps_per_sample=2, 148 total steps)":
            dict(n_warmup=50, steps_per_sample=2),
        "PROPOSED (n_warmup=0,  steps_per_sample=1, 50 total steps)":
            dict(n_warmup=0, steps_per_sample=1),
    }

    results = {}
    for label, cfg in configs.items():
        print(f"  Running: {label} ...")
        p_emp, wall = run_thrml(J, K=K, seed=SEED, **cfg)
        results[label] = (p_emp, wall)
        max_err_vs_analytic = float(jnp.abs(p_emp - p_analytic).max())
        print(f"    wall-clock (steady-state, post-JIT): {wall*1000:.2f} ms")
        print(f"    max|empirical - softmax(J)|:         {max_err_vs_analytic:.4f}")
        print()

    (label_a, (p_a, wall_a)), (label_b, (p_b, wall_b)) = list(results.items())

    max_diff_ab = float(jnp.abs(p_a - p_b).max())
    speedup = wall_a / wall_b if wall_b > 0 else float("nan")

    print("-" * 78)
    print("COMPARISON")
    print("-" * 78)
    print(f"  max|p_a - p_b| (CURRENT vs PROPOSED):     {max_diff_ab:.4f}")
    print(f"  wall-clock CURRENT:                       {wall_a*1000:.2f} ms")
    print(f"  wall-clock PROPOSED:                      {wall_b*1000:.2f} ms")
    print(f"  speedup (CURRENT / PROPOSED):             {speedup:.2f}x")
    print()

    # Rough MC-noise expectation: for K=50 draws from a categorical with up
    # to S=60 outcomes, per-cell empirical proportions have stdev up to
    # ~sqrt(p(1-p)/K) <= sqrt(0.25/50) ~= 0.07. Use a generous multiple of
    # that as a "looks like MC noise" threshold.
    mc_noise_threshold = 0.20

    if max_diff_ab < mc_noise_threshold:
        print(f"  [PASS-ish] max diff ({max_diff_ab:.4f}) is within rough MC-noise")
        print(f"             range (<{mc_noise_threshold}). Consistent with the")
        print(f"             hypothesis that warmup/extra steps are i.i.d. draws")
        print(f"             from the same distribution, not chain mixing.")
    else:
        print(f"  [FLAG] max diff ({max_diff_ab:.4f}) exceeds rough MC-noise")
        print(f"         threshold ({mc_noise_threshold}). Either K=50 is too")
        print(f"         small for this S to bound MC noise this tightly, or")
        print(f"         the i.i.d.-draws hypothesis needs re-examination.")
    print()

    if speedup > 1.5:
        print(f"  [SPEEDUP CONFIRMED] PROPOSED config is {speedup:.2f}x faster.")
        print(f"             Consistent with warmup/steps being pure overhead.")
    else:
        print(f"  [NO SPEEDUP] PROPOSED config is only {speedup:.2f}x faster.")
        print(f"             Gibbs-step count is likely NOT the dominant cost.")
        print(f"             Points back toward the per-token JIT/reconstruction")
        print(f"             hypothesis from last night as the throughput driver.")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
