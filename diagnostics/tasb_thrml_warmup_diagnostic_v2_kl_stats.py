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

def kl_to_analytic(p_emp: jnp.ndarray, p_analytic: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Per-row KL(p_analytic || p_emp), using log_softmax-style stabilization
    (bug registry #3/#10: no eps-clamp on the analytic side, which is exact;
    eps only guards log(0) on the empirical side for cells with zero counts).
    Returns shape (S,).
    """
    log_p_emp = jnp.log(p_emp + eps)
    return jnp.sum(p_analytic * (jnp.log(p_analytic + eps) - log_p_emp), axis=-1)


def main():
    print("=" * 78)
    print("TASB THRML warmup/steps_per_sample diagnostic")
    print("=" * 78)
    print()
    print(f"  JAX devices: {jax.devices()}")
    print()

    S = 60
    K = 500  # larger K to tighten MC noise so the KL comparison is meaningful
    SEED = 42
    N_TRIALS = 5  # multiple seeds to characterize MC noise floor itself

    J = build_test_J(S=S, seed=0)
    p_analytic = jax.nn.softmax(J, axis=-1)

    print(f"  S={S}, K={K}, base seed={SEED}, n_trials={N_TRIALS}")
    print()

    configs = {
        "CURRENT ": dict(n_warmup=50, steps_per_sample=2),  # 50 + 499*2 = 1048 steps
        "PROPOSED": dict(n_warmup=0, steps_per_sample=1),   # 500 steps
    }

    # For each config, run N_TRIALS independent seeds. Track per-trial
    # mean-KL-to-analytic (across rows) and wall-clock.
    summary = {}
    for label, cfg in configs.items():
        print(f"  Running: {label} {cfg} ...")
        kls, walls = [], []
        for trial in range(N_TRIALS):
            p_emp, wall = run_thrml(J, K=K, seed=SEED + trial, **cfg)
            kl_rows = kl_to_analytic(p_emp, p_analytic)
            kls.append(float(jnp.mean(kl_rows)))
            walls.append(wall)
            print(f"    trial {trial}: mean KL-to-analytic = {kls[-1]:.5f}, "
                  f"wall = {wall*1000:.1f} ms")
        summary[label] = dict(kl_mean=float(np.mean(kls)), kl_std=float(np.std(kls)),
                               wall_mean=float(np.mean(walls)), wall_std=float(np.std(walls)))
        print()

    print("-" * 78)
    print("SUMMARY")
    print("-" * 78)
    for label, s in summary.items():
        print(f"  {label}: KL-to-analytic = {s['kl_mean']:.5f} +/- {s['kl_std']:.5f}   "
              f"wall = {s['wall_mean']*1000:.1f} +/- {s['wall_std']*1000:.1f} ms")
    print()

    kl_a, kl_b = summary["CURRENT "]["kl_mean"], summary["PROPOSED"]["kl_mean"]
    wall_a, wall_b = summary["CURRENT "]["wall_mean"], summary["PROPOSED"]["wall_mean"]
    speedup = wall_a / wall_b if wall_b > 0 else float("nan")

    # Compare KL means against the spread (std) observed across trials within
    # each config. If the two configs' KL means are within ~1-2 std of each
    # other (using the larger of the two stds as the yardstick), that's
    # consistent with "both configs are sampling the same distribution,
    # difference is just seed-to-seed MC noise" -- NOT a max-cell-diff
    # heuristic, but a same-units (KL) comparison against its own noise floor.
    pooled_std = max(summary["CURRENT "]["kl_std"], summary["PROPOSED"]["kl_std"], 1e-12)
    kl_diff = abs(kl_a - kl_b)
    z_like = kl_diff / pooled_std

    print(f"  |KL_current - KL_proposed| = {kl_diff:.5f}")
    print(f"  trial-to-trial KL std (max of the two configs) = {pooled_std:.5f}")
    print(f"  ratio = {z_like:.2f}  (values ~O(1) or less => consistent with same")
    print(f"          underlying distribution, difference within seed noise)")
    print()

    if z_like < 2.0:
        print(f"  [CONSISTENT] KL-to-analytic for both configs agrees within")
        print(f"               trial-to-trial noise. Supports the hypothesis:")
        print(f"               extra warmup/steps are i.i.d. draws from the")
        print(f"               same softmax(J), not chain-mixing work.")
    else:
        print(f"  [FLAG] KL-to-analytic differs by {z_like:.1f}x the trial-to-trial")
        print(f"         noise floor. Worth a closer look before concluding the")
        print(f"         configs are equivalent.")
    print()

    print(f"  speedup (CURRENT / PROPOSED), {1048}/{500} = {1048/500:.2f}x steps:")
    print(f"    wall-clock speedup = {speedup:.2f}x")
    if speedup > 1.5:
        print(f"  [SPEEDUP] PROPOSED config is {speedup:.2f}x faster at K=500.")
    else:
        print(f"  [NO SPEEDUP] PROPOSED config is only {speedup:.2f}x faster.")
        print(f"             Gibbs-step count may not be the dominant cost at")
        print(f"             this scale -- fixed per-call overhead may dominate.")
    print()
    print("  NOTE: this run is on CPU (JAX fell back -- see CUDA error above")
    print("  if present). Speedup ratio on GPU with the live runtime's 24")
    print("  heads x real model may differ; re-run there once CUDA is fixed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
