"""
tasb_thrml_construction_diag_v1.py
==============================================================================
DIAGNOSTIC: Fresh-per-call vs Pre-built program path for THRML

PURPOSE:
    The live runtime currently rebuilds CategoricalNode×S, Block,
    CategoricalEBMFactor, CategoricalGibbsConditional, BlockGibbsSpec,
    FactorSamplingProgram, and SamplingSchedule from scratch for every head
    of every token. Per-token timing data (tasb_llama32_chat_runtime_v2)
    showed ~22s/call at S=37-77, flat (S-independent) — meaning the cost
    is NOT JAX shape-recompilation but fixed per-call construction overhead.

    This diagnostic tests whether building the program ONCE and swapping
    weights via eqx.tree_at (which doesn't trigger JAX retracing) brings
    per-call time down to the ~1s/call range seen in the standalone
    diagnostic (tasb_thrml_warmup_diagnostic_v2_kl_stats.py).

WHAT IT MEASURES:
    PATH A (CURRENT): Fresh CategoricalNode/Block/Factor/Program per call
                      (mirrors live runtime's thrml_sample() per-head loop)
    PATH B (PROPOSED): Build program once, swap weights via eqx.tree_at

    Both paths run 24 calls (= 1 token's worth of heads) at realistic S,
    with the same K=50, n_warmup=0, steps_per_sample=1 as the patched
    live runtime. Faithfulness check: compare p_thermo from both paths
    against analytic softmax(J) via mean KL.

EXPECTED RESULT:
    PATH A: ~22s × 24 = long (reproduces live runtime behavior)
    PATH B: construction × 1 + sample_states × 24 (amortized JIT)
            IF JIT cache stays hot across 24 calls: ~1-2s total
            IF JIT retraces each call regardless: same as PATH A

USAGE:
    stdbuf -oL -eL python tasb_thrml_construction_diag_v1.py | tee -a thrml_construction_diag.log
==============================================================================
"""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False
    warnings.warn("equinox not installed — PATH B will be skipped. pip install equinox")

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
# Helpers
# ---------------------------------------------------------------------------

def build_test_J(S: int, n_heads: int, seed: int = 0) -> jnp.ndarray:
    """
    Build (n_heads, S, S) batch of realistic attention logit matrices:
    causal mask, mixed sharp/diffuse rows.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n_heads, S, S)).astype(np.float32)
    for h in range(n_heads):
        for i in range(S):
            if i % 3 == 0 and i > 0:
                j = rng.integers(0, i)
                base[h, i, j] += 8.0
    J = jnp.asarray(base)
    causal = jnp.triu(jnp.ones((S, S), dtype=bool), k=1)
    J = jnp.where(causal[None], jnp.finfo(jnp.float32).min, J)
    return J


def kl_to_analytic(p_emp: jnp.ndarray, p_analytic: jnp.ndarray,
                   eps: float = 1e-8) -> float:
    log_p = jnp.log(p_emp + eps)
    kl = jnp.sum(p_analytic * (jnp.log(p_analytic + eps) - log_p), axis=-1)
    return float(jnp.mean(kl))


# ---------------------------------------------------------------------------
# PATH A: Fresh construction per call (current live runtime behavior)
# ---------------------------------------------------------------------------

def path_a_single_call(J_bh: jnp.ndarray, K: int, key: jax.Array,
                       S: int) -> tuple[jnp.ndarray, float, float]:
    """
    Build all objects fresh, run sample_states.
    Returns (p_empirical, construction_time, sample_time).
    """
    t0 = time.perf_counter()
    nodes      = [CategoricalNode() for _ in range(S)]
    free_block = Block(nodes)
    factor     = CategoricalEBMFactor([free_block], J_bh)
    cond       = CategoricalGibbsConditional(S)
    spec       = BlockGibbsSpec(free_super_blocks=[free_block], clamped_blocks=[])
    program    = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[cond],
        factors=[factor], other_interaction_groups=[],
    )
    schedule   = SamplingSchedule(n_warmup=0, n_samples=K, steps_per_sample=1)
    t_construct = time.perf_counter() - t0

    key, sk1, sk2 = jax.random.split(key, 3)
    init = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]

    t0 = time.perf_counter()
    samples = sample_states(sk2, program, schedule, init, [], [free_block])
    samples[0].block_until_ready()
    t_sample = time.perf_counter() - t0

    one_hot = jax.nn.one_hot(samples[0].astype(jnp.int32), S, dtype=jnp.float32)
    p_emp   = one_hot.mean(axis=0)
    return p_emp, t_construct, t_sample


# ---------------------------------------------------------------------------
# PATH B: Pre-built program, weight swap via eqx.tree_at
# ---------------------------------------------------------------------------

def build_program_once(S: int, K: int) -> tuple:
    """
    Build all structural objects once for a given S.
    Returns (program, schedule, free_block).
    """
    nodes      = [CategoricalNode() for _ in range(S)]
    free_block = Block(nodes)
    J_dummy    = jnp.zeros((S, S))
    factor     = CategoricalEBMFactor([free_block], J_dummy)
    cond       = CategoricalGibbsConditional(S)
    spec       = BlockGibbsSpec(free_super_blocks=[free_block], clamped_blocks=[])
    program    = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[cond],
        factors=[factor], other_interaction_groups=[],
    )
    schedule   = SamplingSchedule(n_warmup=0, n_samples=K, steps_per_sample=1)
    return program, schedule, free_block


def path_b_single_call(program, schedule, free_block,
                       J_bh: jnp.ndarray, key: jax.Array,
                       S: int) -> tuple[jnp.ndarray, float, float]:
    """
    Swap weights via eqx.tree_at, run sample_states with pre-built program.
    Returns (p_empirical, swap_time, sample_time).
    """
    W_new = J_bh[:, None, :]  # (S, 1, S) — internal shape

    t0 = time.perf_counter()
    updated_program = eqx.tree_at(
        lambda p: p.per_block_interactions[0][0].weights,
        program, W_new
    )
    t_swap = time.perf_counter() - t0

    key, sk1, sk2 = jax.random.split(key, 3)
    init = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]

    t0 = time.perf_counter()
    samples = sample_states(sk2, updated_program, schedule, init, [], [free_block])
    samples[0].block_until_ready()
    t_sample = time.perf_counter() - t0

    one_hot = jax.nn.one_hot(samples[0].astype(jnp.int32), S, dtype=jnp.float32)
    p_emp   = one_hot.mean(axis=0)
    return p_emp, t_swap, t_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("TASB THRML construction overhead diagnostic")
    print("=" * 78)
    print()
    print(f"  JAX devices: {jax.devices()}")
    print(f"  equinox available: {EQX_AVAILABLE}")
    print()

    S       = 60    # realistic seq_len from timing data
    N_HEADS = 24    # LLaMA 3.2-3B n_q heads
    K       = 50    # matches live runtime default
    SEED    = 42

    J_all    = build_test_J(S=S, n_heads=N_HEADS, seed=SEED)
    p_analytic = jax.nn.softmax(J_all, axis=-1)  # (24, S, S)
    key      = jax.random.key(SEED)

    # ── PATH A ───────────────────────────────────────────────────────────────
    print(f"PATH A: Fresh construction per call ({N_HEADS} heads, S={S}, K={K})")
    print(f"        [mirrors current live runtime thrml_sample() per-head loop]")
    print()

    # One throwaway call to warm JIT before timing
    print("  Warming JIT (1 throwaway call)...")
    key, sk = jax.random.split(key)
    _, _, _ = path_a_single_call(J_all[0], K, sk, S)
    print("  Done. Timing 24 calls...")
    print()

    a_construct_times, a_sample_times, a_kls = [], [], []
    for h in range(N_HEADS):
        key, sk = jax.random.split(key)
        p_emp, t_c, t_s = path_a_single_call(J_all[h], K, sk, S)
        kl = kl_to_analytic(p_emp, p_analytic[h])
        a_construct_times.append(t_c)
        a_sample_times.append(t_s)
        a_kls.append(kl)
        print(f"  head {h:2d}: construct={t_c*1000:6.1f}ms  "
              f"sample={t_s*1000:7.1f}ms  "
              f"total={(t_c+t_s)*1000:7.1f}ms  kl={kl:.5f}")

    a_total = sum(a_construct_times) + sum(a_sample_times)
    print()
    print(f"  PATH A TOTAL (24 heads): {a_total:.2f}s")
    print(f"    construct: {sum(a_construct_times):.3f}s  "
          f"sample: {sum(a_sample_times):.3f}s")
    print(f"    mean KL-to-analytic: {np.mean(a_kls):.5f}")
    print()

    if not EQX_AVAILABLE:
        print("PATH B skipped — equinox not installed.")
        return

    # ── PATH B ───────────────────────────────────────────────────────────────
    print(f"PATH B: Pre-built program + eqx.tree_at weight swap")
    print(f"        [proposed: build once per S, swap J per head per token]")
    print()

    print("  Building program once...")
    t0 = time.perf_counter()
    program, schedule, free_block = build_program_once(S=S, K=K)
    t_build = time.perf_counter() - t0
    print(f"  One-time build: {t_build*1000:.1f}ms")

    # Warm JIT with one throwaway call on the pre-built program
    print("  Warming JIT (1 throwaway call on pre-built program)...")
    key, sk = jax.random.split(key)
    _, _, _ = path_b_single_call(program, schedule, free_block, J_all[0], sk, S)
    print("  Done. Timing 24 calls...")
    print()

    b_swap_times, b_sample_times, b_kls = [], [], []
    for h in range(N_HEADS):
        key, sk = jax.random.split(key)
        p_emp, t_sw, t_s = path_b_single_call(
            program, schedule, free_block, J_all[h], sk, S)
        kl = kl_to_analytic(p_emp, p_analytic[h])
        b_swap_times.append(t_sw)
        b_sample_times.append(t_s)
        b_kls.append(kl)
        print(f"  head {h:2d}: swap={t_sw*1000:5.2f}ms  "
              f"sample={t_s*1000:7.1f}ms  "
              f"total={(t_sw+t_s)*1000:7.1f}ms  kl={kl:.5f}")

    b_total = t_build + sum(b_swap_times) + sum(b_sample_times)
    print()
    print(f"  PATH B TOTAL (build + 24 heads): {b_total:.2f}s")
    print(f"    one-time build: {t_build:.3f}s  "
          f"swap: {sum(b_swap_times):.3f}s  "
          f"sample: {sum(b_sample_times):.3f}s")
    print(f"    mean KL-to-analytic: {np.mean(b_kls):.5f}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    speedup = a_total / b_total if b_total > 0 else float("nan")
    kl_diff = abs(np.mean(a_kls) - np.mean(b_kls))
    print(f"  PATH A total: {a_total:.2f}s  |  PATH B total: {b_total:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  KL difference (A vs B): {kl_diff:.6f}  "
          f"({'consistent' if kl_diff < 0.005 else 'FLAG — check faithfulness'})")
    print()
    if speedup > 5:
        print("  [STRONG SIGNAL] Pre-built program dramatically faster.")
        print("  Construction overhead is the dominant cost. The fix is to")
        print("  hoist program building outside the per-head loop in")
        print("  thrml_sample(), caching per S value.")
    elif speedup > 1.5:
        print("  [MODERATE SIGNAL] Pre-built program faster but not dramatic.")
        print("  Construction is a significant fraction but not the sole cost.")
    else:
        print("  [WEAK/NO SIGNAL] Pre-built program not meaningfully faster.")
        print("  Construction overhead is not the bottleneck — look elsewhere.")
    print("=" * 78)


if __name__ == "__main__":
    main()
