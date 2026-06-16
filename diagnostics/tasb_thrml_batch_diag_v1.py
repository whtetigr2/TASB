"""
tasb_thrml_batch_diag_v1.py
==============================================================================
DIAGNOSTIC: Batched sampling approaches vs current sequential per-head loop

CONTEXT:
    From construction_diag_v1: sample_states takes ~1-1.5s per head on T4,
    24 sequential calls = 24-36s/token. Construction overhead is ~3.5ms
    total (negligible). The GPU sits idle between sequential calls.

    Fix direction: get all 24 heads into ONE GPU dispatch.

THREE PATHS TESTED:
    A) CURRENT: sequential loop, fresh construction per head
       (baseline, mirrors live runtime)

    B) VMAP: jax.vmap over heads on a pre-built program
       If sample_states is vmap-compatible, JAX fuses 24 calls into one
       vectorized dispatch. Expected: same or better wall-clock than A.

    C) DIRECT: bypass sample_states entirely, use jax.random.categorical
       directly on J. Mathematically equivalent (proven: theta[i,:] = J[i,:],
       i.i.d. draws, no mixing needed). One vectorized call over (n_q, S, K).
       Expected: microseconds. Confirms theoretical speed ceiling.
       Tradeoff: bypasses THRML substrate, not valid for hardware handoff.
       Use as diagnostic only.

FAITHFULNESS CHECK:
    All paths compared against analytic softmax(J) via per-row KL.
    NOTE: KL computed with masked-position handling to avoid eps-inflation
    bug (bug registry #3/#10) — only non-masked rows included in mean.

USAGE:
    stdbuf -oL -eL python tasb_thrml_batch_diag_v1.py | tee -a thrml_batch_diag.log
==============================================================================
"""

import time
import warnings
import numpy as np

import jax
import jax.numpy as jnp

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False

from thrml import (
    CategoricalNode, Block, BlockGibbsSpec,
    FactorSamplingProgram, SamplingSchedule, sample_states,
)
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor, CategoricalGibbsConditional,
)


# ---------------------------------------------------------------------------
# J construction — realistic attention logits
# ---------------------------------------------------------------------------

def build_test_J(S: int, n_heads: int, seed: int = 0) -> jnp.ndarray:
    """
    Returns (n_heads, S, S) attention logit matrix.
    Uses large-negative (not finfo.min) for causal mask to avoid
    softmax numerical edge cases in KL computation.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n_heads, S, S)).astype(np.float32)
    for h in range(n_heads):
        for i in range(S):
            if i % 3 == 0 and i > 0:
                j = rng.integers(0, i)
                base[h, i, j] += 8.0
    J = jnp.asarray(base)
    # Use -1e9 instead of finfo.min — still effectively -inf for softmax
    # but avoids float32 underflow complications in KL computation
    causal = jnp.triu(jnp.ones((S, S), dtype=bool), k=1)
    J = jnp.where(causal[None], -1e9, J)
    return J


def kl_mean(p_emp: jnp.ndarray, p_analytic: jnp.ndarray) -> float:
    """
    Mean KL(p_analytic || p_emp) across rows, excluding masked positions.
    Only counts positions where p_analytic > 1e-6 (i.e., unmasked).
    Avoids the eps-inflation bug at large S.
    """
    mask = p_analytic > 1e-6  # (n_heads, S, S) or (S, S)
    log_ratio = jnp.where(mask,
                           jnp.log(p_analytic + 1e-10) - jnp.log(p_emp + 1e-10),
                           0.0)
    kl = jnp.sum(p_analytic * log_ratio, axis=-1)  # per row
    return float(jnp.mean(kl))


# ---------------------------------------------------------------------------
# PATH A: Current sequential loop (baseline)
# ---------------------------------------------------------------------------

def path_a(J_all: jnp.ndarray, K: int, key: jax.Array) -> tuple:
    n_heads, S, _ = J_all.shape
    results = []
    t0_total = time.perf_counter()

    for h in range(n_heads):
        J_bh = J_all[h]
        nodes = [CategoricalNode() for _ in range(S)]
        block = Block(nodes)
        factor = CategoricalEBMFactor([block], J_bh)
        cond = CategoricalGibbsConditional(S)
        spec = BlockGibbsSpec(free_super_blocks=[block], clamped_blocks=[])
        prog = FactorSamplingProgram(
            gibbs_spec=spec, samplers=[cond],
            factors=[factor], other_interaction_groups=[],
        )
        sched = SamplingSchedule(n_warmup=0, n_samples=K, steps_per_sample=1)

        key, sk1, sk2 = jax.random.split(key, 3)
        init = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]
        samples = sample_states(sk2, prog, sched, init, [], [block])
        one_hot = jax.nn.one_hot(samples[0].astype(jnp.int32), S, dtype=jnp.float32)
        results.append(one_hot.mean(axis=0))

    # force all results to complete before stopping clock
    _ = [r.block_until_ready() for r in results]
    elapsed = time.perf_counter() - t0_total

    p_emp = jnp.stack(results, axis=0)  # (n_heads, S, S)
    return p_emp, elapsed


# ---------------------------------------------------------------------------
# PATH B: vmap over heads on pre-built program
# ---------------------------------------------------------------------------

def path_b(J_all: jnp.ndarray, K: int, key: jax.Array) -> tuple:
    if not EQX_AVAILABLE:
        return None, None

    n_heads, S, _ = J_all.shape

    # Build program once for shape S
    nodes = [CategoricalNode() for _ in range(S)]
    block = Block(nodes)
    J_dummy = jnp.zeros((S, S))
    factor = CategoricalEBMFactor([block], J_dummy)
    cond = CategoricalGibbsConditional(S)
    spec = BlockGibbsSpec(free_super_blocks=[block], clamped_blocks=[])
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[cond],
        factors=[factor], other_interaction_groups=[],
    )
    sched = SamplingSchedule(n_warmup=0, n_samples=K, steps_per_sample=1)

    def sample_one_head(J_h, key):
        """Sample one head using pre-built program with swapped weights."""
        W_h = J_h[:, None, :]  # (S, 1, S) — internal shape
        updated_prog = eqx.tree_at(
            lambda p: p.per_block_interactions[0][0].weights,
            prog, W_h
        )
        key, sk1, sk2 = jax.random.split(key, 3)
        init = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]
        samples = sample_states(sk2, updated_prog, sched, init, [], [block])
        one_hot = jax.nn.one_hot(
            samples[0].astype(jnp.int32), S, dtype=jnp.float32)
        return one_hot.mean(axis=0)  # (S, S)

    # Warmup: one call to compile
    key, sk = jax.random.split(key)
    _ = sample_one_head(J_all[0], sk)

    try:
        # Try vmapping over heads
        sample_all_heads = jax.vmap(sample_one_head, in_axes=(0, 0))
        keys = jax.random.split(key, n_heads)

        # Warmup the vmapped version
        _ = sample_all_heads(J_all, keys)

        t0 = time.perf_counter()
        p_emp = sample_all_heads(J_all, keys)
        p_emp.block_until_ready()
        elapsed = time.perf_counter() - t0
        return p_emp, elapsed

    except Exception as e:
        print(f"  [vmap failed: {e}]")
        # Fallback: sequential with pre-built program (no vmap)
        print("  Falling back to sequential pre-built program (no vmap)...")
        results = []
        keys_seq = jax.random.split(key, n_heads)
        t0 = time.perf_counter()
        for h in range(n_heads):
            results.append(sample_one_head(J_all[h], keys_seq[h]))
        _ = [r.block_until_ready() for r in results]
        elapsed = time.perf_counter() - t0
        return jnp.stack(results, axis=0), elapsed


# ---------------------------------------------------------------------------
# PATH C: Direct jax.random.categorical (bypass sample_states entirely)
# ---------------------------------------------------------------------------

def path_c(J_all: jnp.ndarray, K: int, key: jax.Array) -> tuple:
    """
    Bypass sample_states. Since theta[i,:] = J[i,:] with no state dependence,
    each Gibbs step is i.i.d. from softmax(J[h,i,:]). Draw K samples directly.

    J_all: (n_heads, S, S)
    Returns: p_emp (n_heads, S, S), elapsed
    """
    n_heads, S, _ = J_all.shape

    # Warmup
    key, sk = jax.random.split(key)
    _keys = jax.random.split(sk, K)
    # Draw K samples for all heads and all query positions at once
    # J_all: (n_heads, S, S) — broadcast over K
    _samples = jax.vmap(
        lambda k: jax.random.categorical(k, J_all, axis=-1)
    )(_keys)  # (K, n_heads, S)
    _samples.block_until_ready()

    # Timed run
    key, sk = jax.random.split(key)
    keys_k = jax.random.split(sk, K)

    t0 = time.perf_counter()
    samples = jax.vmap(
        lambda k: jax.random.categorical(k, J_all, axis=-1)
    )(keys_k)  # (K, n_heads, S)
    samples.block_until_ready()
    elapsed = time.perf_counter() - t0

    # Convert to (n_heads, S, S) empirical distribution
    one_hot = jax.nn.one_hot(samples.astype(jnp.int32), S, dtype=jnp.float32)
    # one_hot: (K, n_heads, S, S)
    p_emp = one_hot.mean(axis=0)  # (n_heads, S, S)
    return p_emp, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("TASB THRML batched sampling diagnostic")
    print("=" * 78)
    print()
    print(f"  JAX devices: {jax.devices()}")
    print(f"  equinox available: {EQX_AVAILABLE}")
    print()

    S       = 60
    N_HEADS = 24
    K       = 50
    SEED    = 42

    J_all      = build_test_J(S=S, n_heads=N_HEADS, seed=SEED)
    p_analytic = jax.nn.softmax(J_all, axis=-1)  # (24, S, S)
    key        = jax.random.key(SEED)

    results = {}

    # ── PATH A ───────────────────────────────────────────────────────────────
    print(f"PATH A: Sequential fresh-construction (baseline)")
    print(f"        S={S}, N_HEADS={N_HEADS}, K={K}")
    # Warmup
    key, sk = jax.random.split(key)
    _, _ = path_a(J_all[:1], K, sk)
    # Timed run
    key, sk = jax.random.split(key)
    p_a, t_a = path_a(J_all, K, sk)
    kl_a = kl_mean(p_a, p_analytic)
    results['A'] = (p_a, t_a, kl_a)
    print(f"  Total: {t_a:.3f}s  |  {t_a/N_HEADS:.3f}s/head  |  "
          f"mean KL-to-analytic: {kl_a:.5f}")
    print()

    # ── PATH B ───────────────────────────────────────────────────────────────
    print(f"PATH B: vmap over heads (pre-built program + eqx.tree_at)")
    if not EQX_AVAILABLE:
        print("  SKIPPED — equinox not available")
    else:
        key, sk = jax.random.split(key)
        p_b, t_b = path_b(J_all, K, sk)
        if p_b is not None:
            kl_b = kl_mean(p_b, p_analytic)
            results['B'] = (p_b, t_b, kl_b)
            print(f"  Total: {t_b:.3f}s  |  {t_b/N_HEADS:.3f}s/head  |  "
                  f"mean KL-to-analytic: {kl_b:.5f}")
        else:
            print("  FAILED")
    print()

    # ── PATH C ───────────────────────────────────────────────────────────────
    print(f"PATH C: Direct jax.random.categorical (bypasses sample_states)")
    print(f"        [diagnostic only — mathematically equivalent, not THRML]")
    key, sk = jax.random.split(key)
    p_c, t_c = path_c(J_all, K, sk)
    kl_c = kl_mean(p_c, p_analytic)
    results['C'] = (p_c, t_c, kl_c)
    print(f"  Total: {t_c:.3f}s  |  {t_c/N_HEADS:.3f}s/head  |  "
          f"mean KL-to-analytic: {kl_c:.5f}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    t_a = results['A'][1]
    print(f"  PATH A (baseline):     {t_a:.3f}s  ({t_a/N_HEADS:.3f}s/head)")
    if 'B' in results:
        t_b = results['B'][1]
        print(f"  PATH B (vmap):         {t_b:.3f}s  ({t_b/N_HEADS:.3f}s/head)  "
              f"speedup vs A: {t_a/t_b:.2f}x")
    t_c = results['C'][1]
    print(f"  PATH C (direct jax):   {t_c:.3f}s  ({t_c/N_HEADS*1000:.2f}ms/head)  "
          f"speedup vs A: {t_a/t_c:.2f}x  [ceiling]")
    print()
    print("  KL-to-analytic (lower = closer to correct softmax(J)):")
    print(f"    A: {results['A'][2]:.5f}")
    if 'B' in results:
        print(f"    B: {results['B'][2]:.5f}")
    print(f"    C: {results['C'][2]:.5f}")
    print()

    if 'B' in results:
        t_b = results['B'][1]
        speedup_b = t_a / t_b
        if speedup_b > 5:
            print("  [PATH B] vmap is dramatically faster — "
                  "use this as the production path.")
        elif speedup_b > 1.5:
            print("  [PATH B] vmap is moderately faster — "
                  "worth pursuing but check vmap compatibility carefully.")
        else:
            print("  [PATH B] vmap is not meaningfully faster — "
                  "sample_states may not be vmap-compatible or the bottleneck "
                  "is inside sample_states itself rather than dispatch overhead.")

    speedup_c = t_a / t_c
    print(f"  [PATH C] Direct JAX is {speedup_c:.0f}x faster than baseline.")
    if speedup_c > 100:
        print("           This is the theoretical ceiling. The gap between C "
              "and B/A")
        print("           is entirely the cost of the THRML sampling infrastructure.")
        print("           If THRML must be used for hardware-handoff validity,")
        print("           investigate vmap or batched factor construction.")
        print("           If C's speed is acceptable for software validation,")
        print("           consider switching to direct JAX sampling for the")
        print("           demo wrapper and reserving THRML for the hardware path.")
    print("=" * 78)


if __name__ == "__main__":
    main()
