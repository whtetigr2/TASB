#!/usr/bin/env python3
"""
tasb_thrml_vmap_patch_v1.py
==============================================================================
Applies the vmap-over-heads patch to tasb_sampler_thrml.py in-place.

WHAT THIS DOES:
    Replaces the per-head Python loop in thrml_sample() with a vmapped
    implementation that dispatches all heads in one GPU call.

    BEFORE: for b in range(B): for h in heads: [build objects] [sample_states]
            → 24 sequential GPU dispatches, ~0.85s each = ~20s/token

    AFTER:  build program once, vmap sample_one_head over head axis
            → 1 fused GPU dispatch across all heads = ~1s/token (22x speedup)

    Also adds equinox import (required for eqx.tree_at weight swap).

USAGE:
    cp tasb_sampler_thrml.py tasb_sampler_thrml.py.bak_prevmap
    python tasb_thrml_vmap_patch_v1.py tasb_sampler_thrml.py
    python tasb_sampler_thrml.py   # run self-test to verify

VERIFIED:
    Path B in tasb_thrml_batch_diag_v1.py: 0.939s / 24 heads on T4
    vs Path A baseline: 20.582s / 24 heads → 21.9x speedup
    KL faithfulness: unchanged (same softmax(J) distribution)
==============================================================================
"""

import sys
import re

TARGET = sys.argv[1] if len(sys.argv) > 1 else "tasb_sampler_thrml.py"

with open(TARGET, "r") as f:
    src = f.read()

# ── 1. Add equinox import after the existing try/except import blocks ─────────
EQX_IMPORT = '''
try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False
    warnings.warn(
        "equinox not installed — vmap batching unavailable. "
        "Run: pip install equinox", stacklevel=2)

'''

# Insert after the DLPACK try/except block (ends with "DLPACK_AVAILABLE = False")
DLPACK_SENTINEL = "    DLPACK_AVAILABLE = False\n"
if EQX_IMPORT not in src:
    src = src.replace(
        DLPACK_SENTINEL,
        DLPACK_SENTINEL + EQX_IMPORT,
        1  # only first occurrence
    )
    print("  [+] Added equinox import block")
else:
    print("  [=] equinox import already present, skipping")

# ── 2. Replace the per-head loop with vmapped version ────────────────────────
OLD_LOOP = '''    J_jax = _torch_to_jax(J)              # (B, n_q, S, S)
    key   = jax.random.key(seed)
    out   = jnp.zeros((B, n_q, S, S), dtype=jnp.float32)
    heads = [head_idx] if head_idx is not None else range(n_q)

    for b in range(B):
        for h in heads:
            J_bh = J_jax[b, h]            # (S, S)

            # S query positions, each choosing among S key positions
            nodes      = [CategoricalNode() for _ in range(S)]
            free_block = Block(nodes)

            # CategoricalEBMFactor(node_groups, weights)
            # node_groups = [Block of S categorical nodes]
            # weights shape (S, S): weights[i,j] = logit for key j at query i
            factor      = CategoricalEBMFactor([free_block], J_bh)
            conditional = CategoricalGibbsConditional(S)

            # FactorSamplingProgram is the cleaner high-level wrapper
            # node_shape_dtypes defaults cover CategoricalNode → uint8
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

            key, sk1, sk2 = jax.random.split(key, 3)
            init_state = [
                jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)
            ]
            schedule = SamplingSchedule(
                n_warmup=n_warmup,
                n_samples=K,
                steps_per_sample=steps_per_sample,
            )

            # ── HARDWARE HANDOFF ──────────────────────────────────────────
            # Today:  JAX GPU simulation via THRML block Gibbs
            # Future: chip.sample(program, schedule)  ← TSU hardware
            samples = sample_states(
                sk2, program, schedule,
                init_state,
                [],           # no clamped blocks
                [free_block], # collect these
            )
            # samples[0]: (K, S) — K draws, samples[0][k,i] = key chosen at query i

            one_hot = jax.nn.one_hot(
                samples[0].astype(jnp.int32), S, dtype=jnp.float32
            )                             # (K, S, S)
            p_bh = one_hot.mean(axis=0)   # (S, S) empirical distribution
            out  = out.at[b, h].set(p_bh)'''

NEW_LOOP = '''    J_jax = _torch_to_jax(J)              # (B, n_q, S, S)
    key   = jax.random.key(seed)

    # ── Build program structure ONCE per (S, K) configuration ────────────
    # All heads share the same graph topology; only J_bh differs per head.
    # We build one program with a dummy J, then swap weights per head via
    # eqx.tree_at (0.2ms overhead, no JAX retrace).
    nodes_proto  = [CategoricalNode() for _ in range(S)]
    free_block   = Block(nodes_proto)
    factor_proto = CategoricalEBMFactor([free_block], jnp.zeros((S, S)))
    conditional  = CategoricalGibbsConditional(S)
    gibbs_spec   = BlockGibbsSpec(
        free_super_blocks=[free_block],
        clamped_blocks=[],
    )
    program_proto = FactorSamplingProgram(
        gibbs_spec=gibbs_spec,
        samplers=[conditional],
        factors=[factor_proto],
        other_interaction_groups=[],
    )
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=K,
        steps_per_sample=steps_per_sample,
    )

    def _sample_one_head(J_h: jnp.ndarray, head_key: jax.Array) -> jnp.ndarray:
        """
        Sample p_thermo for a single head.
        J_h: (S, S) logit matrix for this head.
        Returns: (S, S) empirical distribution.

        ── HARDWARE HANDOFF ──────────────────────────────────────────────
        Today:  sample_states() — JAX GPU simulation via THRML block Gibbs
        Future: chip.sample(program, schedule) ← TSU hardware
        The only change for hardware: replace sample_states() call below.
        ──────────────────────────────────────────────────────────────────
        """
        # Swap weights into the pre-built program (no retrace)
        W_h = J_h[:, None, :]             # (S, 1, S) — internal THRML shape
        program = eqx.tree_at(
            lambda p: p.per_block_interactions[0][0].weights,
            program_proto, W_h,
        )
        head_key, sk1, sk2 = jax.random.split(head_key, 3)
        init_state = [jax.random.randint(sk1, (S,), 0, S, dtype=jnp.uint8)]
        samples = sample_states(
            sk2, program, schedule,
            init_state,
            [],           # no clamped blocks
            [free_block], # collect these
        )
        # samples[0]: (K, S) — K draws, samples[0][k,i] = key at query i
        one_hot = jax.nn.one_hot(
            samples[0].astype(jnp.int32), S, dtype=jnp.float32
        )                                  # (K, S, S)
        return one_hot.mean(axis=0)        # (S, S) empirical distribution

    if head_idx is not None:
        # Single-head override path — used for diagnostics
        heads = [head_idx]
        out = jnp.zeros((B, n_q, S, S), dtype=jnp.float32)
        for b in range(B):
            key, sk = jax.random.split(key)
            p_bh = _sample_one_head(J_jax[b, head_idx], sk)
            out  = out.at[b, head_idx].set(p_bh)
    else:
        # ── Production path: vmap over all heads in one GPU dispatch ─────
        # jax.vmap fuses the 24 per-head sample_states calls into one
        # vectorized kernel, eliminating sequential dispatch overhead.
        _sample_all_heads = jax.vmap(_sample_one_head, in_axes=(0, 0))
        out_list = []
        for b in range(B):
            head_keys = jax.random.split(key, n_q + 1)
            key       = head_keys[0]
            head_keys = head_keys[1:]      # (n_q, 2) key array
            p_b = _sample_all_heads(J_jax[b], head_keys)  # (n_q, S, S)
            out_list.append(p_b)
        out = jnp.stack(out_list, axis=0)  # (B, n_q, S, S)'''

if OLD_LOOP in src:
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    print("  [+] Replaced per-head loop with vmapped version")
else:
    print("  [!] ERROR: Could not find the loop to replace.")
    print("      The source may have been modified already.")
    print("      Check the file manually.")
    sys.exit(1)

with open(TARGET, "w") as f:
    f.write(src)

print(f"  [✓] Patch written to {TARGET}")
print()
print("  Next steps:")
print("    1. Run self-test:  python tasb_sampler_thrml.py")
print("    2. Run live test:  XLA_PYTHON_CLIENT_PREALLOCATE=false \\")
print("                       python tasb_llama32_chat_runtime.py")
print("       /backend thrml, then 'hi!' — compare Elapsed vs ~22s/tok baseline")
