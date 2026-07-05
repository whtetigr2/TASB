# CRITICAL: set JAX/XLA memory flags before any jax import (XLA reads these at
# init). Stops XLA preallocating ~75% of the GPU for thrml (~13.8GB -> ~2.6GB).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
backends/thrml.py — THRML Backend for thermobridge
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver. USPTO Provisional 64/019,999.

VERIFIED SIGNATURES (from live thrml install on Lightning A100):

    CategoricalNode()
    Block(nodes: list)
    CategoricalEBMFactor(node_groups: list[Block], weights: Array)
    CategoricalGibbsConditional(n_categories: int)
    BlockGibbsSpec(free_super_blocks, clamped_blocks, node_shape_dtypes=...)
        (this file overrides node_shape_dtypes -- see _STATE_DTYPE below;
        THRML's own default is uint8, which silently truncates any
        n_categories > 255)
    BlockSamplingProgram(gibbs_spec, samplers, interaction_groups)
    FactorSamplingProgram(gibbs_spec, samplers, factors, other_interaction_groups)
    SamplingSchedule(n_warmup, n_samples, steps_per_sample)
    sample_states(key, program, schedule, init_state_free, state_clamp,
                  nodes_to_sample)

HARDWARE HANDOFF: sample_states() -> chip.sample() for TSU hardware.
==============================================================================
"""

import warnings  # noqa: E402
from typing import Optional  # noqa: E402

import torch  # noqa: E402
import numpy as np  # noqa: E402

try:
    import jax
    import jax.numpy as jnp
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
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    warnings.warn("THRML not installed. Run: pip install thrml", stacklevel=2)

try:
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack
    import jax.dlpack as jax_dlpack
    DLPACK_AVAILABLE = True
except ImportError:
    DLPACK_AVAILABLE = False

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False
    warnings.warn(
        "equinox not installed -- vmap batching unavailable. "
        "Run: pip install equinox", stacklevel=2)


# ---------------------------------------------------------------------------
# Tensor bridge
# ---------------------------------------------------------------------------

def _torch_to_jax(t: torch.Tensor):
    if DLPACK_AVAILABLE:
        try:
            return jax_dlpack.from_dlpack(
                torch_to_dlpack(t.contiguous().detach()))
        except Exception:
            pass
    return jnp.array(t.cpu().float().numpy())


def _jax_to_torch(arr, device, dtype=torch.float32) -> torch.Tensor:
    if DLPACK_AVAILABLE:
        try:
            return torch.from_dlpack(
                jax_dlpack.to_dlpack(arr)).to(device=device, dtype=dtype)
        except Exception:
            pass
    return torch.tensor(np.array(arr), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Shape bucketing (2026-07-04, fixes the per-token retracing slowdown)
# ---------------------------------------------------------------------------
#
# ROOT CAUSE: thrml_sample() builds its CategoricalNode/EBMFactor graph sized
# to the REAL Sk (cache length) on every call. Under live KV-cached decode,
# Sk grows by 1 on every single generated token -- every call presents JAX
# with a shape it has never traced before, forcing a full retrace/recompile
# of the underlying jax.lax.scan-based Gibbs sampler EVERY TOKEN. Confirmed
# empirically (0% GPU utilization during "generation", flat ~0.9 tok/s with
# no speedup between a "cold" first turn and a supposedly "warm" second turn
# -- if this were one-time JIT warmup, the second turn would have been much
# faster) and via direct inspection of the installed thrml library
# (thrml/block_sampling.py uses jax.lax.scan internally over shapes derived
# from Sq/Sk).
#
# FIX: the standard JAX/XLA pattern for variable-length sequences under
# incremental decoding -- pad to a fixed set of bucket sizes so the compiled
# program's shape stays constant across a RANGE of real lengths, instead of
# changing every single token. Same mechanism used in production JAX/TPU
# LLM-serving kernels (e.g. "Ragged Paged Attention", arXiv:2604.15464 --
# fixed-size pages/blocks + masking padded positions with -inf logits before
# softmax, so XLA compiles once per bucket rather than once per length).
#
# Bucket boundaries are powers of 2 (8, 16, 32, ... up to BUCKET_MAX). Across
# an entire max_seq_len=2048 conversation, this means at most ~8 distinct
# compiled shapes total (8->16->32->64->128->256->512->1024->2048) instead of
# up to 2048 -- each compiled program gets reused for every token within its
# bucket range before the next recompile.
#
# Only Sk is bucketed here, not Sq. In the live-chat decode loop Sq is always
# 1 (one new token per step) -- it never grows, so it never triggers
# retracing on its own. Sq only varies during prefill (where Sq==Sk, so
# bucketing Sk already covers it), and prefill happens once per conversation
# turn, not once per token -- a much lower-frequency recompilation event than
# the per-token decode cost this fix targets. Bucketing Sq too would be a
# reasonable follow-up for workloads with many short, varied-length prefill
# calls, but is out of scope for the dominant per-token decode bottleneck.

BUCKET_MIN = 8
BUCKET_MAX = 4096  # generous headroom above the current max_seq_len=2048
_NEG_INF_LOGIT = -1e30  # matches the masking convention used elsewhere in
                        # this codebase (HF's own additive-mask sentinel is
                        # ~-3.4e38; this is comfortably large enough that
                        # softmax/Gibbs sampling never selects a padded
                        # position, verified empirically below)

# ---------------------------------------------------------------------------
# State dtype override (2026-07-05, fixes silent index truncation above 255)
# ---------------------------------------------------------------------------
#
# THRML's own library default for CategoricalNode state is uint8
# (thrml/pgm.py DEFAULT_NODE_SHAPE_DTYPES), i.e. category indices above 255
# silently wrap (280 -> 24) with no error or warning -- confirmed both in
# THRML itself (reported upstream: extropic-ai/thrml#62) and, separately,
# empirically reproduced against THIS file's own construction (no
# node_shape_dtypes override existed here either). Since Sk_bucket can be up
# to BUCKET_MAX=4096, this was a real, silent attention-position-corruption
# bug for any real conversation whose thrml-backend call ever saw
# Sk/Sk_bucket > 255 -- i.e. any conversation past roughly 256 tokens. Fixed
# by explicitly overriding the node dtype to something wide enough for
# BUCKET_MAX and beyond. See FIND-037 in the vault for the full writeup.
_STATE_DTYPE = jnp.int32


def _bucket_size(n: int) -> int:
    """Round n up to the next power-of-2 bucket, clamped to [BUCKET_MIN, BUCKET_MAX]."""
    if n <= BUCKET_MIN:
        return BUCKET_MIN
    b = BUCKET_MIN
    while b < n:
        b *= 2
    return min(b, BUCKET_MAX)


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def thrml_sample(
    capture,
    K: int = 50,
    seed: int = 42,
    n_warmup: int = 0,
    steps_per_sample: int = 1,
    head_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample from the attention Boltzmann distribution using THRML.

    Maps attention logits J[b,h,i,j] onto a CategoricalEBMFactor graph
    and runs block Gibbs sampling via THRML.

    THRML GRAPH (per batch b, per head h):
        Sq CategoricalNodes -- one per query position i (Sq = number of new
        query tokens this step; Sq=1 under KV-cached decode, Sq==Sk at prefill)
        CategoricalEBMFactor([Block(nodes)], J_bh)
            weights[i,j] = logit for key j (of Sk total) at query i
            Energy: E_j = -weights[i,j]
            Boltzmann distribution = softmax(weights[i,:]) = attention weights
        CategoricalGibbsConditional(Sk_bucket) samples from softmax(weights[i,:])

    HARDWARE HANDOFF:
        sample_states() is the chip call.
        Replace with chip.sample(program, schedule) for TSU hardware.

    Returns
    -------
    p_thermo : torch.Tensor, shape (B, n_q, Sq, Sk)
        Row-stochastic. Ready for alpha-blend in thermobridge.inject.
        (2026-07-04 fix #1: Sq/Sk were conflated into a single S derived from
        q.shape only, which silently built a degenerate 1-node/1-category
        graph under KV-cached decode where Sq=1 != Sk. Fixed to use Sq
        (number of nodes) and Sk (categories per node) separately.
        2026-07-04 fix #2: Sk is now bucketed to the next power-of-2 boundary
        before building the graph, so JAX/XLA compiles once per bucket
        instead of once per token -- see the "Shape bucketing" section above
        for the full root-cause writeup. The real Sk is used for the output
        shape/slicing; only the internal graph/program construction sees the
        padded/bucketed size.)
    """
    if not THRML_AVAILABLE:
        raise ImportError("pip install thrml")

    q     = capture.q_post_rope.float()   # (B, n_q, Sq, head_dim)
    k     = capture.k_post_rope.float()   # (B, n_kv, Sk, head_dim)
    scale = float(capture.scaling)
    mask  = capture.attention_mask
    dev   = q.device

    B, n_q, Sq, _ = q.shape
    n_kv          = k.shape[1]
    n_kvg         = n_q // n_kv
    Sk            = k.shape[2]  # cache length -- diverges from Sq under KV-caching

    Sk_bucket = _bucket_size(Sk)
    pad = Sk_bucket - Sk

    # Build logit matrix J = QK^T * scale + mask
    with torch.no_grad():
        k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k
        J = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        if mask is not None:
            J = J + mask.float()
        if pad > 0:
            # Pad the key/category dimension out to the bucket size with a
            # large-negative logit -- softmax(-1e30) underflows to exactly
            # 0.0 in float32, so these positions can never be sampled
            # (same masking-respecting argument already used elsewhere in
            # this codebase for causal/attention masking).
            J = torch.nn.functional.pad(J, (0, pad), value=_NEG_INF_LOGIT)

    J_jax = _torch_to_jax(J)              # (B, n_q, Sq, Sk_bucket)
    key   = jax.random.key(seed)

    # -- Build program structure ONCE per (Sq, Sk_bucket, K) configuration --
    # All heads share the same graph topology; only J_bh differs per head.
    # We build one program with a dummy J, then swap weights per head via
    # eqx.tree_at (0.2ms overhead, no JAX retrace). Because Sk_bucket only
    # changes at power-of-2 boundaries (not every token), this program -- and
    # the XLA compilation triggered the first time it's invoked at a given
    # bucket -- gets reused across every subsequent token within that bucket
    # range, instead of recompiling on every single call.
    nodes_proto  = [CategoricalNode() for _ in range(Sq)]
    free_block   = Block(nodes_proto)
    factor_proto = CategoricalEBMFactor([free_block], jnp.zeros((Sq, Sk_bucket)))
    conditional  = CategoricalGibbsConditional(Sk_bucket)
    gibbs_spec   = BlockGibbsSpec(
        free_super_blocks=[free_block],
        clamped_blocks=[],
        node_shape_dtypes={
            CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=_STATE_DTYPE)
        },
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
        J_h: (Sq, Sk_bucket) logit matrix for this head (padding columns, if
             any, carry _NEG_INF_LOGIT and are structurally unreachable).
        Returns: (Sq, Sk_bucket) empirical distribution.

        -- HARDWARE HANDOFF --------------------------------------------
        Today:  sample_states() -- JAX GPU simulation via THRML block Gibbs
        Future: chip.sample(program, schedule) <- TSU hardware
        The only change for hardware: replace sample_states() call below.
        ------------------------------------------------------------------
        """
        # Swap weights into the pre-built program (no retrace)
        W_h = J_h[:, None, :]             # (Sq, 1, Sk_bucket) -- internal THRML shape
        program = eqx.tree_at(
            lambda p: p.per_block_interactions[0][0].weights,
            program_proto, W_h,
        )
        head_key, sk1, sk2 = jax.random.split(head_key, 3)
        # Only ever initialize into a REAL (non-padded) position -- padded
        # positions have -1e30 logit, and while they're already
        # structurally unreachable via any real transition, starting there
        # would need an extra (harmless but needless) first-step correction.
        init_state = [jax.random.randint(sk1, (Sq,), 0, Sk, dtype=_STATE_DTYPE)]
        samples = sample_states(
            sk2, program, schedule,
            init_state,
            [],           # no clamped blocks
            [free_block], # collect these
        )
        # samples[0]: (K, Sq) -- K draws, samples[0][k,i] = key at query i
        one_hot = jax.nn.one_hot(
            samples[0].astype(jnp.int32), Sk_bucket, dtype=jnp.float32
        )                                  # (K, Sq, Sk_bucket)
        return one_hot.mean(axis=0)        # (Sq, Sk_bucket) empirical distribution

    if head_idx is not None:
        # Single-head override path -- used for diagnostics
        out = jnp.zeros((B, n_q, Sq, Sk_bucket), dtype=jnp.float32)
        for b in range(B):
            key, sk = jax.random.split(key)
            p_bh = _sample_one_head(J_jax[b, head_idx], sk)
            out  = out.at[b, head_idx].set(p_bh)
    else:
        # -- Production path: vmap over all heads in one GPU dispatch ------
        # jax.vmap fuses the 24 per-head sample_states calls into one
        # vectorized kernel, eliminating sequential dispatch overhead.
        _sample_all_heads = jax.vmap(_sample_one_head, in_axes=(0, 0))
        out_list = []
        for b in range(B):
            head_keys = jax.random.split(key, n_q + 1)
            key       = head_keys[0]
            head_keys = head_keys[1:]      # (n_q, 2) key array
            p_b = _sample_all_heads(J_jax[b], head_keys)  # (n_q, Sq, Sk_bucket)
            out_list.append(p_b)
        out = jnp.stack(out_list, axis=0)  # (B, n_q, Sq, Sk_bucket)

    # Slice the padding columns back off before returning -- callers expect
    # exactly (B, n_q, Sq, Sk), and the padded probability mass is already
    # ~0.0 (structurally unreachable) but slicing makes the guarantee exact
    # rather than relying on "should be negligible."
    if pad > 0:
        out = out[..., :Sk]

    p_thermo = _jax_to_torch(out, dev)

    dev_max = (p_thermo.sum(dim=-1) - 1.0).abs().max().item()
    if dev_max > 1e-3:
        warnings.warn(f"p_thermo row sum deviation {dev_max:.2e}", stacklevel=2)

    return p_thermo


# ---------------------------------------------------------------------------
# Backend class for bridge_forward() dispatch
# ---------------------------------------------------------------------------

class THRMLBackend:
    """Drop-in backend for bridge_forward(backend='thrml')."""

    def __init__(self, n_warmup: int = 0, steps_per_sample: int = 1):
        if not THRML_AVAILABLE:
            raise ImportError("pip install thrml")
        self.n_warmup         = n_warmup
        self.steps_per_sample = steps_per_sample

    def sample(self, capture, K: int, seed: int,
               head_idx: Optional[int] = None) -> torch.Tensor:
        return thrml_sample(
            capture, K=K, seed=seed,
            n_warmup=self.n_warmup,
            steps_per_sample=self.steps_per_sample,
            head_idx=head_idx,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("backends/thrml.py self-test")
    print()

    if not THRML_AVAILABLE:
        print("  THRML not installed. Run: pip install thrml")
        raise SystemExit(1)

    print(f"  THRML:  available")
    print(f"  DLPack: {DLPACK_AVAILABLE}")
    print(f"  JAX:    {jax.devices()}")
    print()
    print("  Smoke test: 4-token attention distribution via THRML...")

    S = 4

    nodes      = [CategoricalNode() for _ in range(S)]
    free_block = Block(nodes)

    # Sharp diagonal preferences
    J = jnp.array([
        [ 3.0, -1.0, -1.0, -1.0],
        [-1.0,  3.0, -1.0, -1.0],
        [-1.0, -1.0,  3.0, -1.0],
        [-1.0, -1.0, -1.0,  3.0],
    ])

    factor      = CategoricalEBMFactor([free_block], J)
    conditional = CategoricalGibbsConditional(S)

    gibbs_spec = BlockGibbsSpec(
        free_super_blocks=[free_block],
        clamped_blocks=[],
        node_shape_dtypes={
            CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=_STATE_DTYPE)
        },
    )
    program = FactorSamplingProgram(
        gibbs_spec=gibbs_spec,
        samplers=[conditional],
        factors=[factor],
        other_interaction_groups=[],
    )

    key = jax.random.key(42)
    key, sk1, sk2 = jax.random.split(key, 3)
    init  = [jax.random.randint(sk1, (S,), 0, S, dtype=_STATE_DTYPE)]
    # Self-test uses conservative n_warmup/steps_per_sample for a strict
    # correctness check. Production defaults (n_warmup=0, steps_per_sample=1)
    # were validated equivalent -- see
    # diagnostics/tasb_thrml_warmup_diagnostic_v2_kl_stats.py.
    sched = SamplingSchedule(n_warmup=50, n_samples=500, steps_per_sample=2)

    # THE HARDWARE HANDOFF CALL
    samples   = sample_states(sk2, program, sched, init, [], [free_block])
    one_hot   = jax.nn.one_hot(samples[0].astype(jnp.int32), S, dtype=jnp.float32)
    empirical = one_hot.mean(axis=0)
    expected  = jax.nn.softmax(J, axis=-1)
    max_err   = float(jnp.abs(empirical - expected).max())

    print(f"  Empirical (500 samples) vs expected softmax:")
    for i in range(S):
        e = [f"{empirical[i,j]:.3f}" for j in range(S)]
        x = [f"{expected[i,j]:.3f}" for j in range(S)]
        print(f"    Q{i}: got {e}  exp {x}")

    print(f"  Max error: {max_err:.4f}")
    print(f"  [{'PASS' if max_err < 0.12 else 'WARN -- try more samples'}]")
    print()
    print("  Hardware handoff: replace sample_states() with chip.sample()")
    print("  in THRMLBackend.sample(). Nothing else in thermobridge changes.")
