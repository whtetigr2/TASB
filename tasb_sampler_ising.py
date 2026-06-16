"""
tasb_sampler_ising.py
==============================================================================
TASB Ising Backend — Hardware-faithful spin-domain Boltzmann sampler.

Maps post-RoPE attention logits to a QUBO-encoded Ising model and samples
spin configurations via THRML's IsingEBM / IsingSamplingProgram. The output
is converted back to the standard p_thermo (B, n_q, S, S) row-stochastic
attention distribution for compatibility with the existing injection layer.

Why this backend exists:
    Real probabilistic hardware (Camsari p-bits, Purdue FPGA, Extropic TSU)
    physically samples Ising spin configurations from
        E(s) = -beta ( sum_i b_i s_i + sum_{(i,j)} J_ij s_i s_j )
    The categorical THRML backend (tasb_sampler_thrml.py) validates the
    bridge in software, but does not exercise the spin domain that physical
    hardware operates in. This backend closes that gap.

Design:
    QUBO encoding with one spin node per (query_position, key_position) cell.
    Per query row i, the energy contribution is
        E_i(s_{i,:}) = - sum_j J[i,j] * (s_{i,j} + 1) / 2          (logit reward)
                       + lambda * (sum_j (s_{i,j}+1)/2 - 1)^2        (one-hot penalty)
    Expanding the penalty into bias + edge terms gives the Ising form:
        b_i = J[i,i_self] / 2 + lambda * (1/2 - n_cells/2)
        J_ising[(i,j),(i,k)] = lambda / 2  for j != k (intra-row edges)
        J_ising[(i,j),(k,l)] = 0           for i != k (no cross-query coupling)
    Output: aggregate K samples, count fraction of times each (i,j) spin was
    True. Renormalize per query row. Handle invalid configurations (zero or
    multi-on rows) via uniform fallback for safety.

References:
    - Kim, "Thermodynamic Isomorphism of Transformers" (arXiv:2602.08216):
      softmax IS the Boltzmann distribution at T = sqrt(d_k).
    - "Thermodynamic significance of QUBO encoding on quantum annealers"
      (arXiv:2601.04402): grounds the QUBO penalty framing.
    - THRML models/ising.py: the IsingEBM / IsingSamplingProgram pattern we
      build on top of.

API:
    sample(capture, K, seed, ...)         -> torch.Tensor (B, n_q, S, S)
    IsingBackend()                         class wrapper for chat runtime

(c) 2026 Paul W. Shaver. USPTO Provisional 64/019,999.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# JAX / THRML imports — required
try:
    import jax
    import jax.numpy as jnp
    from jax import dlpack as jax_dlpack
    JAX_OK = True
except Exception as _e:
    JAX_OK = False
    JAX_IMPORT_ERROR = _e

try:
    from thrml.block_sampling import (
        Block, BlockGibbsSpec, SamplingSchedule, SuperBlock,
        sample_states,
    )
    from thrml.block_management import make_empty_block_state
    from thrml.models.ising import IsingEBM, IsingSamplingProgram
    from thrml.pgm import SpinNode
    THRML_OK = True
except Exception as _e:
    THRML_OK = False
    THRML_IMPORT_ERROR = _e


# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class IsingSamplerConfig:
    """Configuration mirroring SamplerConfig from tasb_sampler_v2 plus
    Ising-specific knobs."""
    K: int                = 50           # samples per (B, head, query) row
    seed: int             = 42           # JAX PRNG seed
    n_warmup: int         = 100          # Gibbs warmup sweeps
    steps_per_sample: int = 5            # inter-sample Gibbs sweeps
    penalty_lambda: Optional[float] = None  # None = auto-compute
    fallback: str         = "uniform"    # "uniform" | "argmax" | "reject"

    def __post_init__(self):
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.fallback not in ("uniform", "argmax", "reject"):
            raise ValueError(f"fallback must be uniform/argmax/reject, "
                             f"got {self.fallback!r}")


# ── Capture -> J matrix (shared with THRML backend) ─────────────────────────
def _compute_J_from_capture(capture) -> torch.Tensor:
    """
    Build the raw J = Q K^T / sqrt(d_k) + mask logits matrix from a
    LayerCapture. Shape: (B, n_q, S, S). Float32 on capture's device.
    """
    q = capture.q_post_rope.float()           # (B, n_q,  S, head_dim)
    k = capture.k_post_rope.float()           # (B, n_kv, S, head_dim)
    sc = float(capture.scaling)

    n_q  = q.shape[1]
    n_kv = k.shape[1]
    if n_q != n_kv:
        n_kvg = n_q // n_kv
        k_exp = k.repeat_interleave(n_kvg, dim=1)
    else:
        k_exp = k

    J = torch.matmul(q, k_exp.transpose(-2, -1)) * sc   # (B, n_q, S, S)
    if capture.attention_mask is not None:
        J = J + capture.attention_mask.float()
    return J


# ── PyTorch -> JAX conversion ───────────────────────────────────────────────
def _torch_to_jax(t: torch.Tensor):
    """Convert a contiguous torch float32 tensor to a jax array via DLPack
    when possible, with a numpy fallback. Bug registry #16."""
    t = t.contiguous().detach()
    try:
        return jax_dlpack.from_dlpack(t)
    except Exception:
        return jnp.array(t.cpu().numpy())


def _jax_to_torch(arr, device) -> torch.Tensor:
    """Convert a jax array back to torch on the requested device."""
    return torch.from_numpy(np.asarray(arr)).to(device)


# ── Build the Ising graph for one (B, head, query) slice ────────────────────
def _build_ising_for_query_row(
    J_row: torch.Tensor,            # (S,) — J[b, h, i, :] for this query
    S: int,
    penalty_lambda: float,
) -> tuple:
    """
    Build the Ising EBM nodes/edges/biases/weights for a single query row.

    QUBO penalty for "exactly one of S binaries is on":
        P(x_1,...,x_S) = lambda * ( (sum_j x_j) - 1 )^2
                       = lambda * ( sum_j x_j^2 + 2 sum_{j<k} x_j x_k
                                    - 2 sum_j x_j + 1 )
        Since x_j is binary, x_j^2 = x_j, so
                       = lambda * ( -sum_j x_j + 2 sum_{j<k} x_j x_k + 1 )
        Constant term drops; linear and quadratic remain.

    In spin {0,1} encoding (THRML internal), with logit reward
        - sum_j J[i,j] * x_j
    the per-row energy becomes
        E_i = sum_j ( -J[i,j] - lambda ) * x_j
              + sum_{j<k} 2*lambda * x_j * x_k

    Returns:
        nodes_row : list[SpinNode] of length S
        edges_row : list[(node_a, node_b)] — all S*(S-1)/2 within-row pairs
        biases_row: jnp.array of shape (S,)
        weights_row: jnp.array of shape (n_edges,)
    """
    J_clipped = J_row.clamp(min=-1e6, max=1e6)
    J_np = J_clipped.cpu().numpy().astype(np.float32)
    nodes_row = [SpinNode() for _ in range(S)]

    # THRML's BernoulliConditional samples P(s) propto exp(gamma * s) with s in {-1,+1}.
    # IsingEBM bias and edge weights add directly to gamma.
    # We want P(spin j = +1) propto exp(J[i,j]) -> bias contributes J[i,j]/2 to gamma_j.
    #
    # QUBO penalty "exactly one +1" in spin coords:
    #   penalty = lam * ((sum (s_j+1)/2) - 1)^2 = (lam/4)*(sum s_j + S - 2)^2
    #   linear contribution to gamma_bias_j: -(S-2)*lam/2
    #   quadratic contribution to gamma_edge(j,k): -lam/2
    lam = penalty_lambda
    bias_per_spin = J_np / 2.0 - (S - 2) * lam / 2.0
    biases_row = jnp.asarray(bias_per_spin, dtype=jnp.float32)

    edges_row = []
    weights = []
    edge_weight = -lam / 2.0
    for a in range(S):
        for b in range(a + 1, S):
            edges_row.append((nodes_row[a], nodes_row[b]))
            weights.append(edge_weight)
    weights_row = jnp.asarray(weights, dtype=jnp.float32)

    return nodes_row, edges_row, biases_row, weights_row


# ── Run Ising sampler for one row ───────────────────────────────────────────
def _sample_one_row(
    J_row: torch.Tensor,
    S: int,
    cfg: IsingSamplerConfig,
    beta: float,
    key,
) -> np.ndarray:
    """
    Sample K spin configurations for one query row.
    Returns: np.ndarray of shape (K, S), boolean (True = spin on / attended).
    """
    nodes_row, edges_row, biases_row, weights_row = _build_ising_for_query_row(
        J_row, S, cfg.penalty_lambda
    )

    ebm = IsingEBM(
        nodes=nodes_row,
        edges=edges_row,
        biases=biases_row,
        weights=weights_row,
        beta=jnp.asarray(beta, dtype=jnp.float32),
    )

    # All S spins are free (no clamping); they form a single SuperBlock so
    # they update together each Gibbs sweep.
    free_block = Block(nodes_row)
    program      = IsingSamplingProgram(
        ebm=ebm,
        free_blocks=[free_block],
        clamped_blocks=[],
    )

    sched = SamplingSchedule(
        n_warmup=cfg.n_warmup,
        n_samples=cfg.K,
        steps_per_sample=cfg.steps_per_sample,
    )

    # init: all spins False (off) — Gibbs will mix from there
    # Use THRML's helper to build the correct pytree structure
    init_state = make_empty_block_state(
        [free_block], ebm.node_shape_dtypes
    )

    samples_state = sample_states(
        key, program, sched, init_state, [], [free_block]
    )
    # samples_state is a list (one entry per free block).
    # Our single free block contains S SpinNodes, so we take entry [0].
    # Shape comes back as (K, S) bool.
    block_state = samples_state[0]
    samples_bool = np.asarray(block_state)
    if samples_bool.ndim == 3 and samples_bool.shape[1] == 1:
        samples_bool = samples_bool[:, 0, :]
    return samples_bool   # (K, S) bool


# ── Convert spin samples to attention probabilities ─────────────────────────
def _samples_to_probs(
    samples: np.ndarray,            # (K, S) bool
    fallback: str,
) -> np.ndarray:
    """
    Aggregate K spin samples into a single attention probability vector
    over S keys. Handles invalid configurations per `fallback`.
    """
    K, S = samples.shape
    counts = np.zeros(S, dtype=np.float32)
    n_valid = 0
    n_zero = 0
    n_multi = 0

    for k in range(K):
        on_positions = np.where(samples[k])[0]
        n_on = len(on_positions)
        if n_on == 1:
            counts[on_positions[0]] += 1.0
            n_valid += 1
        elif n_on == 0:
            if fallback == "uniform":
                counts += 1.0 / S
            elif fallback == "argmax":
                # No spins on -> uniform is the only sensible default
                counts += 1.0 / S
            # "reject" -> just don't count this sample
            n_zero += 1
        else:
            if fallback == "uniform":
                counts[on_positions] += 1.0 / n_on
            elif fallback == "argmax":
                # No logit info here; pick uniformly among on-positions
                counts[on_positions] += 1.0 / n_on
            # "reject" -> drop
            n_multi += 1

    total = counts.sum()
    if total <= 0.0:
        # All samples were rejected. Uniform fallback.
        probs = np.full(S, 1.0 / S, dtype=np.float32)
    else:
        probs = counts / total

    # diag stats — caller can log if desired
    return probs, (n_zero, n_multi, n_valid)


# ── Main entry point ────────────────────────────────────────────────────────
def sample(capture, config: IsingSamplerConfig) -> torch.Tensor:
    """
    Sample p_thermo from a LayerCapture using Ising QUBO encoding.

    Returns: torch.Tensor of shape (B, n_q, S, S), row-stochastic.
    """
    if not JAX_OK:
        raise RuntimeError(
            f"JAX required for Ising backend: {JAX_IMPORT_ERROR!r}")
    if not THRML_OK:
        raise RuntimeError(
            f"THRML required for Ising backend: {THRML_IMPORT_ERROR!r}")

    J = _compute_J_from_capture(capture)           # (B, n_q, S, S)
    B, n_q, S, _ = J.shape
    device = J.device

    # Clamp J to a reasonable energy scale. The attention mask uses
    # very large negative numbers (~-3.4e38 from HF eager attention)
    # which would otherwise dominate the energy landscape.
    # T_struct = sqrt(head_dim) is the natural attention temperature,
    # so logits outside ±50*T_struct are causal-mask artifacts.
    T_struct = float(capture.head_dim ** 0.5) if hasattr(capture, "head_dim") else 11.3
    J_clamp_range = 50.0 * T_struct
    J = J.clamp(min=-J_clamp_range, max=J_clamp_range)

    # Auto-compute penalty if not set
    lam = config.penalty_lambda
    if lam is None:
        lam = 5.0 * float(J.abs().max().item())
        config.penalty_lambda = lam

    # Temperature: T_struct = sqrt(head_dim); already baked into capture.scaling
    # which divides Q.K by sqrt(head_dim). So beta = 1.0 here — we already
    # have logits at the right Boltzmann scale.
    beta = 1.0

    rng = jax.random.PRNGKey(config.seed)

    out = np.zeros((B, n_q, S, S), dtype=np.float32)
    total_zero = 0
    total_multi = 0
    total_valid = 0

    for b in range(B):
        for h in range(n_q):
            for i in range(S):
                rng, sub = jax.random.split(rng)
                row_samples = _sample_one_row(
                    J[b, h, i, :], S, config, beta, sub
                )                                          # (K, S) bool
                probs, (z, m, v) = _samples_to_probs(row_samples, config.fallback)
                out[b, h, i, :] = probs
                total_zero += z
                total_multi += m
                total_valid += v

    total_samples = B * n_q * S * config.K
    if total_samples > 0:
        viol_rate = (total_zero + total_multi) / total_samples
        if viol_rate > 0.05:
            import warnings
            warnings.warn(
                f"Ising sampler constraint violation rate: {viol_rate:.3f} "
                f"(zero={total_zero}, multi={total_multi} of {total_samples}). "
                f"Consider increasing penalty_lambda (currently {lam:.3f})."
            )

    return torch.from_numpy(out).to(device)


# ── Class wrapper for chat runtime ──────────────────────────────────────────
class IsingBackend:
    """Drop-in mirror of THRMLBackend pattern."""

    def __init__(
        self,
        n_warmup: int = 100,
        steps_per_sample: int = 5,
        penalty_lambda: Optional[float] = None,
        fallback: str = "uniform",
    ):
        self.n_warmup = n_warmup
        self.steps_per_sample = steps_per_sample
        self.penalty_lambda = penalty_lambda
        self.fallback = fallback

    def sample(self, capture, K: int = 50, seed: int = 42,
               head_idx: Optional[int] = None) -> torch.Tensor:
        cfg = IsingSamplerConfig(
            K=K,
            seed=seed,
            n_warmup=self.n_warmup,
            steps_per_sample=self.steps_per_sample,
            penalty_lambda=self.penalty_lambda,
            fallback=self.fallback,
        )
        return sample(capture, cfg)
