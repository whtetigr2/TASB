"""
tasb_sampler_qubo.py — QUBO Backend for TASB
==============================================================================
© 2026 Paul W. Shaver. All rights reserved.

Boltzmann sampling via QUBO encoding on an all-to-all p-bit substrate.
Uses a CPU reference simulator standing in for hardware (Purdue 400 p-bit
FPGA, Extropic TSU all-to-all blocks). Drop-in replacement at the hardware
boundary — swap cpu_gibbs_sampler() for chip.sample() when access is available.

ENCODING:
    For each (batch b, head h, query row i):
        Spins x[i,j] ∈ {0,1} for j ∈ [0, S)
        Linear bias:    h[i,j]   = -J[b,h,i,j] - λ
        Pairwise (j<k): J_pair   = 2λ for spins within same row
        Cross-row coupling: 0 (rows are independent)

    Empirically validated operating point: λ_mult = 1.0
    Operating window: λ_mult ∈ [0.8, 1.5]
    At λ_mult=1.0, S=4: KL ≈ 0.04, one-hot rate ≈ 0.88
    At λ_mult=1.0, S=8: KL ≈ 0.07, one-hot rate ≈ 0.77

HARDWARE HANDOFF:
    cpu_gibbs_sampler() is the substrate stand-in.
    Replace with chip.sample(biases, couplings, n_samples) for FPGA hardware.
==============================================================================
"""

import warnings
from typing import Optional

import torch
import numpy as np

try:
    from tasb_qubo_encoder import (
        encode_attention_qubo,
        cpu_gibbs_sampler,
        decode_samples_to_attention,
    )
    QUBO_AVAILABLE = True
except ImportError:
    QUBO_AVAILABLE = False
    warnings.warn(
        "tasb_qubo_encoder not found. Place tasb_qubo_encoder.py in the same dir.",
        stacklevel=2,
    )


def qubo_sample(
    capture,
    K: int = 50,
    seed: int = 42,
    lam_multiplier: float = 1.0,
    n_warmup: int = 200,
    steps_per_sample: int = 2,
    head_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample from the attention Boltzmann distribution via QUBO encoding.

    Maps J[b,h,i,j] onto a (S*S)-binary-spin all-to-all QUBO problem and
    samples via CPU Gibbs. Returns row-stochastic p_thermo for the bridge.

    Parameters
    ----------
    capture : LayerCapture
        Contains q_post_rope, k_post_rope, scaling, attention_mask.
    K : int
        Number of substrate samples to average per head.
    seed : int
        RNG seed for reproducibility across the K samples.
    lam_multiplier : float
        One-hot penalty strength. Default 1.0 from empirical validation.
    n_warmup : int
        Gibbs sweeps before sample collection.
    steps_per_sample : int
        Gibbs sweeps between samples.
    head_idx : int, optional
        If set, only sample for this head. Otherwise all heads.

    Returns
    -------
    p_thermo : torch.Tensor, shape (B, n_q, S, S)
        Row-stochastic. Ready for alpha-blend in tasb_injector_v2.py.
    """
    if not QUBO_AVAILABLE:
        raise ImportError("tasb_qubo_encoder missing — ensure it's importable")

    q     = capture.q_post_rope.float()
    k     = capture.k_post_rope.float()
    scale = float(capture.scaling)
    mask  = capture.attention_mask
    dev   = q.device

    B, n_q, S, _ = q.shape
    n_kv         = k.shape[1]
    n_kvg        = n_q // n_kv

    # Build J = QK^T * scale + mask (post-RoPE)
    with torch.no_grad():
        k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k
        J = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        if mask is not None:
            J = J + mask.float()

    # Move to CPU for the substrate simulator
    # (Real FPGA would also receive data from host — same memcpy cost)
    J_np = J.cpu().numpy().astype(np.float32)

    out = np.zeros((B, n_q, S, S), dtype=np.float32)
    heads = [head_idx] if head_idx is not None else range(n_q)
    rng_master = np.random.default_rng(seed)

    for b in range(B):
        for h in heads:
            J_bh = J_np[b, h]                                        # (S, S)

            # Encode to QUBO biases + couplings
            biases, couplings = encode_attention_qubo(
                J_bh, lam_multiplier=lam_multiplier
            )

            # Sample on the substrate (CPU stand-in for FPGA)
            head_seed = int(rng_master.integers(0, 2**31 - 1))
            samples = cpu_gibbs_sampler(
                biases, couplings,
                n_samples=K,
                n_warmup=n_warmup,
                steps_per_sample=steps_per_sample,
                seed=head_seed,
            )                                                        # (K, S*S)

            # Decode to empirical row-stochastic distribution
            p_bh, one_hot_rate = decode_samples_to_attention(samples, S)

            # Sanity warning if constraint violations climb above operating window
            if one_hot_rate < 0.70:
                warnings.warn(
                    f"QUBO backend: low one-hot rate ({one_hot_rate:.3f}) at "
                    f"b={b}, h={h}, S={S}. Consider raising lam_multiplier.",
                    stacklevel=2,
                )

            out[b, h] = p_bh

    # Back to torch on the original device, fp32
    p_thermo = torch.from_numpy(out).to(device=dev, dtype=torch.float32)

    # Row-stochastic sanity
    dev_max = (p_thermo.sum(dim=-1) - 1.0).abs().max().item()
    if dev_max > 1e-3:
        warnings.warn(f"p_thermo row sum deviation {dev_max:.2e}", stacklevel=2)

    return p_thermo


class QUBOBackend:
    """Drop-in backend for bridge_forward(backend='qubo')."""

    def __init__(
        self,
        lam_multiplier: float = 1.0,
        n_warmup: int = 200,
        steps_per_sample: int = 2,
    ):
        if not QUBO_AVAILABLE:
            raise ImportError("tasb_qubo_encoder missing")
        self.lam_multiplier   = lam_multiplier
        self.n_warmup         = n_warmup
        self.steps_per_sample = steps_per_sample

    def sample(
        self,
        capture,
        K: int,
        seed: int,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        return qubo_sample(
            capture, K=K, seed=seed,
            lam_multiplier=self.lam_multiplier,
            n_warmup=self.n_warmup,
            steps_per_sample=self.steps_per_sample,
            head_idx=head_idx,
        )


if __name__ == "__main__":
    print("tasb_sampler_qubo.py self-test")
    print()

    if not QUBO_AVAILABLE:
        print("  tasb_qubo_encoder not importable. Exiting.")
        raise SystemExit(1)

    # Synthetic capture for testing without LLaMA
    class FakeCapture:
        pass

    S = 6
    B = 1
    n_q = 2  # small to keep CPU test fast
    head_dim = 8

    cap = FakeCapture()
    torch.manual_seed(42)
    cap.q_post_rope    = torch.randn(B, n_q, S, head_dim)
    cap.k_post_rope    = torch.randn(B, n_q, S, head_dim)
    cap.scaling        = 1.0 / np.sqrt(head_dim)
    cap.attention_mask = None

    print(f"  Synthetic capture: B={B}, n_q={n_q}, S={S}, head_dim={head_dim}")
    print(f"  Sampling K=20 via QUBO backend (CPU; slow)...")

    import time
    t0 = time.time()
    p = qubo_sample(cap, K=20, seed=42, lam_multiplier=1.0,
                    n_warmup=200, steps_per_sample=2)
    elapsed = time.time() - t0

    # Compute vanilla softmax for comparison
    with torch.no_grad():
        J = torch.matmul(cap.q_post_rope, cap.k_post_rope.transpose(-2, -1)) * cap.scaling
        p_vanilla = torch.softmax(J, dim=-1)

    # KL between vanilla softmax and QUBO empirical
    eps = 1e-8
    kl = (p_vanilla * (torch.log(p_vanilla + eps) - torch.log(p + eps))).sum(-1).mean().item()

    print()
    print(f"  Elapsed:     {elapsed:.1f}s for {B*n_q} heads at S={S}")
    print(f"  Per-head:    {elapsed/(B*n_q):.2f}s")
    print(f"  Output shape: {p.shape}")
    print(f"  Row sums OK:  {(p.sum(dim=-1) - 1.0).abs().max().item():.2e}")
    print(f"  KL(vanilla || qubo): {kl:.4f}")
    print()
    print(f"  Sample row (head 0, query 0):")
    print(f"    vanilla:  {p_vanilla[0, 0, 0].numpy()}")
    print(f"    qubo:     {p[0, 0, 0].numpy()}")
    print()
    print(f"  [{'PASS' if kl < 0.1 else 'CHECK'} — KL < 0.1 expected at S=6]")
