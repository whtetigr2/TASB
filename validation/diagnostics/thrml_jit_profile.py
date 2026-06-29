# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_thrml_jit_profile.py — JAX JIT Compilation Overhead Tracking
==============================================================================
TASB Validation Suite — Tier 1.A
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
JAX compiles (traces) computation graphs on first call per input shape. If
sequence length S varies every call (which it does during autoregressive
generation with use_cache=False), each new S may trigger XLA recompilation
adding 10-30s per shape. This test characterizes that cost.

For each S in {10, 20, 40, 60, 80, 100}:
  - Call 1: compilation + execution (slow)
  - Call 2: same S, cached execution only (should be fast)
  - Ratio = call2_time / call1_time  (target: < 0.2)
  - After all shapes seen: call S=10 again (cache hit test)

Does NOT require the full LLaMA model. Uses synthetic Q/K tensors that
match LLaMA 3.2-3B dimensions (n_q=24, n_kv=8, head_dim=128, GQA ratio=3).

PASS CONDITION
--------------
  - Any second-call / first-call ratio < 0.2 confirms caching is working.
  - Recompile spike: if call2_time > call1_time * 0.5, that S is not caching.
  - Cache persistence: S=10 after full sequence must be < 0.2x its first-call time.

REPRODUCE
---------
  XLA_PYTHON_CLIENT_PREALLOCATE=false python diagnostics/tasb_thrml_jit_profile.py
==============================================================================
"""

import sys
import time
import csv
import datetime
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


try:
    import jax
    print(f"  JAX:    {jax.devices()}")
except ImportError:
    print("  JAX not installed. Run: pip install jax jaxlib")
    sys.exit(1)

try:
    from thermobridge.backends.thrml import thrml_sample
    print(f"  THRML:  imported")
except ImportError:
    print("  thermobridge.backends.thrml not available. Install: pip install thermobridge thrml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LLaMA 3.2-3B dimensions (matches production config)
# ---------------------------------------------------------------------------
N_Q      = 24      # query heads
N_KV     = 8       # key-value heads (GQA: n_kvg = 24/8 = 3)
HEAD_DIM = 128
SCALE    = 1.0 / (HEAD_DIM ** 0.5)
K_SAMPLES = 10     # small K for speed — testing JIT overhead, not fidelity
SEED     = 42

# Sequence lengths to test
S_VALUES = [10, 20, 40, 60, 80, 100]


@dataclass
class MockCapture:
    """Synthetic attention capture matching LLaMA 3.2-3B GQA dimensions."""
    q_post_rope:    torch.Tensor
    k_post_rope:    torch.Tensor
    scaling:        float
    attention_mask: Optional[torch.Tensor]


def make_capture(S: int, device: str = "cuda") -> MockCapture:
    """Create a random attention capture at sequence length S."""
    torch.manual_seed(42)
    q = torch.randn(1, N_Q, S, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(1, N_KV, S, HEAD_DIM, device=device, dtype=torch.float32)
    # Causal mask: upper triangle = -1e9
    mask = torch.full((1, 1, S, S), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return MockCapture(
        q_post_rope=q,
        k_post_rope=k,
        scaling=SCALE,
        attention_mask=mask,
    )


def timed_call(capture: MockCapture) -> float:
    """Call thrml_sample and return wall time in seconds."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    jax.effects_barrier()  # drain any pending JAX work
    t0 = time.perf_counter()
    _ = thrml_sample(capture, K=K_SAMPLES, seed=SEED)
    jax.effects_barrier()  # wait for async GPU work
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  TASB THRML JIT Compilation Profile")
    print(f"  Device: {device} | n_q={N_Q} n_kv={N_KV} head_dim={HEAD_DIM}")
    print(f"  K={K_SAMPLES} samples per call (small — testing overhead only)")
    print(f"{'='*70}\n")

    # Columns: S, call1_time, call2_time, ratio, verdict
    results = []

    print(f"  {'S':>6}  {'call1 (s)':>10}  {'call2 (s)':>10}  {'ratio':>7}  verdict")
    print(f"  {'-'*56}")

    for S in S_VALUES:
        capture = make_capture(S, device=device)

        # Call 1: triggers compilation (or reuse if same shape seen before)
        t1 = timed_call(capture)
        # Call 2: should hit XLA cache
        t2 = timed_call(capture)

        ratio = t2 / t1 if t1 > 0 else float('inf')
        verdict = "CACHED" if ratio < 0.5 else "RECOMPILE?"

        print(f"  {S:>6}  {t1:>10.3f}  {t2:>10.3f}  {ratio:>7.3f}  {verdict}")
        results.append({
            "S": S,
            "call1_s": round(t1, 4),
            "call2_s": round(t2, 4),
            "ratio": round(ratio, 4),
            "verdict": verdict,
        })

    # Cache persistence: repeat S=10 after all other shapes
    print(f"\n  Cache persistence check (S=10 after full S sweep):")
    capture10 = make_capture(10, device=device)
    t_persist = timed_call(capture10)
    first_10_call1 = results[0]["call1_s"]
    persist_ratio = t_persist / first_10_call1 if first_10_call1 > 0 else float('inf')
    persist_ok = persist_ratio < 0.5
    print(f"  S=10 persist call: {t_persist:.3f}s  "
          f"ratio vs first call: {persist_ratio:.3f}  "
          f"{'CACHE HIT' if persist_ok else 'CACHE MISS'}")

    results.append({
        "S": "10_persist",
        "call1_s": first_10_call1,
        "call2_s": round(t_persist, 4),
        "ratio": round(persist_ratio, 4),
        "verdict": "CACHE_HIT" if persist_ok else "CACHE_MISS",
    })

    # Overall verdict
    recompile_hits = [r for r in results[:-1] if r["verdict"] == "RECOMPILE?"]
    passed = len(recompile_hits) == 0 and persist_ok

    print(f"\n{'='*70}")
    print(f"  RESULT: {'PASS' if passed else 'WARN'}")
    if recompile_hits:
        print(f"  Recompile suspects at S: {[r['S'] for r in recompile_hits]}")
        print(f"  These shapes may be triggering XLA retracing.")
        print(f"  Investigate with XLA_FLAGS='--xla_dump_to=/tmp/xla_dump'")
    else:
        print(f"  All shapes cached after first call.")
    if not persist_ok:
        print(f"  WARN: S=10 cache was evicted by larger shapes.")
        print(f"  XLA cache may be size-limited. Check XLA_CLIENT_MEM_FRACTION.")
    else:
        print(f"  Cache persistence confirmed: shapes not evicted.")
    print(f"{'='*70}")

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_thrml_jit_profile_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S", "call1_s", "call2_s", "ratio", "verdict"])
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
