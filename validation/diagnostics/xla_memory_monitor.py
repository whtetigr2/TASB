# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_xla_memory_monitor.py — XLA HBM Monotonicity Test
==============================================================================
TASB Validation Suite — Tier 1.B
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

WHAT THIS MEASURES
------------------
The THRML backend creates JAX DeviceArrays inside XLA's memory manager.
Standard Python GC doesn't release these — they're collected by JAX's own
memory manager. If intermediate attention distributions (p_thermo) aren't
released, VRAM climbs monotonically across tokens, eventually OOM-ing.

This test runs 50-token generation with the THRML backend active and measures
GPU memory (VRAM via nvidia-smi and JAX profiler) after every token. A flat
memory profile after warmup = no leak. A staircase = accumulated arrays.

Why this matters: the THRML backend does O(S^2) JAX work per token where
S grows with context. With XLA_PYTHON_CLIENT_PREALLOCATE=false, the allocator
grows the heap on demand. If old allocations aren't released, the heap only
ratchets up.

KNOWN CONTEXT (BUG_REGISTRY #21 reference)
-------------------------------------------
The XLA_PYTHON_CLIENT_PREALLOCATE=false + XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
flags are already set at the top of tasb_llama32_chat_runtime.py. That fixes
the INITIAL allocation spike (13.8GB → 2.6GB). This test checks for
ONGOING memory growth across tokens (different problem).

PASS CONDITION
--------------
After token 5 (warmup), VRAM should be flat within ±5% of the mean.
Monotonically increasing VRAM across 50 tokens = likely DeviceArray leak.

REPRODUCE
---------
  python diagnostics/tasb_xla_memory_monitor.py
  python diagnostics/tasb_xla_memory_monitor.py --tokens 25  # faster
==============================================================================
"""

import argparse
import csv
import datetime
import subprocess
import sys
import time

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID    = "meta-llama/Llama-3.2-3B"
LAYER_IDX   = 18
ALPHA       = 0.3
BACKEND     = "thrml"
K           = 50
SEED        = 42
N_TOKENS    = 50
WARMUP_SKIP = 5   # tokens to skip before measuring monotonicity

PROMPT = "The relationship between thermodynamics and computation is"


# ---------------------------------------------------------------------------
# VRAM polling
# ---------------------------------------------------------------------------
def get_vram_mb() -> float:
    """Poll current GPU VRAM used in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        )
        return float(out.decode().strip().split('\n')[0])
    except Exception:
        return float('nan')


def get_jax_memory_bytes() -> int:
    """Return current JAX device memory usage in bytes, if available."""
    try:
        import jax
        backend = jax.lib.xla_bridge.get_backend()
        # Try get_device_memory_info (available on some JAX versions)
        for method in ['get_device_memory_info', 'memory_stats']:
            fn = getattr(backend, method, None)
            if fn:
                info = fn()
                if isinstance(info, dict):
                    return int(info.get('bytes_in_use', info.get('in_use', 0)))
        return -1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str):
    print(f"  Loading {model_id} in 4-bit (nf4)...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model ready on {device}\n", flush=True)
    return model, tok


# ---------------------------------------------------------------------------
# Token-by-token generation with VRAM monitoring
# ---------------------------------------------------------------------------
def run_monitored_generation(model, tok, n_tokens: int) -> list:
    """Generate n_tokens one at a time, measuring VRAM after each."""
    from thermobridge import bridge_forward

    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tok(PROMPT, return_tensors='pt').to(device)['input_ids']

    rows = []
    print(f"  {'tok':>4}  {'VRAM(MB)':>10}  {'JAX(MB)':>10}  elapsed(s)")
    print(f"  {'-'*42}")

    for step in range(n_tokens):
        vram_before = get_vram_mb()
        jax_before  = get_jax_memory_bytes()

        t0 = time.perf_counter()

        # Single forward pass (capture + inject) — extends context by 1 logit step
        with torch.no_grad():
            result = bridge_forward(
                model, None,
                input_ids=input_ids,
                layer_idx=LAYER_IDX,
                alpha=ALPHA,
                backend=BACKEND,
                K=K,
                seed=SEED,
                return_intermediates=False,
            )

        elapsed = time.perf_counter() - t0

        # Sample next token (greedy)
        next_token = result[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
        input_ids  = torch.cat([input_ids, next_token], dim=1)

        # Force GC of any cached tensors
        import gc
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import jax
            jax.clear_backends()  # nudge JAX GC if available
        except Exception:
            pass

        vram_after = get_vram_mb()
        jax_after  = get_jax_memory_bytes()

        jax_mb = jax_after / 1024**2 if jax_after >= 0 else float('nan')
        print(f"  {step+1:>4}  {vram_after:>10.1f}  {jax_mb:>10.1f}  {elapsed:.2f}")

        rows.append({
            "step":        step + 1,
            "vram_mb":     round(vram_after, 1),
            "jax_bytes":   jax_after,
            "elapsed_s":   round(elapsed, 3),
            "seq_len":     input_ids.shape[1],
        })

    return rows


# ---------------------------------------------------------------------------
# Monotonicity analysis
# ---------------------------------------------------------------------------
def analyze_monotonicity(rows: list, warmup: int = WARMUP_SKIP) -> dict:
    """Check for monotonically increasing VRAM after warmup."""
    import numpy as np

    post_warmup = [r for r in rows if r['step'] > warmup]
    if not post_warmup:
        return {"pass": False, "reason": "No post-warmup data"}

    vram = np.array([r['vram_mb'] for r in post_warmup])
    mean = np.mean(vram)
    std  = np.std(vram)
    peak = np.max(vram)
    low  = np.min(vram)

    # Linear trend (slope > 0 = growing)
    steps = np.arange(len(vram), dtype=float)
    slope = np.polyfit(steps, vram, 1)[0] if len(vram) > 1 else 0.0

    # Variation from mean
    max_deviation_pct = (peak - mean) / mean * 100 if mean > 0 else 0.0

    # Pass: deviation < 5% of mean and slope < 1 MB/token
    flat = max_deviation_pct < 5.0 and abs(slope) < 1.0

    return {
        "pass":             flat,
        "mean_vram_mb":     round(float(mean), 1),
        "std_vram_mb":      round(float(std), 1),
        "max_vram_mb":      round(float(peak), 1),
        "min_vram_mb":      round(float(low), 1),
        "slope_mb_per_tok": round(float(slope), 3),
        "max_deviation_pct":round(max_deviation_pct, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB XLA Memory Monitor")
    parser.add_argument("--tokens", type=int, default=N_TOKENS)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--backend", default=BACKEND,
                        choices=["exact", "thrml"],
                        help="Backend to monitor (thrml is the important one)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  TASB XLA Memory Monotonicity Test (Tier 1.B Validation)")
    print(f"  Backend: {args.backend} | Layer {LAYER_IDX} | α={ALPHA} | K={K}")
    print(f"  Tokens: {args.tokens} | Warmup skip: {WARMUP_SKIP}")
    print(f"  XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'not set')}")
    print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'not set')}")
    print(f"{'='*70}")

    vram_baseline = get_vram_mb()
    print(f"\n  VRAM at start (baseline): {vram_baseline:.1f} MB")

    model, tok = load_model(args.model)

    vram_after_load = get_vram_mb()
    print(f"  VRAM after model load: {vram_after_load:.1f} MB")
    print(f"  Model memory: {vram_after_load - vram_baseline:.1f} MB\n")

    rows = run_monitored_generation(model, tok, args.tokens)

    stats = analyze_monotonicity(rows, WARMUP_SKIP)

    print(f"\n{'='*70}")
    print(f"  Memory analysis (post-warmup, tokens {WARMUP_SKIP+1}–{args.tokens}):")
    print(f"  Mean VRAM:     {stats['mean_vram_mb']:.1f} MB")
    print(f"  Std VRAM:      {stats['std_vram_mb']:.1f} MB")
    print(f"  Range:         {stats['min_vram_mb']:.1f} – {stats['max_vram_mb']:.1f} MB")
    print(f"  Slope:         {stats['slope_mb_per_tok']:.3f} MB/token")
    print(f"  Max deviation: {stats['max_deviation_pct']:.1f}% of mean")
    print(f"\n  RESULT: {'PASS — memory is flat' if stats['pass'] else 'WARN — possible memory leak'}")
    if not stats['pass']:
        if stats['slope_mb_per_tok'] >= 1.0:
            print(f"  Slope {stats['slope_mb_per_tok']:.1f} MB/tok suggests accumulated DeviceArrays.")
            print(f"  Check: are p_thermo tensors being GC'd after each token?")
        if stats['max_deviation_pct'] >= 5.0:
            print(f"  {stats['max_deviation_pct']:.1f}% variation exceeds 5% threshold.")
    print(f"{'='*70}")

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_xla_memory_monitor_{ts}.csv"
    fieldnames = ["step", "vram_mb", "jax_bytes", "elapsed_s", "seq_len"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    # Append summary row
    with open(out_path.replace(".csv", "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)
    print(f"  Saved: {out_path}")

    return 0 if stats['pass'] else 1


if __name__ == "__main__":
    sys.exit(main())
