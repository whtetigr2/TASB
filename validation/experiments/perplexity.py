# CRITICAL: JAX/XLA flags before any jax import
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_perplexity.py — WikiText-2 Perplexity Evaluation (T3)
==============================================================================
TASB Validation Suite — Tier 3 (Downstream Task Evaluation)
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

PURPOSE
-------
Answers FIND-008 Publication Blocker #1: no downstream perplexity benchmark.
Evaluates TASB bridge on WikiText-2 (standard LM benchmark) against vanilla
softmax baseline. Expected result: perplexity unchanged — TASB sampling is
weight-preserving by construction (zero confident flip theorem, FIND-008).

CONDITIONS EVALUATED
--------------------
  (a) vanilla     — pure model forward, no bridge
  (b) exact-K10   — exact backend, alpha=0.3, K=10   (production default)
  (c) exact-K50   — exact backend, alpha=0.3, K=50   (higher K)
  (d) thrml-K50   — thrml backend, alpha=0.3, K=50   (limited chunks)

PROTOCOL
--------
Dataset:    WikiText-2 raw (wikitext-2-raw-v1), test split
Encoding:   LLaMA tokenizer (LlamaTokenizer / AutoTokenizer)
Chunks:     Non-overlapping windows of STRIDE tokens
            For each chunk: NLL = mean(-log p(token_t | tokens_0..t-1))
Layer:      18 (production default)

PERPLEXITY FORMULA
------------------
For each chunk of length S:
  logits = f(input_ids)          # (1, S, vocab)
  shifted_logits = logits[:, :-1, :]    # predict tokens 1..S-1
  shifted_labels = input_ids[:, 1:]      # true targets 1..S-1
  nll = cross_entropy(shifted_logits.reshape(-1, vocab), shifted_labels.reshape(-1))
  PPL_chunk = exp(nll)

Overall PPL = exp(mean(nll across all chunks))

WHY THIS IS THE KEY RESULT
---------------------------
A skeptic's strongest objection to TASB: "stochastic sampling must degrade
quality." If PPL(TASB) ≈ PPL(vanilla) within noise, this objection fails
empirically. Combined with T2.C (chi-squared convergence to exact Boltzmann)
and T1.C (per-head KL uniformity), this creates a three-layer validation:
  Layer 1: sampler is mathematically correct (T2.C)
  Layer 2: sampler doesn't hurt individual attention heads (T1.C)
  Layer 3: sampler doesn't hurt downstream token prediction (T3)

MODEL
-----
LLaMA 3.2-3B from HuggingFace (frozen, float16, attn_implementation='eager').
Uses same model loading as existing TASB tests (M5–M7).

REPRODUCE
---------
  python experiments/tasb_perplexity.py
  python experiments/tasb_perplexity.py --thrml-chunks 10  # faster
  python experiments/tasb_perplexity.py --stride 256       # smaller chunks
==============================================================================
"""

import argparse
import csv
import datetime
import sys
import time
import math

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID      = "meta-llama/Llama-3.2-3B"
LAYER_IDX     = 18
ALPHA         = 0.3
STRIDE        = 512       # non-overlapping chunks (tokens)
THRML_CHUNKS  = 25        # max chunks to run for THRML (speed limit)
SEED          = 42

EVAL_CONDITIONS = [
    ("vanilla",   "exact",  0.0,  10),
    ("exact-K10", "exact",  0.3,  10),
    ("exact-K50", "exact",  0.3,  50),
    # thrml-K50 is handled separately with THRML_CHUNKS limit
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Model loaded on {device}")
    return model, tok


# ---------------------------------------------------------------------------
# WikiText-2 loading and tokenization
# ---------------------------------------------------------------------------
def load_wikitext2(tok, stride: int, device: str):
    """Load WikiText-2 test split, tokenize, split into chunks."""
    from datasets import load_dataset

    print(f"  Loading WikiText-2 test split ...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text, skipping blank lines
    text = "\n\n".join(
        item["text"] for item in dataset if item["text"].strip()
    )

    print(f"  Tokenizing ({len(text):,} characters) ...")
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]   # (1, N_total)
    N = input_ids.shape[1]
    print(f"  Total tokens: {N:,}")

    # Split into non-overlapping chunks of `stride` tokens
    # Each chunk has exactly `stride` tokens; last partial chunk is dropped
    chunks = []
    for start in range(0, N - stride, stride):
        end = start + stride
        chunk = input_ids[:, start:end].to(device)   # (1, stride)
        chunks.append(chunk)

    print(f"  Chunks (len={stride}): {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Vanilla perplexity (no bridge)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_vanilla(model, chunks: list) -> dict:
    """Standard causal LM perplexity: model(input_ids, labels=input_ids).loss"""
    print(f"\n  [vanilla] Evaluating {len(chunks)} chunks ...")
    t0 = time.perf_counter()

    nll_sum = 0.0
    n_chunks = 0
    chunk_times = []

    for i, chunk in enumerate(chunks):
        tc = time.perf_counter()
        out = model(input_ids=chunk, labels=chunk)
        # HuggingFace labels shift internally: loss = mean NLL over S-1 next-tokens
        nll = out.loss.item()
        nll_sum += nll
        n_chunks += 1
        chunk_times.append(time.perf_counter() - tc)

        if (i + 1) % 50 == 0:
            running_ppl = math.exp(nll_sum / n_chunks)
            print(f"    chunk {i+1}/{len(chunks)}  running PPL={running_ppl:.4f}  "
                  f"chunk_t={chunk_times[-1]:.3f}s")

    mean_nll = nll_sum / n_chunks
    ppl = math.exp(mean_nll)
    elapsed = time.perf_counter() - t0

    print(f"  [vanilla] PPL={ppl:.4f}  mean_NLL={mean_nll:.6f}  "
          f"total_t={elapsed:.1f}s  chunks={n_chunks}")

    return {
        "condition":    "vanilla",
        "backend":      "none",
        "alpha":        0.0,
        "K":            0,
        "n_chunks":     n_chunks,
        "mean_nll":     round(mean_nll, 6),
        "ppl":          round(ppl, 4),
        "elapsed_s":    round(elapsed, 2),
        "chunk_t_mean": round(float(np.mean(chunk_times)), 4),
    }


# ---------------------------------------------------------------------------
# Bridge perplexity
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_bridge(model, chunks: list, backend: str, alpha: float, K: int,
                layer_idx: int, seed: int, max_chunks=None,
                condition_name: str = "") -> dict:
    """
    Perplexity using bridge_forward() for each chunk.

    bridge_forward returns (1, S, vocab_size) logits from the inject pass.
    NLL = cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    Note: HuggingFace model() with labels= shifts internally; here we shift
    manually since bridge_forward returns raw logits.
    """
    from thermobridge import bridge_forward

    eval_chunks = chunks if max_chunks is None else chunks[:max_chunks]
    n_total = len(eval_chunks)
    label = condition_name or f"{backend}-K{K}"

    print(f"\n  [{label}] Evaluating {n_total} chunks "
          f"(alpha={alpha}, K={K}, layer={layer_idx}) ...")

    t0 = time.perf_counter()
    nll_sum = 0.0
    n_chunks = 0
    chunk_times = []

    for i, chunk in enumerate(eval_chunks):
        tc = time.perf_counter()

        # bridge_forward with alpha=0.0 is bit-exact vanilla (injector fast-path)
        logits = bridge_forward(
            model,
            tok=None,
            input_ids=chunk,        # (1, S), already on device
            layer_idx=layer_idx,
            alpha=alpha,
            backend=backend,
            K=K,
            seed=seed,
            strict_verify=False,    # skip invariant check in tight loop
            return_intermediates=False,
        )  # (1, S, vocab_size), float16 or float32

        # Shift: predict token t+1 from positions 0..t
        # logits[:, :-1, :] → predicts tokens at positions 1..S-1
        # chunk[:, 1:]       → true labels at positions 1..S-1
        shifted_logits = logits[:, :-1, :].float().contiguous()   # (1, S-1, V)
        shifted_labels = chunk[:, 1:].contiguous()                 # (1, S-1)

        nll = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
        ).item()

        nll_sum += nll
        n_chunks += 1
        chunk_times.append(time.perf_counter() - tc)

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            running_ppl = math.exp(nll_sum / n_chunks)
            print(f"    chunk {i+1}/{n_total}  running PPL={running_ppl:.4f}  "
                  f"chunk_t={chunk_times[-1]:.3f}s")

    mean_nll = nll_sum / n_chunks
    ppl = math.exp(mean_nll)
    elapsed = time.perf_counter() - t0

    print(f"  [{label}] PPL={ppl:.4f}  mean_NLL={mean_nll:.6f}  "
          f"total_t={elapsed:.1f}s  chunks={n_chunks}")

    return {
        "condition":    label,
        "backend":      backend,
        "alpha":        alpha,
        "K":            K,
        "n_chunks":     n_chunks,
        "mean_nll":     round(mean_nll, 6),
        "ppl":          round(ppl, 4),
        "elapsed_s":    round(elapsed, 2),
        "chunk_t_mean": round(float(np.mean(chunk_times)), 4),
    }


# ---------------------------------------------------------------------------
# Summary / delta table
# ---------------------------------------------------------------------------
def print_summary(results: list):
    print(f"\n{'='*72}")
    print(f"  T3 PERPLEXITY SUMMARY — WikiText-2 test split")
    print(f"{'='*72}")

    # Find vanilla baseline
    baseline = next((r for r in results if r["condition"] == "vanilla"), None)
    if baseline is None:
        baseline_ppl = float('nan')
    else:
        baseline_ppl = baseline["ppl"]

    print(f"  {'Condition':<14} {'Backend':<10} {'alpha':>5} {'K':>5} "
          f"{'Chunks':>6} {'PPL':>9} {'ΔPPL':>9} {'sec/chunk':>10}")
    print(f"  {'-'*75}")

    for r in results:
        delta = r["ppl"] - baseline_ppl if not math.isnan(baseline_ppl) else float('nan')
        delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "   —"
        print(f"  {r['condition']:<14} {r['backend']:<10} {r['alpha']:>5.2f} "
              f"{r['K']:>5} {r['n_chunks']:>6} {r['ppl']:>9.4f} "
              f"{delta_str:>9} {r['chunk_t_mean']:>10.3f}s")

    print(f"{'='*72}")

    # Verdict
    if baseline is not None:
        exact_results = [r for r in results if r["backend"] == "exact" and r["alpha"] > 0]
        if exact_results:
            max_delta = max(abs(r["ppl"] - baseline_ppl) for r in exact_results)
            if max_delta < 0.5:
                print(f"  RESULT: PASS — max |ΔPPL| = {max_delta:.4f} < 0.5 (within noise)")
                print(f"  TASB exact backend does not degrade perplexity.")
            else:
                print(f"  RESULT: WARN — max |ΔPPL| = {max_delta:.4f} ≥ 0.5")
                print(f"  Investigate: stochastic attention may affect NLL.")

    print(f"  Note: PPL differences < 1.0 are within typical run-to-run noise")
    print(f"  for stochastic models. THRML chunks may be fewer than exact.")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TASB WikiText-2 Perplexity Evaluation (T3)")
    parser.add_argument("--stride",       type=int, default=STRIDE)
    parser.add_argument("--layer",        type=int, default=LAYER_IDX)
    parser.add_argument("--alpha",        type=float, default=ALPHA)
    parser.add_argument("--seed",         type=int, default=SEED)
    parser.add_argument("--thrml-chunks", type=int, default=THRML_CHUNKS,
                        dest="thrml_chunks")
    parser.add_argument("--no-thrml",     action="store_true", dest="no_thrml",
                        help="Skip THRML condition (faster)")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  TASB T3 — WikiText-2 Perplexity Evaluation")
    print(f"  Layer: {args.layer}  |  alpha={args.alpha}  |  stride={args.stride}")
    print(f"  THRML max chunks: {args.thrml_chunks}  |  Device: {args.device}")
    print(f"  Conditions: vanilla, exact-K10, exact-K50"
          + (", thrml-K50" if not args.no_thrml else " (thrml skipped)"))
    print(f"{'='*72}\n")

    model, tok = load_model(args.device)
    chunks = load_wikitext2(tok, args.stride, args.device)

    results = []

    # (a) Vanilla
    results.append(eval_vanilla(model, chunks))

    # (b) exact-K10  (production default)
    results.append(eval_bridge(
        model, chunks,
        backend="exact", alpha=args.alpha, K=10,
        layer_idx=args.layer, seed=args.seed,
        condition_name="exact-K10",
    ))

    # (c) exact-K50
    results.append(eval_bridge(
        model, chunks,
        backend="exact", alpha=args.alpha, K=50,
        layer_idx=args.layer, seed=args.seed,
        condition_name="exact-K50",
    ))

    # (d) thrml-K50 (limited chunks)
    if not args.no_thrml:
        thrml_chunks = min(args.thrml_chunks, len(chunks))
        results.append(eval_bridge(
            model, chunks,
            backend="thrml", alpha=args.alpha, K=50,
            layer_idx=args.layer, seed=args.seed,
            max_chunks=thrml_chunks,
            condition_name="thrml-K50",
        ))
        # Also run vanilla on same THRML subset for fair comparison
        print(f"\n  [vanilla-thrml-subset] Evaluating first {thrml_chunks} chunks ...")
        subset_result = eval_vanilla(model, chunks[:thrml_chunks])
        subset_result["condition"] = "vanilla-subset"
        results.append(subset_result)

    print_summary(results)

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tasb_perplexity_{ts}.csv"
    fieldnames = ["condition", "backend", "alpha", "K", "n_chunks",
                  "mean_nll", "ppl", "elapsed_s", "chunk_t_mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {out_path}")

    # Return 0 if all exact conditions pass, 1 if any WARN
    baseline_ppl = next(
        (r["ppl"] for r in results if r["condition"] == "vanilla"), float("nan"))
    exact_res = [r for r in results
                 if r["backend"] == "exact" and r["alpha"] > 0]
    if exact_res and not math.isnan(baseline_ppl):
        max_delta = max(abs(r["ppl"] - baseline_ppl) for r in exact_res)
        return 0 if max_delta < 0.5 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
