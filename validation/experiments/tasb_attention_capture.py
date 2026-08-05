import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

"""
tasb_attention_capture.py — Actual Attention Matrix Capture for Demo
==============================================================================
TASB Demo Data Collection
Author: Paul W. Shaver

WHAT THIS CAPTURES
------------------
For each of 5 real prompts, captures the full softmax attention matrix
(shape: S × S) for every head across all 28 layers. Saves token labels
so the demo can display real token strings on the axes.

This is the data behind the "Real Attention Viewer" demo tab — showing
actual per-token attention behavior from a frozen LLaMA 3.2-3B model,
not synthetic random matrices.

Output: demo/data/attention_matrices.json
Structure:
  {
    "factual": {
      "tokens": ["The", " capital", " of", ...],
      "n_tokens": 12,
      "layers": {
        "0": {"heads": {"0": [[...]], "1": [[...]], ..., "23": [[...]]}},
        ...
        "27": {...}
      }
    },
    ...
  }

REPRODUCE
---------
  python experiments/tasb_attention_capture.py
  python experiments/tasb_attention_capture.py --prompt factual   # single prompt
==============================================================================
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
ALL_LAYERS = list(range(28))
ALPHA      = 0.0   # capture only — no injection, frozen model behavior
K          = 1     # minimal; we compute softmax analytically
SEED       = 42

PROMPTS = {
    "factual":      "The capital of France is Paris. The capital of Germany is",
    "code":         "def fibonacci(n): if n <= 1: return n else: return fibonacci",
    "reasoning":    "All mammals are warm-blooded. Whales are mammals. Therefore,",
    "creative":     "In the year 2150, when gravity was finally understood as",
    "mathematical": "The Riemann zeta function zeta(s) has non-trivial zeros at s =",
}

# Output path relative to this script (goes up to repo root, then into demo/data)
_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT_PATH = os.path.join(_REPO, "demo", "data", "attention_matrices.json")


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
# Token label extraction
# ---------------------------------------------------------------------------
def get_token_labels(tok, prompt: str) -> list:
    """Return human-readable token strings for a prompt."""
    input_ids = tok.encode(prompt, return_tensors="pt")[0]
    labels = []
    for id_ in input_ids:
        raw = tok.decode([int(id_)])
        # Clean up SentencePiece ▁ prefix and leading space
        clean = raw.replace("▁", " ").replace("▁", " ")
        labels.append(clean if clean else raw)
    return labels


# ---------------------------------------------------------------------------
# Compute softmax attention from capture
# ---------------------------------------------------------------------------
def compute_softmax(capture) -> torch.Tensor:
    """Returns (n_q, S, S) float32 — softmax attention for all heads."""
    from transformers.models.llama.modeling_llama import repeat_kv
    Q = capture.q_post_rope.float()   # (1, n_q, S, head_dim)
    K = capture.k_post_rope.float()   # (1, n_kv, S, head_dim)
    n_q, n_kv = Q.shape[1], K.shape[1]
    K_rep = repeat_kv(K, n_q // n_kv)
    logits = torch.matmul(Q, K_rep.transpose(-2, -1)) * float(capture.scaling)
    if capture.attention_mask is not None:
        logits = logits + capture.attention_mask.float()
    return torch.softmax(logits, dim=-1)[0]  # (n_q, S, S)


# ---------------------------------------------------------------------------
# Main capture
# ---------------------------------------------------------------------------
def capture_prompt(model, tok, prompt_id: str, prompt: str) -> dict:
    from thermobridge import bridge_forward

    tokens = get_token_labels(tok, prompt)
    n_tok  = len(tokens)
    print(f"\n  Prompt: {prompt_id!r}  ({n_tok} tokens)", flush=True)
    print(f"  Tokens: {tokens}", flush=True)

    t0 = time.perf_counter()
    result = bridge_forward(
        model, tok,
        prompt=prompt,
        layer_idx=ALL_LAYERS,
        alpha=ALPHA,
        backend='exact',
        K=K,
        seed=SEED,
        return_intermediates=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Forward pass: {elapsed:.1f}s", flush=True)

    layers_data = {}
    for layer in ALL_LAYERS:
        capture  = result.layer_captures[layer]
        softmax  = compute_softmax(capture)  # (n_q, S, S)
        n_q      = softmax.shape[0]

        heads_data = {}
        for h in range(n_q):
            mat = softmax[h].cpu().float().numpy()  # (S, S)
            # Round to 4 decimal places to keep JSON compact
            heads_data[str(h)] = [[round(float(v), 4) for v in row] for row in mat]

        layers_data[str(layer)] = {"heads": heads_data}
        if layer % 7 == 0:
            print(f"    Layer {layer:2d} captured  (n_q={n_q}, S={softmax.shape[1]})", flush=True)

    return {
        "tokens":   tokens,
        "n_tokens": n_tok,
        "layers":   layers_data,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TASB Attention Matrix Capture")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt", choices=list(PROMPTS.keys()),
                        help="Capture a single prompt (default: all 5)")
    args = parser.parse_args()

    prompts = ({args.prompt: PROMPTS[args.prompt]} if args.prompt else PROMPTS)

    print(f"\n{'='*70}")
    print(f"  TASB Attention Matrix Capture — Real Token Attention for Demo")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {list(prompts.keys())}")
    print(f"  All 28 layers | 24 heads | alpha=0 (frozen model, no injection)")
    print(f"  Output: {OUT_PATH}")
    print(f"{'='*70}")

    model, tok = load_model(args.model)

    out = {}
    for prompt_id, prompt in prompts.items():
        out[prompt_id] = capture_prompt(model, tok, prompt_id, prompt)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(',', ':'))

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  Prompts captured: {list(out.keys())}")
    for pid, data in out.items():
        print(f"    {pid}: {data['n_tokens']} tokens, {len(data['layers'])} layers")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
