#!/system/conda/miniconda3/envs/cloudspace/bin/python
"""
capture_qk_layer18.py  --  EXP-001 Phase 1

NOTE (2026-07-04, Alfred): historical/completed -- output
layer18_qk_pre_rope.npz already exists (2026-07-01). Do not rerun as-is:
TASB_PATH below points to a location that moved to relics/Active_Dev/TASB,
AND that capture module was rewritten 2026-07-03/04 (LlamaAttentionCapture
no longer has _rope_scratch, which this script reads directly) -- would
need a real port, not a path fix, to work again.

Capture pre-RoPE Q and K tensors at LLaMA 3.2-3B layer 18.

Output:
  /teamspace/studios/this_studio/claude_work/thermobridge_cv/audit/layer18_qk_pre_rope.npz
  Keys: q_pre_<key>, k_pre_<key>  for key in [factual, code, reasoning, creative, mathematical]
  Shapes: q_pre_* -> (n_q_heads=24, S, 128),  k_pre_* -> (n_kv_heads=8, S, 128)
"""

import os
import sys
import numpy as np
import torch

TASB_PATH = '/teamspace/studios/this_studio/claude_work/Active_Dev/TASB'
SAVE_PATH = '/teamspace/studios/this_studio/claude_work/thermobridge_cv/audit/layer18_qk_pre_rope.npz'
MODEL_ID  = 'meta-llama/Llama-3.2-3B'
LAYER     = 18

PROMPTS = [
    ("factual",     "The capital of France is Paris. The capital of Germany is"),
    ("code",        "def fibonacci(n): if n <= 1: return n else: return fibonacci"),
    ("reasoning",   "All mammals are warm-blooded. Whales are mammals. Therefore,"),
    ("creative",    "In the year 2150, when gravity was finally understood as"),
    ("mathematical","The Riemann zeta function zeta(s) has non-trivial zeros at s ="),
]

def main():
    sys.path.insert(0, TASB_PATH)
    from tasb_capture_v2 import LlamaAttentionCapture
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("[load] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[load] Loading model (4-bit nf4)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    print(f"[load] Ready on {next(model.parameters()).device}", flush=True)

    capturer = LlamaAttentionCapture(
        model=model,
        layers_to_capture=[LAYER],
        strict_verify=False,
    )

    results = {}
    for i, (key, prompt) in enumerate(PROMPTS):
        print(f"[{i+1}/5] Prompt '{key}' ...", flush=True)
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        with capturer.capture():
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)

        rs = capturer._rope_scratch[LAYER]
        q_pre = rs['q_pre_rope'].cpu().float().numpy()   # (B, n_q_heads, S, 128)
        k_pre = rs['k_pre_rope'].cpu().float().numpy()   # (B, n_kv_heads, S, 128)

        results[f'q_pre_{key}'] = q_pre[0]   # drop batch dim
        results[f'k_pre_{key}'] = k_pre[0]
        print(f"       q_pre: {q_pre[0].shape}  k_pre: {k_pre[0].shape}", flush=True)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(SAVE_PATH, **results)
    size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"\n[done] Saved {len(results)} arrays -> {SAVE_PATH}  ({size_kb:.1f} KB)", flush=True)


if __name__ == '__main__':
    main()
