"""
test_rope_bf16_roundtrip.py — Confirm residual is bf16 quantization, not design bug
==============================================================================
Combines Codex's targeted bf16-roundtrip test with a source dump of the actual
LlamaAttention.forward we'll need to patch in Stage 1.

TWO QUESTIONS THIS ANSWERS
--------------------------
Q1 (Codex's test): If we softmax in fp32 then round to bf16 then back to
   fp32 — matching what LlamaAttention does at modeling_llama.py:189 —
   does the residual diff vs captured weights collapse to near zero?

   PASS = residual is bf16 quantization on the probabilities themselves.
          The post-RoPE reconstruction is conceptually correct.
          Tighten invariant to bf16-aware bounds. Refactor proceeds.

   FAIL = something else is going on. Possibilities: position_embeddings
          mismatch, repeat_kv variant, scaling factor we missed, or a real
          RoPE replication bug. Continue diagnosing before refactor.

Q2 (Stage 1 prep): What does LlamaAttention.forward actually do in this
   transformers version? Print it so we know what tasb_capture_v2.py
   needs to monkey-patch.

==============================================================================
"""

import sys, math, inspect
import numpy as np


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, apply_rotary_pos_emb)

    MODEL_ID = "meta-llama/Llama-3.2-3B"
    TEST_LAYER = 18
    PROMPT = "The capital of France is"

    HEAD_DIM = 128
    NUM_Q_HEADS = 24
    NUM_KV_HEADS = 8
    KV_GROUPS = NUM_Q_HEADS // NUM_KV_HEADS
    T_STRUCT = math.sqrt(HEAD_DIM)

    print("=" * 78)
    print("  BF16-ROUNDTRIP TEST + SOURCE DUMP")
    print("=" * 78)

    print("[setup] Loading...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),
        attn_implementation='eager', device_map='auto')
    model.eval()
    device = next(model.parameters()).device
    print(f"[setup] Ready on {device}\n")

    # ── Q2: Source dump for Stage 1 design ──────────────────────────────────
    print("=" * 78)
    print("  LlamaAttention.forward source (for Stage 1 monkey-patch design)")
    print("=" * 78)
    print(inspect.getsource(LlamaAttention.forward))
    print("=" * 78)

    try:
        from transformers.models.llama.modeling_llama import eager_attention_forward
        print("\n  eager_attention_forward source:")
        print("=" * 78)
        print(inspect.getsource(eager_attention_forward))
        print("=" * 78)
    except ImportError:
        print("\n  (no separate eager_attention_forward; check forward source above)")

    # ── Q1: Codex's bf16-roundtrip test ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Q1: Does bf16-roundtrip on probabilities collapse the residual?")
    print("=" * 78)

    # Capture pre-RoPE Q, K and the HF attention weights
    captured = {'q': None, 'k': None, 'w': None, 'w_dtype': None}
    attn_module = model.model.layers[TEST_LAYER].self_attn

    def hk_q(m, a, o): captured['q'] = o.detach().clone()
    def hk_k(m, a, o): captured['k'] = o.detach().clone()
    def hk_w(m, a, kw, o):
        if len(o) > 1 and o[1] is not None:
            captured['w_dtype'] = o[1].dtype
            captured['w'] = o[1].detach().float().cpu()
    h1 = attn_module.q_proj.register_forward_hook(hk_q)
    h2 = attn_module.k_proj.register_forward_hook(hk_k)
    h3 = attn_module.register_forward_hook(hk_w, with_kwargs=True)

    inputs = tok(PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        _ = model(**inputs, output_attentions=True, use_cache=False)

    h1.remove(); h2.remove(); h3.remove()

    print(f"  Captured weights dtype: {captured['w_dtype']}")
    print(f"  Captured q_proj dtype:  {captured['q'].dtype}")

    B, S, _ = captured['q'].shape

    Q = captured['q'].view(B, S, NUM_Q_HEADS,  HEAD_DIM).transpose(1, 2)
    K = captured['k'].view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)

    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(S, device=device).unsqueeze(0)
    dummy = torch.zeros(B, S, HEAD_DIM, device=device, dtype=torch.float32)
    cos, sin = rotary_emb(dummy, position_ids)

    Q_post, K_post = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=1)
    K_post_exp = K_post.repeat_interleave(KV_GROUPS, dim=1)
    scores = torch.matmul(Q_post, K_post_exp.transpose(-2, -1)) / T_STRUCT

    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)
    scores_masked = scores.masked_fill(causal, float('-inf'))

    # Path A: pure fp32 softmax (what we did before)
    probs_fp32 = torch.softmax(scores_masked, dim=-1)

    # Path B: fp32 softmax then bf16 roundtrip (Codex's test)
    probs_fp32_then_bf16 = probs_fp32.to(torch.bfloat16).to(torch.float32)

    hf_w = captured['w'].numpy()
    lower_tri = ~causal.cpu().numpy()

    diff_A = np.abs(probs_fp32.cpu().numpy() - hf_w)[..., lower_tri]
    diff_B = np.abs(probs_fp32_then_bf16.cpu().numpy() - hf_w)[..., lower_tri]

    print(f"\n  Path A (pure fp32 softmax):")
    print(f"    max abs diff:  {diff_A.max():.6f}")
    print(f"    mean abs diff: {diff_A.mean():.6f}")
    print(f"    99th pct:      {np.percentile(diff_A, 99):.6f}")

    print(f"\n  Path B (fp32 softmax → bf16 → fp32, matches HF):")
    print(f"    max abs diff:  {diff_B.max():.6f}")
    print(f"    mean abs diff: {diff_B.mean():.6f}")
    print(f"    99th pct:      {np.percentile(diff_B, 99):.6f}")

    # Reduction in residual
    reduction_max  = (diff_A.max()  - diff_B.max())  / max(diff_A.max(),  1e-10)
    reduction_mean = (diff_A.mean() - diff_B.mean()) / max(diff_A.mean(), 1e-10)

    print(f"\n  Reduction by bf16 roundtrip:")
    print(f"    max:  {100*reduction_max:>+6.1f}%")
    print(f"    mean: {100*reduction_mean:>+6.1f}%")

    # Verdict
    print("\n" + "=" * 78)
    if diff_B.max() < 1e-4:
        print(f"  \033[32m✓ Residual collapses under bf16 roundtrip.\033[0m")
        print(f"  Post-RoPE reconstruction is correct. Residual was HF's")
        print(f"  softmax(..., dtype=fp32).to(query.dtype) bf16 cast.")
        print(f"  Tighten invariant: bf16-aware comparison passes at <1e-4.")
        print(f"  Refactor proceeds to Stage 1.")
        verdict = "BF16_CONFIRMED"
    elif diff_B.max() < diff_A.max() * 0.5:
        print(f"  \033[33m~ Partial collapse.\033[0m bf16 roundtrip helps but doesn't fully")
        print(f"  explain the residual. There's another small source — investigate")
        print(f"  position_embeddings source and repeat_kv variant next.")
        verdict = "PARTIAL"
    else:
        print(f"  \033[31m✗ bf16 roundtrip did not collapse the residual.\033[0m")
        print(f"  The post-RoPE reconstruction has another issue. Check:")
        print(f"  - position_embeddings: are we using the same cos/sin as L18 received?")
        print(f"  - repeat_kv: does HF use the same expansion rule we used?")
        print(f"  - any operation between matmul and softmax in the source above?")
        verdict = "STILL_DIFFERS"
    print("=" * 78)

    # ── Recommended bounds for Stage 1 invariant ───────────────────────────
    print(f"\n  Empirical bounds from this run (for tasb_verify_v2.py):")
    print(f"    bf16-aware:  max < {max(diff_B.max() * 2, 1e-4):.2e}, "
          f"mean < {max(diff_B.mean() * 3, 1e-5):.2e}")
    print(f"    fp32 control (when achievable): max < 1e-4, mean < 1e-5")

    return verdict


if __name__ == '__main__':
    main()
