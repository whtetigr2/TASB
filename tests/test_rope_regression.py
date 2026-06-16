"""
test_rope_regression.py — The Anchor Test
==============================================================================
This test exists to lock in the foundational bug fix for the TASB refactor.

WHAT IT PROVES
--------------
The pre-refactor capture path in `tasb_two_pass.py` reconstructs raw_scores as
`Q @ K.T / sqrt(d_k)` from the OUTPUTS of `q_proj` and `k_proj`. But in LLaMA,
RoPE (rotary position embedding) is applied AFTER the projections and BEFORE
the matrix multiply. The actual scores the model computes are
`Q_rotated @ K_rotated.T / sqrt(d_k)`.

So pre-refactor: `softmax(raw_scores) != captured weights` by construction —
not because of float precision, not because of GQA averaging, but because the
two tensors come from different points in the computation graph.

This test confirms that empirically:

  TEST 1 (regression):  pre-RoPE reconstruction FAILS the invariant.
  TEST 2 (forward):     post-RoPE reconstruction PASSES the invariant.

When TEST 1 fails (= confirms broken behavior) and TEST 2 passes (= confirms
fix), the refactor anchor is established. Stage 1 work can proceed.

If TEST 1 passes (= we DON'T see the bug we claim exists), STOP. We have
misdiagnosed the problem and need to re-examine the code before changing it.

USAGE
-----
    python -m tests.test_rope_regression
or
    python tests/test_rope_regression.py

EXPECTED OUTPUT
---------------
    [TEST 1] Pre-RoPE capture vs HF weights:    FAIL  (max diff > 0.05) ← expected
    [TEST 2] Post-RoPE capture vs HF weights:   PASS  (max diff < 1e-4) ← expected

    ✓ Regression test confirmed: RoPE bug is real. Refactor anchor established.

==============================================================================
"""

import sys, math
import numpy as np

# Tolerance for the post-RoPE invariant. fp32 softmax vs the bf16 path inside
# LLaMA's attention can introduce small differences. 1e-3 absolute is the
# generous-but-defensible bound that catches real bugs without false alarms.
INVARIANT_TOL = 1e-3

# A pre-RoPE capture should be wildly different from the real weights. If max
# diff is under this, we have misunderstood the bug.
REGRESSION_MIN_DIFF = 0.05


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_ID = "meta-llama/Llama-3.2-3B"
    TEST_LAYER = 18   # v2 winner; representative late layer
    PROMPT = "The capital of France is"

    print("=" * 78)
    print("  ROPE REGRESSION TEST — anchor for TASB Refactor")
    print("=" * 78)
    print(f"  Model:  {MODEL_ID}")
    print(f"  Layer:  L{TEST_LAYER}")
    print(f"  Prompt: {PROMPT!r}")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    print("[setup] Loading model (4-bit quant)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[setup] Ready on {device}\n", flush=True)

    # Architecture constants
    HEAD_DIM     = 128
    NUM_Q_HEADS  = 24
    NUM_KV_HEADS = 8
    KV_GROUPS    = NUM_Q_HEADS // NUM_KV_HEADS   # 3
    T_STRUCT     = math.sqrt(HEAD_DIM)           # √128

    # ── Capture infrastructure ──────────────────────────────────────────────
    captured = {
        'q_proj_out': None,   # pre-RoPE Q (current broken path)
        'k_proj_out': None,   # pre-RoPE K (current broken path)
        'hf_weights': None,   # what the model actually used (ground truth)
    }

    attn_module = model.model.layers[TEST_LAYER].self_attn

    # Hook 1: pre-RoPE Q — mirrors current tasb_two_pass.py exactly
    def hook_q(module, args, output):
        captured['q_proj_out'] = output.detach().float().cpu()
    h_q = attn_module.q_proj.register_forward_hook(hook_q)

    # Hook 2: pre-RoPE K — mirrors current tasb_two_pass.py exactly
    def hook_k(module, args, output):
        captured['k_proj_out'] = output.detach().float().cpu()
    h_k = attn_module.k_proj.register_forward_hook(hook_k)

    # Hook 3: actual attention weights (post-softmax, post-RoPE, post-mask)
    def hook_attn(module, args, kwargs, output):
        if len(output) > 1 and output[1] is not None:
            captured['hf_weights'] = output[1].detach().float().cpu()
    h_attn = attn_module.register_forward_hook(hook_attn, with_kwargs=True)

    # ── Run forward pass ────────────────────────────────────────────────────
    print("[capture] Running forward pass with hooks attached...", flush=True)
    inputs = tok(PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        _ = model(**inputs, output_attentions=True, use_cache=False)

    h_q.remove(); h_k.remove(); h_attn.remove()

    q_out = captured['q_proj_out']
    k_out = captured['k_proj_out']
    hf_w  = captured['hf_weights']

    if q_out is None or k_out is None or hf_w is None:
        print("FATAL: capture incomplete. Q={}, K={}, W={}".format(
            q_out is not None, k_out is not None, hf_w is not None))
        sys.exit(2)

    B, S, _ = q_out.shape
    assert q_out.shape == (B, S, NUM_Q_HEADS * HEAD_DIM), \
        f"Q shape: {q_out.shape}"
    assert k_out.shape == (B, S, NUM_KV_HEADS * HEAD_DIM), \
        f"K shape: {k_out.shape}"
    assert hf_w.shape == (B, NUM_Q_HEADS, S, S), \
        f"W shape: {hf_w.shape}"

    print(f"[capture] Captured: Q_proj{tuple(q_out.shape)}, "
          f"K_proj{tuple(k_out.shape)}, weights{tuple(hf_w.shape)}\n",
          flush=True)

    # ── TEST 1: pre-RoPE reconstruction (current broken path) ───────────────
    print("[TEST 1] Reconstructing scores from PRE-RoPE q_proj/k_proj outputs")
    print("         (this mirrors tasb_two_pass.py:_try_compute_raw_scores)\n")

    # Reshape: (B, S, n*dh) -> (B, n, S, dh)
    Q_pre = q_out.view(B, S, NUM_Q_HEADS,  HEAD_DIM).transpose(1, 2)
    K_pre = k_out.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    # GQA: expand K to 24 heads
    K_pre_exp = K_pre.repeat_interleave(KV_GROUPS, dim=1)
    # Score: QK^T / √d_k
    raw_pre = (torch.matmul(Q_pre, K_pre_exp.transpose(-2, -1))
               / T_STRUCT)   # (B, 24, S, S)
    # Apply causal mask in LOGIT space (-inf upper triangle)
    causal = torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1)
    raw_pre_masked = raw_pre.masked_fill(causal, float('-inf'))
    # Softmax to get reconstructed weights
    recon_pre = torch.softmax(raw_pre_masked, dim=-1).cpu().numpy()

    # Compare to actual HF weights (per Q head)
    hf_w_np = hf_w.cpu().numpy()
    diff_pre = np.abs(recon_pre - hf_w_np)
    # Only compare where the mask doesn't zero things out
    lower_tri = ~causal.cpu().numpy()
    diff_pre_lower = diff_pre[..., lower_tri]
    max_diff_pre = float(diff_pre_lower.max())
    mean_diff_pre = float(diff_pre_lower.mean())

    print(f"         max abs diff:  {max_diff_pre:.6f}")
    print(f"         mean abs diff: {mean_diff_pre:.6f}")
    print(f"         tolerance for invariant: {INVARIANT_TOL}")
    print(f"         regression min diff (must exceed): {REGRESSION_MIN_DIFF}")

    test1_failed_as_expected = max_diff_pre > REGRESSION_MIN_DIFF
    if test1_failed_as_expected:
        print(f"         RESULT: \033[31mFAIL\033[0m (as expected — RoPE bug "
              f"confirmed)\n")
    else:
        print(f"         RESULT: \033[33mPASS\033[0m (UNEXPECTED — pre-RoPE "
              f"matches?!)\n")
        print("         If this test passes, we have misdiagnosed the bug.")
        print("         STOP and re-examine before refactoring.\n")

    # ── TEST 2: post-RoPE reconstruction (the correct path) ─────────────────
    # To get post-RoPE Q and K, we need to actually run RoPE ourselves.
    # We re-do the projection -> reshape -> rotate pipeline that LlamaAttention
    # does internally between line 236 (q_proj) and line 252 (matmul).
    print("[TEST 2] Reconstructing scores from POST-RoPE Q, K (the fix)\n")

    # Import the rotary helper from transformers
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb)
    except ImportError as e:
        print(f"FATAL: cannot import apply_rotary_pos_emb: {e}")
        sys.exit(3)

    # Get position embeddings for our sequence
    # In recent transformers versions, the rotary embedding is computed
    # at the model level via model.model.rotary_emb and passed down.
    # We need to call it on a dummy hidden_states tensor to get (cos, sin).
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(S, device=device).unsqueeze(0)
    # rotary_emb takes (x, position_ids) where x just provides dtype/device
    dummy_x = torch.zeros(B, S, HEAD_DIM, device=device, dtype=torch.float32)
    cos, sin = rotary_emb(dummy_x, position_ids)
    # cos/sin shape: (B, S, HEAD_DIM) — we need to apply per head

    # Move Q and K to device for RoPE math
    Q_dev = Q_pre.to(device).to(torch.float32)
    K_dev = K_pre.to(device).to(torch.float32)

    # apply_rotary_pos_emb signature in recent transformers:
    #   apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
    # The unsqueeze_dim=1 broadcasts cos/sin across the head dimension.
    Q_post, K_post = apply_rotary_pos_emb(Q_dev, K_dev, cos, sin,
                                          unsqueeze_dim=1)

    # GQA expand and matmul
    K_post_exp = K_post.repeat_interleave(KV_GROUPS, dim=1)
    raw_post = (torch.matmul(Q_post, K_post_exp.transpose(-2, -1))
                / T_STRUCT)
    raw_post_masked = raw_post.masked_fill(causal.to(device), float('-inf'))
    recon_post = torch.softmax(raw_post_masked, dim=-1).cpu().numpy()

    diff_post = np.abs(recon_post - hf_w_np)
    diff_post_lower = diff_post[..., lower_tri]
    max_diff_post = float(diff_post_lower.max())
    mean_diff_post = float(diff_post_lower.mean())

    print(f"         max abs diff:  {max_diff_post:.6f}")
    print(f"         mean abs diff: {mean_diff_post:.6f}")
    print(f"         tolerance for invariant: {INVARIANT_TOL}")

    test2_passed = max_diff_post < INVARIANT_TOL
    if test2_passed:
        print(f"         RESULT: \033[32mPASS\033[0m (post-RoPE reconstruction "
              f"matches HF within {INVARIANT_TOL})\n")
    else:
        print(f"         RESULT: \033[31mFAIL\033[0m (even with RoPE applied, "
              f"reconstruction doesn't match)\n")
        print("         This is unexpected. Possible causes:")
        print("         - RoPE math differs from what we replicated")
        print("         - Causal mask convention differs")
        print("         - bf16 vs fp32 precision drift larger than expected")
        print("         Tighten the test before proceeding to refactor.\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 (pre-RoPE):   "
          f"{'FAIL ✓ (regression confirmed)' if test1_failed_as_expected else 'PASS ✗ (unexpected)'}")
    print(f"  TEST 2 (post-RoPE):  "
          f"{'PASS ✓ (fix works)' if test2_passed else 'FAIL ✗ (fix needs work)'}")
    print()

    if test1_failed_as_expected and test2_passed:
        print("  \033[32m✓ Anchor established.\033[0m")
        print("  The RoPE bug is real and the fix is sound.")
        print("  Refactor Stage 1 can proceed.")
        sys.exit(0)
    elif not test1_failed_as_expected:
        print("  \033[33m✗ Regression test passed — we did NOT confirm the bug.\033[0m")
        print("  Stop and re-examine. Do not refactor yet.")
        sys.exit(1)
    else:
        print("  \033[33m✗ Fix did not pass invariant — the post-RoPE path "
              "still differs.\033[0m")
        print("  Investigate before committing to capture redesign.")
        sys.exit(1)


if __name__ == '__main__':
    main()
