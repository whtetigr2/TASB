"""
test_rope_diagnostic.py — Localize the residual post-RoPE diff
==============================================================================
TEST 2 in test_rope_regression.py came back at max diff 0.0042, mean 0.000335.
That's three orders of magnitude better than the broken path but above our
1e-3 invariant tolerance. Don't relax the tolerance — find the source.

Three hypotheses for the residual:
  H1: bf16 compute → fp32 reconstruction precision drift
  H2: RoPE replication has a small bug (wrong unsqueeze_dim, wrong cos/sin
      shape, wrong frequency basis, etc.)
  H3: LlamaAttention does something between RoPE and matmul that we missed
      (scaling factor, dtype cast order, attention scale != 1/√d_k, etc.)

If H1: the worst diffs concentrate on near-zero-probability entries and the
       error grows with bf16/fp16 contamination.
If H2: errors are roughly uniform across all elements and rows.
If H3: errors are systematic — either constant offset or a multiplicative
       factor that's recoverable by inspecting LlamaAttention source.

This script generates evidence for which hypothesis is true.
==============================================================================
"""

import sys, math
import numpy as np


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    MODEL_ID = "meta-llama/Llama-3.2-3B"
    TEST_LAYER = 18
    PROMPT = "The capital of France is"

    HEAD_DIM = 128
    NUM_Q_HEADS = 24
    NUM_KV_HEADS = 8
    KV_GROUPS = NUM_Q_HEADS // NUM_KV_HEADS

    print("=" * 78)
    print("  ROPE DIAGNOSTIC — localizing the residual diff")
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
    print(f"[setup] Ready on {device}\n", flush=True)

    # ── Capture EVERYTHING relevant ─────────────────────────────────────────
    captured = {
        'q_proj_out': None, 'k_proj_out': None,
        'hf_weights': None, 'hf_weights_raw_dtype': None,
        'attn_input_hidden': None,
    }
    attn_module = model.model.layers[TEST_LAYER].self_attn

    def hook_attn_input(module, args, kwargs):
        hidden = args[0] if args else kwargs.get('hidden_states')
        if hidden is not None:
            captured['attn_input_hidden'] = hidden.detach().clone()
    h_input = attn_module.register_forward_pre_hook(
        hook_attn_input, with_kwargs=True)

    def hook_q(module, args, output):
        captured['q_proj_out'] = output.detach().clone()
    h_q = attn_module.q_proj.register_forward_hook(hook_q)

    def hook_k(module, args, output):
        captured['k_proj_out'] = output.detach().clone()
    h_k = attn_module.k_proj.register_forward_hook(hook_k)

    def hook_attn(module, args, kwargs, output):
        if len(output) > 1 and output[1] is not None:
            w = output[1].detach()
            captured['hf_weights'] = w.float().cpu()
            captured['hf_weights_raw_dtype'] = w.dtype
    h_attn = attn_module.register_forward_hook(hook_attn, with_kwargs=True)

    inputs = tok(PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        _ = model(**inputs, output_attentions=True, use_cache=False)
    h_input.remove(); h_q.remove(); h_k.remove(); h_attn.remove()

    B, S, _ = captured['q_proj_out'].shape
    print(f"[capture] S={S}, weights dtype as captured: {captured['hf_weights_raw_dtype']}")
    print(f"[capture] q_proj dtype: {captured['q_proj_out'].dtype}")

    # ── Reconstruct post-RoPE scores in FULL fp32 from the start ───────────
    Q = captured['q_proj_out'].view(B, S, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    K = captured['k_proj_out'].view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)

    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(S, device=device).unsqueeze(0)
    dummy = torch.zeros(B, S, HEAD_DIM, device=device, dtype=torch.float32)
    cos, sin = rotary_emb(dummy, position_ids)

    Q_post, K_post = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=1)
    K_post_exp = K_post.repeat_interleave(KV_GROUPS, dim=1)
    T_struct = math.sqrt(HEAD_DIM)
    scores = torch.matmul(Q_post, K_post_exp.transpose(-2, -1)) / T_struct

    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)
    scores_masked = scores.masked_fill(causal, float('-inf'))
    recon = torch.softmax(scores_masked, dim=-1).cpu().numpy()

    hf_w = captured['hf_weights'].numpy()
    diff = np.abs(recon - hf_w)

    lower_tri = ~causal.cpu().numpy()
    diff_lower = diff[..., lower_tri]

    print(f"\n[overall]  max abs diff: {diff_lower.max():.6f}")
    print(f"[overall]  mean abs diff: {diff_lower.mean():.6f}")
    print(f"[overall]  median diff:   {np.median(diff_lower):.6f}")
    print(f"[overall]  99th pct:      {np.percentile(diff_lower, 99):.6f}")
    print(f"[overall]  99.9th pct:    {np.percentile(diff_lower, 99.9):.6f}")

    # ── H1 evidence: do worst diffs concentrate where prob is near zero? ───
    print("\n" + "=" * 78)
    print("  H1: Precision drift on near-zero-probability entries")
    print("=" * 78)
    # Bin the differences by the probability magnitude
    p_flat = hf_w[..., lower_tri].flatten()
    d_flat = diff[..., lower_tri].flatten()

    bins = [(0, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, 1e-1), (1e-1, 1.0)]
    print(f"  {'prob range':<20} {'count':>8} {'max diff':>12} {'mean diff':>12}")
    print("  " + "-" * 56)
    for lo, hi in bins:
        mask = (p_flat >= lo) & (p_flat < hi)
        n = int(mask.sum())
        if n > 0:
            d_in_bin = d_flat[mask]
            print(f"  [{lo:.0e}, {hi:.0e}){'':<5} {n:>8} {d_in_bin.max():>12.6f} "
                  f"{d_in_bin.mean():>12.6f}")
        else:
            print(f"  [{lo:.0e}, {hi:.0e}){'':<5} {n:>8} {'--':>12} {'--':>12}")

    # ── H2 evidence: spatial distribution of errors per head/row ──────────
    print("\n" + "=" * 78)
    print("  H2: Spatial pattern of errors")
    print("=" * 78)
    # Per-head max diff
    diff_per_head = diff[..., lower_tri].reshape(NUM_Q_HEADS, -1).max(axis=1)
    diff_per_head_mean = diff[..., lower_tri].reshape(NUM_Q_HEADS, -1).mean(axis=1)
    print(f"  Per-Q-head max diff range: [{diff_per_head.min():.6f}, "
          f"{diff_per_head.max():.6f}]")
    print(f"  Per-Q-head mean diff range: [{diff_per_head_mean.min():.6f}, "
          f"{diff_per_head_mean.max():.6f}]")
    # Show top 3 heads
    top3 = diff_per_head.argsort()[-3:][::-1]
    for h in top3:
        print(f"    head {h:>2}: max={diff_per_head[h]:.6f}, "
              f"mean={diff_per_head_mean[h]:.6f}")

    # ── H3 evidence: check the actual scaling factor LlamaAttention uses ──
    print("\n" + "=" * 78)
    print("  H3: Inspect LlamaAttention internals")
    print("=" * 78)
    print(f"  attn module type: {type(attn_module).__name__}")
    if hasattr(attn_module, 'scaling'):
        print(f"  attn_module.scaling: {attn_module.scaling}  "
              f"(expected: 1/√d_k = {1/T_struct:.6f})")
    if hasattr(attn_module, 'head_dim'):
        print(f"  attn_module.head_dim: {attn_module.head_dim}")
    if hasattr(attn_module, 'num_heads'):
        print(f"  attn_module.num_heads: {attn_module.num_heads}")
    if hasattr(attn_module, 'num_key_value_heads'):
        print(f"  attn_module.num_key_value_heads: {attn_module.num_key_value_heads}")

    # Re-do scoring using the module's actual scaling if available
    if hasattr(attn_module, 'scaling'):
        module_scaling = attn_module.scaling
        scores2 = torch.matmul(
            Q_post, K_post_exp.transpose(-2, -1)) * module_scaling
        scores2_masked = scores2.masked_fill(causal, float('-inf'))
        recon2 = torch.softmax(scores2_masked, dim=-1).cpu().numpy()
        diff2 = np.abs(recon2 - hf_w)
        diff2_lower = diff2[..., lower_tri]
        print(f"\n  Using attn_module.scaling instead of 1/√d_k:")
        print(f"    max abs diff:  {diff2_lower.max():.6f}")
        print(f"    mean abs diff: {diff2_lower.mean():.6f}")

    # ── H1 confirmation: do the matmul in bf16 to mimic the real path ─────
    print("\n" + "=" * 78)
    print("  H1 confirmation: matmul in bf16 to mimic LlamaAttention path")
    print("=" * 78)
    Q_bf = Q_post.to(torch.bfloat16)
    K_bf = K_post_exp.to(torch.bfloat16)
    scores_bf = torch.matmul(Q_bf, K_bf.transpose(-2, -1)) / T_struct
    scores_bf_masked = scores_bf.to(torch.float32).masked_fill(
        causal, float('-inf'))
    recon_bf = torch.softmax(scores_bf_masked, dim=-1).cpu().numpy()
    diff_bf = np.abs(recon_bf - hf_w)
    diff_bf_lower = diff_bf[..., lower_tri]
    print(f"  bf16 matmul → fp32 softmax: max diff {diff_bf_lower.max():.6f}, "
          f"mean {diff_bf_lower.mean():.6f}")

    # And fp16
    Q_f16 = Q_post.to(torch.float16)
    K_f16 = K_post_exp.to(torch.float16)
    scores_f16 = torch.matmul(Q_f16, K_f16.transpose(-2, -1)) / T_struct
    scores_f16_masked = scores_f16.to(torch.float32).masked_fill(
        causal, float('-inf'))
    recon_f16 = torch.softmax(scores_f16_masked, dim=-1).cpu().numpy()
    diff_f16 = np.abs(recon_f16 - hf_w)
    diff_f16_lower = diff_f16[..., lower_tri]
    print(f"  fp16 matmul → fp32 softmax: max diff {diff_f16_lower.max():.6f}, "
          f"mean {diff_f16_lower.mean():.6f}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Pure fp32 path:        max {diff_lower.max():.6f}")
    print(f"  fp16-matmul path:      max {diff_f16_lower.max():.6f}")
    print(f"  bf16-matmul path:      max {diff_bf_lower.max():.6f}")
    if diff_bf_lower.max() < diff_lower.max() * 0.5 or diff_f16_lower.max() < diff_lower.max() * 0.5:
        print(f"\n  H1 likely: matching compute dtype reduces max diff substantially.")
        print(f"  Resolution: invariant tolerance should reflect compute-path dtype,")
        print(f"  not a uniform fp32 bound. Set tolerance based on the dtype of the")
        print(f"  shortest matmul in the chain.")
    elif hasattr(attn_module, 'scaling') and abs(attn_module.scaling - 1/T_struct) > 1e-5:
        print(f"\n  H3 likely: the attention module uses scaling != 1/√d_k.")
    else:
        print(f"\n  Inconclusive — investigate per-head and per-position patterns.")


if __name__ == '__main__':
    main()
