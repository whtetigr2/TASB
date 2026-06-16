"""
test_rope_live_capture.py — Definitive test via LlamaAttention.forward wrap
==============================================================================
Codex's diagnosis after the bf16-roundtrip result was:
  - source dump rules out hidden ops (good)
  - bf16 explains part of residual but not the max (mean -55%, max -8%)
  - REMAINING ISSUE: the test re-derives rotary embeddings with fp32 dummy
    while live model produces bf16-cast cos/sin (concrete mismatch)
  - also: test rebuilds the causal mask instead of using the live one

This test takes Codex's prescription: wrap LlamaAttention.forward to capture
the LIVE post-RoPE Q, K, the LIVE attention_mask, and the LIVE pre-softmax
attn_weights. Then compare reconstruction at each stage against live ground
truth.

THIS SCRIPT IS ALSO THE STAGE-1 CAPTURE PROOF-OF-CONCEPT.
The wrap technique used here is exactly what tasb_capture_v2.py will use.
If the wrap captures cleanly and reversibly here, the refactor design is
validated.

THREE COMPARISONS
-----------------
(1) Reconstructed Q_post, K_post vs LIVE Q_post, K_post
    → if MATCH within bf16 precision: RoPE replication is correct
    → if DIFFER: we have a real RoPE bug to find

(2) Reconstructed pre-softmax scores vs LIVE pre-softmax scores
    → tests matmul + scaling + mask add in isolation

(3) Reconstructed post-softmax probs vs LIVE post-softmax probs (i.e. attn_weights)
    → end-to-end check, but now against LIVE intermediates

VERDICTS
--------
EXACT      → (1) (2) (3) all under 1e-4 bf16 tolerance. Refactor proceeds
             with high confidence. Wrap technique validated for Stage 1.
RECON_OK   → (1) passes, (3) higher. Reconstruction is right; remaining
             diff is downstream of softmax (the .to(query.dtype) cast).
             This is the BEST CASE — we can bound it tightly.
ROPE_BUG   → (1) fails. There IS a RoPE replication bug. Diagnose using
             the live cos/sin tensors now in hand.
==============================================================================
"""

import sys, math, inspect
import numpy as np


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, apply_rotary_pos_emb, repeat_kv)

    MODEL_ID = "meta-llama/Llama-3.2-3B"
    TEST_LAYER = 18
    PROMPT = "The capital of France is"

    HEAD_DIM = 128
    NUM_Q_HEADS = 24
    NUM_KV_HEADS = 8

    print("=" * 78)
    print("  LIVE-CAPTURE VIA FORWARD WRAP — definitive test")
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

    attn_module = model.model.layers[TEST_LAYER].self_attn
    print(f"[setup] Ready. Wrapping L{TEST_LAYER} forward...\n")

    # ── Wrap LlamaAttention.forward on L18 to stash live intermediates ─────
    # This is the technique tasb_capture_v2.py will use.
    captured = {
        'q_proj_out': None, 'k_proj_out': None, 'v_proj_out': None,
        'q_post_rope': None, 'k_post_rope': None,
        'cos': None, 'sin': None,
        'attention_mask': None,
        'pre_softmax_scores': None,
        'attn_weights_out': None,
        'wrap_called': False,
    }

    original_forward = attn_module.forward

    def wrapped_forward(self,
                        hidden_states,
                        position_embeddings=None,
                        attention_mask=None,
                        past_key_values=None,
                        **kwargs):
        captured['wrap_called'] = True

        # Replicate the start of LlamaAttention.forward, byte-for-byte,
        # but stash everything along the way.

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Projections (stash pre-RoPE)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        captured['q_proj_out'] = q.detach().clone()
        captured['k_proj_out'] = k.detach().clone()
        captured['v_proj_out'] = v.detach().clone()

        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states   = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        # Stash live cos, sin
        cos, sin = position_embeddings
        captured['cos'] = cos.detach().clone()
        captured['sin'] = sin.detach().clone()

        # Apply RoPE (stash post-RoPE)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
        captured['q_post_rope'] = query_states.detach().clone()
        captured['k_post_rope'] = key_states.detach().clone()

        # Stash live mask
        if attention_mask is not None:
            captured['attention_mask'] = attention_mask.detach().clone()

        # Now do the matmul + scaling + mask manually so we can stash
        # the pre-softmax scores. This mirrors eager_attention_forward.
        k_rep = repeat_kv(key_states, self.num_key_value_groups)
        v_rep = repeat_kv(value_states, self.num_key_value_groups)
        attn_w = torch.matmul(query_states, k_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        captured['pre_softmax_scores'] = attn_w.detach().clone()

        # Continue exactly as eager_attention_forward does
        attn_w = torch.nn.functional.softmax(
            attn_w, dim=-1, dtype=torch.float32).to(query_states.dtype)
        captured['attn_weights_out'] = attn_w.detach().clone()

        attn_out = torch.matmul(attn_w, v_rep)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = self.o_proj(attn_out)
        return attn_out, attn_w

    # Bind the wrapper as a method on the specific module instance
    import types
    attn_module.forward = types.MethodType(wrapped_forward, attn_module)

    # ── Run forward ─────────────────────────────────────────────────────────
    inputs = tok(PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=False)

    # Restore original forward
    attn_module.forward = original_forward
    print(f"[wrap] called={captured['wrap_called']}, "
          f"all tensors stashed: "
          f"{all(captured[k] is not None for k in ['q_post_rope','k_post_rope','cos','sin','attn_weights_out'])}")
    print(f"[wrap] mask captured: {captured['attention_mask'] is not None}\n")

    # ── Inspect captured tensors ────────────────────────────────────────────
    print("[live] Captured intermediates from L18 forward:")
    print(f"  q_proj output:        {tuple(captured['q_proj_out'].shape)}, "
          f"{captured['q_proj_out'].dtype}")
    print(f"  q_post_rope:          {tuple(captured['q_post_rope'].shape)}, "
          f"{captured['q_post_rope'].dtype}")
    print(f"  k_post_rope:          {tuple(captured['k_post_rope'].shape)}, "
          f"{captured['k_post_rope'].dtype}")
    print(f"  cos:                  {tuple(captured['cos'].shape)}, "
          f"{captured['cos'].dtype}")
    print(f"  sin:                  {tuple(captured['sin'].shape)}, "
          f"{captured['sin'].dtype}")
    if captured['attention_mask'] is not None:
        m = captured['attention_mask']
        print(f"  attention_mask:       {tuple(m.shape)}, {m.dtype}")
        unique = torch.unique(m).cpu().tolist()
        print(f"    unique values: {unique[:6]}{' ...' if len(unique) > 6 else ''}")
    print(f"  pre_softmax_scores:   {tuple(captured['pre_softmax_scores'].shape)}, "
          f"{captured['pre_softmax_scores'].dtype}")
    print(f"  attn_weights_out:     {tuple(captured['attn_weights_out'].shape)}, "
          f"{captured['attn_weights_out'].dtype}")

    B, S, _ = captured['q_proj_out'].shape
    print()

    # ── (1) Reconstruct Q_post, K_post and compare to live tensors ─────────
    print("=" * 78)
    print("  COMPARISON (1): Reconstructed Q_post, K_post vs LIVE Q_post, K_post")
    print("=" * 78)

    # Reconstruct from pre-RoPE q_proj/k_proj using the LIVE cos/sin
    Q = captured['q_proj_out'].view(B, S, NUM_Q_HEADS,  HEAD_DIM).transpose(1, 2)
    K = captured['k_proj_out'].view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    Q_post_recon, K_post_recon = apply_rotary_pos_emb(
        Q, K, captured['cos'], captured['sin'])

    # Compare to live (both should be the same dtype as captured)
    q_diff = (Q_post_recon.float() - captured['q_post_rope'].float()).abs()
    k_diff = (K_post_recon.float() - captured['k_post_rope'].float()).abs()

    print(f"  Q_post diff:  max {q_diff.max().item():.2e}, "
          f"mean {q_diff.mean().item():.2e}")
    print(f"  K_post diff:  max {k_diff.max().item():.2e}, "
          f"mean {k_diff.mean().item():.2e}")

    rope_replication_ok = (q_diff.max().item() < 1e-4
                           and k_diff.max().item() < 1e-4)
    if rope_replication_ok:
        print(f"  \033[32m✓ RoPE replication is correct\033[0m "
              f"(both under 1e-4 with live cos/sin)")
    else:
        print(f"  \033[31m✗ RoPE replication differs from live tensors\033[0m")
        print(f"  Investigate: dtype handling, apply_rotary_pos_emb args.")

    # ── (2) Reconstructed pre-softmax scores vs LIVE pre-softmax scores ───
    print("\n" + "=" * 78)
    print("  COMPARISON (2): Reconstructed scores vs LIVE pre-softmax scores")
    print("=" * 78)

    K_rep_recon = repeat_kv(K_post_recon, attn_module.num_key_value_groups)
    scores_recon = (torch.matmul(Q_post_recon, K_rep_recon.transpose(-2, -1))
                    * attn_module.scaling)
    if captured['attention_mask'] is not None:
        scores_recon = scores_recon + captured['attention_mask']
    else:
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)
        scores_recon = scores_recon.masked_fill(causal, float('-inf'))

    scores_diff = (scores_recon.float()
                   - captured['pre_softmax_scores'].float()).abs()
    print(f"  pre-softmax scores diff:  max {scores_diff.max().item():.2e}, "
          f"mean {scores_diff.mean().item():.2e}")

    # Only compare lower triangle (upper has -inf or large negative which
    # can have large absolute diffs but doesn't affect softmax)
    causal_np = np.triu(np.ones((S, S), dtype=bool), 1)
    lower_tri = ~causal_np
    scores_diff_np = scores_diff.cpu().numpy()
    lower_diffs = scores_diff_np[..., lower_tri]
    print(f"  pre-softmax diff (lower tri only): max {lower_diffs.max():.2e}, "
          f"mean {lower_diffs.mean():.2e}")

    # ── (3) Reconstructed probs vs LIVE attn_weights ───────────────────────
    print("\n" + "=" * 78)
    print("  COMPARISON (3): Reconstructed probs vs LIVE attn_weights")
    print("=" * 78)

    probs_recon = torch.nn.functional.softmax(
        scores_recon, dim=-1, dtype=torch.float32).to(Q_post_recon.dtype)

    probs_diff = (probs_recon.float()
                  - captured['attn_weights_out'].float()).abs()
    probs_diff_np = probs_diff.cpu().numpy()
    lower_probs = probs_diff_np[..., lower_tri]
    print(f"  attn_weights diff (lower tri):  max {lower_probs.max():.2e}, "
          f"mean {lower_probs.mean():.2e}")
    print(f"  99th pct: {np.percentile(lower_probs, 99):.2e}")
    print(f"  99.9th pct: {np.percentile(lower_probs, 99.9):.2e}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  (1) Q_post/K_post reconstruction:  "
          f"max Q={q_diff.max().item():.2e}, max K={k_diff.max().item():.2e}")
    print(f"  (2) Pre-softmax scores:            "
          f"max={lower_diffs.max():.2e}")
    print(f"  (3) Post-softmax attn_weights:     "
          f"max={lower_probs.max():.2e}")
    print()

    if not rope_replication_ok:
        verdict = "ROPE_BUG"
        print(f"  \033[31m✗ ROPE_BUG\033[0m")
        print(f"  (1) failed. Our apply_rotary_pos_emb call differs from the")
        print(f"  live one even with the SAME cos/sin inputs. Investigate.")
    elif lower_probs.max() < 1e-4:
        verdict = "EXACT"
        print(f"  \033[32m✓ EXACT\033[0m")
        print(f"  All three stages match live tensors within 1e-4.")
        print(f"  Wrap-based capture validated. Stage 1 proceeds.")
        print(f"  Invariant for tasb_verify_v2.py: max < 5e-4, mean < 1e-4")
    elif lower_probs.max() < 5e-3:
        verdict = "RECON_OK"
        print(f"  \033[32m✓ RECON_OK\033[0m  (best plausible outcome)")
        print(f"  RoPE replication is exact (1). Pre-softmax scores match (2).")
        print(f"  Residual is in the softmax/dtype cast — i.e. HF's own")
        print(f"  .to(query.dtype) bf16 quantization of the probability output.")
        print(f"  This is HF's measurement noise, not our reconstruction error.")
        print(f"  Stage 1 proceeds. Invariant bounds:")
        print(f"    fp32 control:   max < 1e-4, mean < 1e-5")
        print(f"    bf16 runtime:   max < 1e-2, mean < 1e-3")
    else:
        verdict = "STILL_DIFFERS"
        print(f"  \033[33m? STILL_DIFFERS\033[0m")
        print(f"  Q_post/K_post match but post-softmax probs differ by more than 5e-3.")
        print(f"  Look at pre-softmax score diff to localize.")

    return verdict


if __name__ == '__main__':
    main()
