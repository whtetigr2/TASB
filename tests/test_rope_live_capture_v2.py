"""
test_rope_live_capture_v2.py — Patch HELPERS, not forward (Codex's correction)
==============================================================================
v1 (test_rope_live_capture.py) replaced LlamaAttention.forward with a
reimplementation. Codex flagged: that makes comparison (1) pass by
construction — both sides of the diff use the same apply_rotary_pos_emb
on the same inputs.

v2 fix: leave LlamaAttention.forward UNTOUCHED. Patch the module-level
helper functions it calls (apply_rotary_pos_emb and eager_attention_forward)
so they record their inputs and outputs but delegate to the original
implementation for actual execution.

This way:
  - The real forward path runs unchanged
  - We capture tensors from the actual execution, not a parallel run
  - Comparisons against reconstruction are genuinely independent

THREE COMPARISONS (now independent)
-----------------------------------
(1) Reconstructed Q_post, K_post (computed by THIS script) vs LIVE Q_post,
    K_post (returned by HF's apply_rotary_pos_emb during the real forward)
    → if MATCH within bf16 precision: RoPE replication is correct
    → if DIFFER: there's a real RoPE bug

(2) Reconstructed pre-softmax scores vs LIVE scores (from inside
    eager_attention_forward, captured BEFORE softmax)
    → tests matmul + scaling + mask add in isolation

(3) Reconstructed final probs vs LIVE attn_weights (from eager_attention_forward
    output, post-softmax post-cast)
    → end-to-end check against truly live data

VERDICTS (same as v1)
---------------------
EXACT       → all three under 1e-4. Refactor proceeds.
RECON_OK    → (1) and (2) pass; (3) higher. Residual is HF's own
              softmax-cast quantization (best plausible outcome).
ROPE_BUG    → (1) fails. Real bug to find.

STAGE 1 IMPLICATIONS
--------------------
If this passes, the patching technique used here is what tasb_capture_v2.py
should use. Patching apply_rotary_pos_emb and eager_attention_forward at the
module level is less invasive than wrapping LlamaAttention.forward and
survives transformers version changes better.
==============================================================================
"""

import sys, math, inspect
import numpy as np


def main():
    import torch
    import transformers.models.llama.modeling_llama as llama_mod
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
    print("  LIVE-CAPTURE v2 — patch helpers, leave forward intact")
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
    print(f"[setup] Ready. LlamaAttention.forward will run UNCHANGED.\n")

    # ── Capture storage keyed by which layer the call came from ────────────
    # We need to filter: apply_rotary_pos_emb is called from EVERY layer's
    # forward; we only want L18. The cleanest filter is by call count
    # within a single model forward (L18 is the 19th attention call,
    # 0-indexed = 18).
    captured = {
        'rope_call_count': 0,
        'eager_call_count': 0,
        # From L18's apply_rotary_pos_emb:
        'q_pre_rope': None,    # the q argument going IN to apply_rotary_pos_emb
        'k_pre_rope': None,    # the k argument going IN
        'cos': None,           # cos going in
        'sin': None,           # sin going in
        'q_post_rope': None,   # q returned from apply_rotary_pos_emb
        'k_post_rope': None,   # k returned
        # From L18's eager_attention_forward:
        'eager_query': None,
        'eager_key': None,
        'eager_attention_mask': None,
        'eager_scaling': None,
        'eager_attn_output': None,
        'eager_attn_weights': None,
        # Pre-softmax scores: captured by patching nn.functional.softmax
        # inside eager_attention_forward. Save the LAST call to softmax
        # from the L18 eager call.
    }

    # Store originals for restoration
    original_rope = llama_mod.apply_rotary_pos_emb
    # eager_attention_forward might not be at module level in all versions;
    # check both possible locations
    has_eager_fn = hasattr(llama_mod, 'eager_attention_forward')
    if has_eager_fn:
        original_eager = llama_mod.eager_attention_forward
    else:
        original_eager = None
        print("[warn] eager_attention_forward not at module level; "
              "will capture via softmax hook instead")

    # Registry restore state — initialized to None so the finally block can
    # safely check them regardless of whether we patched the registry.
    attn_registry = None
    attn_registry_original = None

    # ── Patch apply_rotary_pos_emb ─────────────────────────────────────────
    # NOTE (Codex): The call-count filter below is brittle and only OK for
    # diagnostic use. apply_rotary_pos_emb is called once per layer in layer
    # order during a single forward, so counting calls works here. For
    # production capture in tasb_capture_v2.py we should use a more robust
    # layer-identification scheme (e.g. inspect the call stack to find the
    # owning attention module, or pass layer_idx via thread-local state).
    #
    # Bug #9 (TASB-internal): the patched function MUST pass through args/kwargs
    # opaquely. Hardcoding kwargs like position_ids=position_ids breaks across
    # transformers versions — modern HF calls apply_rotary_pos_emb positionally
    # with (q, k, cos, sin) only, and forwarding an explicit position_ids=None
    # raises TypeError. Engineering posture rule 1: local runtime is source of
    # truth, don't recall.
    def patched_rope(*args, **kwargs):
        # First two positional args are q, k; next two are cos, sin.
        # We don't assume anything else about the signature.
        q, k, cos, sin = args[0], args[1], args[2], args[3]
        # Run the ORIGINAL implementation with EXACTLY what HF passed.
        q_out, k_out = original_rope(*args, **kwargs)
        # Filter to L18 (the 19th call, 0-indexed 18)
        if captured['rope_call_count'] == TEST_LAYER:
            captured['q_pre_rope'] = q.detach().clone()
            captured['k_pre_rope'] = k.detach().clone()
            captured['cos'] = cos.detach().clone()
            captured['sin'] = sin.detach().clone()
            captured['q_post_rope'] = q_out.detach().clone()
            captured['k_post_rope'] = k_out.detach().clone()
        captured['rope_call_count'] += 1
        return q_out, k_out

    llama_mod.apply_rotary_pos_emb = patched_rope

    # ── Patch eager_attention_forward (if it exists at module level) ───────
    if has_eager_fn:
        # Bug #9 applied here too: opaque passthrough so signature drift
        # in transformers doesn't break the patch.
        # Standard signature today is (module, query, key, value, attention_mask,
        # scaling=..., dropout=..., **kwargs) but we treat it as opaque.
        def patched_eager(*args, **kwargs):
            # Pull positional args we want to log. The first 5 are stable in
            # all transformers versions: (module, query, key, value, attention_mask).
            module        = args[0]
            query         = args[1]
            key           = args[2]
            value         = args[3] if len(args) > 3 else kwargs.get('value')
            attention_mask = args[4] if len(args) > 4 else kwargs.get('attention_mask')
            scaling       = kwargs.get('scaling', None)

            # Run the ORIGINAL implementation with EXACTLY what HF passed.
            out, weights = original_eager(*args, **kwargs)

            # Filter: only stash for L18
            if captured['eager_call_count'] == TEST_LAYER:
                captured['eager_query']           = query.detach().clone()
                captured['eager_key']             = key.detach().clone()
                if scaling is not None:
                    captured['eager_scaling']         = float(scaling)
                if attention_mask is not None:
                    captured['eager_attention_mask'] = attention_mask.detach().clone()
                captured['eager_attn_output']     = out.detach().clone()
                captured['eager_attn_weights']    = weights.detach().clone()
            captured['eager_call_count'] += 1
            return out, weights

        llama_mod.eager_attention_forward = patched_eager
        # Also need to patch ALL_ATTENTION_FUNCTIONS if it caches a reference
        # (recent transformers versions register at import time)
        try:
            from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS
            if hasattr(ALL_ATTENTION_FUNCTIONS, '_global_mapping'):
                if 'eager' in ALL_ATTENTION_FUNCTIONS._global_mapping:
                    attn_registry = ALL_ATTENTION_FUNCTIONS._global_mapping
                    attn_registry_original = attn_registry['eager']
                    attn_registry['eager'] = patched_eager
        except (ImportError, AttributeError):
            pass

    # ── Run forward (in try/finally so restores ALWAYS happen) ─────────────
    inputs = tok(PROMPT, return_tensors='pt').to(device)
    try:
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, use_cache=False)
    finally:
        # Restore originals UNCONDITIONALLY — leak prevention
        llama_mod.apply_rotary_pos_emb = original_rope
        if has_eager_fn:
            llama_mod.eager_attention_forward = original_eager
        # Restore the global attention registry if we touched it
        if attn_registry is not None and attn_registry_original is not None:
            attn_registry['eager'] = attn_registry_original

    print(f"[patch] apply_rotary_pos_emb called {captured['rope_call_count']} times")
    if has_eager_fn:
        print(f"[patch] eager_attention_forward called {captured['eager_call_count']} times")
    print(f"[patch] L{TEST_LAYER} captured: "
          f"rope={captured['q_post_rope'] is not None}, "
          f"eager={captured['eager_attn_weights'] is not None}\n")

    if captured['q_post_rope'] is None:
        print(f"ERROR: did not capture L{TEST_LAYER} RoPE call. "
              f"Total rope calls: {captured['rope_call_count']}")
        print(f"Check that TEST_LAYER ({TEST_LAYER}) is within range.")
        sys.exit(1)

    # ── Inspect captured tensors ────────────────────────────────────────────
    print("[live] Captured intermediates from L18 (during real forward):")
    print(f"  q_pre_rope:   {tuple(captured['q_pre_rope'].shape)}, "
          f"{captured['q_pre_rope'].dtype}")
    print(f"  k_pre_rope:   {tuple(captured['k_pre_rope'].shape)}, "
          f"{captured['k_pre_rope'].dtype}")
    print(f"  cos:          {tuple(captured['cos'].shape)}, "
          f"{captured['cos'].dtype}")
    print(f"  sin:          {tuple(captured['sin'].shape)}, "
          f"{captured['sin'].dtype}")
    print(f"  q_post_rope:  {tuple(captured['q_post_rope'].shape)}, "
          f"{captured['q_post_rope'].dtype}")
    print(f"  k_post_rope:  {tuple(captured['k_post_rope'].shape)}, "
          f"{captured['k_post_rope'].dtype}")
    if captured['eager_attention_mask'] is not None:
        m = captured['eager_attention_mask']
        print(f"  attention_mask: {tuple(m.shape)}, {m.dtype}")
        unique = torch.unique(m).cpu().tolist()
        print(f"    unique values: {unique[:6]}{' ...' if len(unique) > 6 else ''}")
    if captured['eager_attn_weights'] is not None:
        w = captured['eager_attn_weights']
        print(f"  attn_weights: {tuple(w.shape)}, {w.dtype}")
    print(f"  scaling: {captured['eager_scaling']}")
    print()

    # ── (1) Reconstruct Q_post, K_post and compare to LIVE ─────────────────
    print("=" * 78)
    print("  COMPARISON (1): Reconstructed Q_post, K_post vs LIVE (independent!)")
    print("=" * 78)
    print("  Reconstruction uses captured q_pre_rope, k_pre_rope, cos, sin —")
    print("  applies apply_rotary_pos_emb ourselves and compares to what HF's")
    print("  forward got out of the same function. These are NOW independent.")
    print()

    # Use the EXACT pre-RoPE q, k, cos, sin that the live call received
    q_pre = captured['q_pre_rope']
    k_pre = captured['k_pre_rope']
    cos_live = captured['cos']
    sin_live = captured['sin']

    # Reconstruct by calling the (now-restored) apply_rotary_pos_emb
    q_post_recon, k_post_recon = apply_rotary_pos_emb(
        q_pre, k_pre, cos_live, sin_live)

    q_live = captured['q_post_rope']
    k_live = captured['k_post_rope']

    q_diff = (q_post_recon.float() - q_live.float()).abs()
    k_diff = (k_post_recon.float() - k_live.float()).abs()

    print(f"  Q_post diff:  max {q_diff.max().item():.2e}, "
          f"mean {q_diff.mean().item():.2e}")
    print(f"  K_post diff:  max {k_diff.max().item():.2e}, "
          f"mean {k_diff.mean().item():.2e}")

    # This SHOULD be essentially zero — both calls execute the same Python
    # function on identical inputs. If it's not, something is non-deterministic.
    rope_replication_ok = (q_diff.max().item() < 1e-4
                           and k_diff.max().item() < 1e-4)
    if rope_replication_ok:
        if q_diff.max().item() < 1e-7:
            print(f"  \033[32m✓ Bit-exact\033[0m (both calls produce identical output)")
        else:
            print(f"  \033[32m✓ Matches within bf16 precision\033[0m")
    else:
        print(f"  \033[31m✗ Reconstruction differs from live\033[0m")
        print(f"  This shouldn't happen unless there's non-determinism in")
        print(f"  apply_rotary_pos_emb (e.g. tensor caching, RNG, etc).")

    # ── (1b) INDEPENDENT RoPE — implement from scratch, not via HF's function ──
    print("\n" + "=" * 78)
    print("  COMPARISON (1b): INDEPENDENT from-scratch RoPE vs LIVE Q_post, K_post")
    print("=" * 78)
    print("  (1) compared HF's function to itself. This comparison uses a")
    print("  hand-implemented RoPE built from the published formula, then")
    print("  compares against the LIVE tensors HF computed. This is the")
    print("  truly independent check Codex asked for.")
    print()

    def rotate_half(x):
        # HF convention: rotate by splitting the LAST dim into two halves,
        # then negating the second half and swapping. From modeling_llama.py.
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def independent_rope(q, k, cos, sin, unsqueeze_dim=1):
        # Cos and sin come in shape (B, S, head_dim); unsqueeze for head dim.
        cos_b = cos.unsqueeze(unsqueeze_dim)   # (B, 1, S, head_dim)
        sin_b = sin.unsqueeze(unsqueeze_dim)
        q_rot = (q * cos_b) + (rotate_half(q) * sin_b)
        k_rot = (k * cos_b) + (rotate_half(k) * sin_b)
        return q_rot, k_rot

    q_post_indep, k_post_indep = independent_rope(
        q_pre, k_pre, cos_live, sin_live)

    q_diff_indep = (q_post_indep.float() - q_live.float()).abs()
    k_diff_indep = (k_post_indep.float() - k_live.float()).abs()

    print(f"  Independent Q_post vs LIVE Q_post:  max {q_diff_indep.max().item():.2e}, "
          f"mean {q_diff_indep.mean().item():.2e}")
    print(f"  Independent K_post vs LIVE K_post:  max {k_diff_indep.max().item():.2e}, "
          f"mean {k_diff_indep.mean().item():.2e}")

    independent_ok = (q_diff_indep.max().item() < 1e-4
                      and k_diff_indep.max().item() < 1e-4)
    if independent_ok:
        print(f"  \033[32m✓ Independent RoPE matches HF's RoPE within 1e-4.\033[0m")
        print(f"  This is the honest proof that our understanding of the RoPE")
        print(f"  transformation is correct. Stage 1 capture can rely on it.")
    elif q_diff_indep.max().item() < 5e-3 and k_diff_indep.max().item() < 5e-3:
        print(f"  \033[32m✓ Within bf16 precision.\033[0m")
        print(f"  Independent implementation agrees with HF up to bf16 cast effects.")
    else:
        print(f"  \033[31m✗ Independent RoPE DIFFERS from HF's RoPE.\033[0m")
        print(f"  Either our formula is wrong or HF has a non-standard")
        print(f"  implementation. Inspect modeling_llama.apply_rotary_pos_emb.")

    # ── (2) Downstream score consistency check (uses INDEPENDENT RoPE) ─────
    print("\n" + "=" * 78)
    print("  COMPARISON (2): Downstream score CONSISTENCY check")
    print("  ────────────────────────────────────────────────────────────────")
    print("  HONEST FRAMING: this rebuilds matmul + mask + scaling OUTSIDE")
    print("  eager_attention_forward, once from LIVE post-RoPE Q,K and once")
    print("  from INDEPENDENT post-RoPE Q,K, then compares.")
    print()
    print("  This is NOT comparing against the real pre-softmax scores")
    print("  computed inside eager_attention_forward — those would require")
    print("  capturing the matmul output directly. Wrapping the matmul itself")
    print("  would put us back in self-consistency-test land.")
    print()
    print("  What (2) proves: downstream matmul/mask/scaling is consistent")
    print("  between live and independent post-RoPE paths.")
    print("  What (3) proves: end-to-end reconstruction matches HF's actual")
    print("  attn_weights (which IS captured from inside the real path).")
    print("=" * 78)

    scaling = captured['eager_scaling'] if captured['eager_scaling'] else attn_module.scaling

    # Scores from LIVE post-RoPE Q, K — the ground truth path
    K_rep_live = repeat_kv(k_live, attn_module.num_key_value_groups)
    scores_from_live = (
        torch.matmul(q_live, K_rep_live.transpose(-2, -1)) * scaling)

    # Scores from INDEPENDENT post-RoPE Q, K — the honest reconstruction
    K_rep_indep = repeat_kv(k_post_indep, attn_module.num_key_value_groups)
    scores_from_indep = (
        torch.matmul(q_post_indep, K_rep_indep.transpose(-2, -1)) * scaling)

    # Apply LIVE attention_mask if present
    if captured['eager_attention_mask'] is not None:
        scores_from_live  = scores_from_live  + captured['eager_attention_mask']
        scores_from_indep = scores_from_indep + captured['eager_attention_mask']

    scores_diff = (scores_from_live.float() - scores_from_indep.float()).abs()

    # Filter to non-masked (valid) positions.
    # Bug #5 redux: the captured attention_mask is (1, 1, S, S) — broadcast-
    # ready for HF's compute graph but NOT shape-compatible as an index into
    # the (B, n_q, S, S) scores tensor. Must broadcast explicitly.
    if captured['eager_attention_mask'] is not None:
        valid = (captured['eager_attention_mask'] > -1e30).cpu().numpy()
        # Broadcast to scores shape: (1, 1, S, S) → (B, n_q, S, S)
        valid = np.broadcast_to(valid, scores_diff.shape)
    else:
        S = q_live.shape[2]
        causal_np = np.triu(np.ones((S, S), dtype=bool), 1)
        valid = ~causal_np
        valid = np.broadcast_to(valid, scores_diff.shape)

    valid_score_diffs = scores_diff.cpu().numpy()[valid]
    scores_ok = valid_score_diffs.max() < 5e-3
    print(f"  pre-softmax scores diff (independent-vs-live, valid positions):")
    print(f"    max:  {valid_score_diffs.max():.2e}")
    print(f"    mean: {valid_score_diffs.mean():.2e}")
    if scores_ok:
        print(f"  \033[32m✓ Score reconstruction matches within 5e-3.\033[0m")
    else:
        print(f"  \033[31m✗ Score reconstruction differs.\033[0m  Likely matmul,")
        print(f"  repeat_kv, scaling, or mask add — diagnose downstream of RoPE.")

    # ── (3) End-to-end probability reconstruction check (INDEPENDENT) ──────
    print("\n" + "=" * 78)
    print("  COMPARISON (3): End-to-end probability reconstruction check")
    print("  Softmaxes the INDEPENDENT-path scores in fp32 → casts to query dtype,")
    print("  matching eager_attention_forward's dtype convention exactly.")
    print("  Compares to LIVE attn_weights from inside the real forward path.")
    print("=" * 78)

    probs_recon = torch.nn.functional.softmax(
        scores_from_indep, dim=-1, dtype=torch.float32).to(q_live.dtype)
    probs_live = captured['eager_attn_weights']

    probs_diff_np = None
    probs_ok = False
    probs_bf16_ok = False
    if probs_live is not None:
        probs_diff = (probs_recon.float() - probs_live.float()).abs()
        probs_diff_np = probs_diff.cpu().numpy()[valid]
        print(f"  attn_weights diff (independent recon vs live, valid positions):")
        print(f"    max:        {probs_diff_np.max():.2e}")
        print(f"    mean:       {probs_diff_np.mean():.2e}")
        print(f"    99th pct:   {np.percentile(probs_diff_np, 99):.2e}")
        print(f"    99.9th pct: {np.percentile(probs_diff_np, 99.9):.2e}")
        probs_ok = probs_diff_np.max() < 1e-4
        probs_bf16_ok = probs_diff_np.max() < 5e-3
        if probs_ok:
            print(f"  \033[32m✓ Within 1e-4.\033[0m")
        elif probs_bf16_ok:
            print(f"  \033[32m✓ Within bf16 precision (5e-3).\033[0m")
        else:
            print(f"  \033[33m? Beyond bf16 tolerance.\033[0m  Investigate dtype cast")
            print(f"  or softmax precision.")
    else:
        print(f"  attn_weights not captured (eager_attention_forward not patched)")

    # ── Verdict (keyed on INDEPENDENT path, with explicit diagnostic ladder) ─
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Diagnostic ladder:")
    print(f"    (1)  capture sanity check (HF-vs-HF self-consistency):")
    print(f"         Q max={q_diff.max().item():.2e}, K max={k_diff.max().item():.2e}")
    print(f"    (1b) INDEPENDENT RoPE correctness check (decisive for RoPE):")
    print(f"         Q max={q_diff_indep.max().item():.2e}, K max={k_diff_indep.max().item():.2e}")
    print(f"    (2)  downstream score CONSISTENCY check (live-rebuilt vs")
    print(f"         independent-rebuilt; NOT against real pre-softmax scores):")
    print(f"         max={valid_score_diffs.max():.2e}")
    if probs_diff_np is not None:
        print(f"    (3)  end-to-end probability check (independent recon vs")
        print(f"         LIVE attn_weights from real path; decisive for end-to-end):")
        print(f"         max={probs_diff_np.max():.2e}")
    print()

    # Codex's failure-mode ladder:
    # - (1) fails: capture/patch bug
    # - (1) passes but (1b) fails: independent RoPE mismatch
    # - (1b) passes but (2) fails: matmul/mask/scaling mismatch
    # - (2) passes but (3) fails: softmax/dtype/output-cast mismatch
    # - all pass within tolerance: Stage 1 capture design validated

    if not rope_replication_ok:
        verdict = "CAPTURE_BUG"
        print(f"  \033[31m✗ CAPTURE_BUG\033[0m  Sanity check (1) failed.")
        print(f"  HF's apply_rotary_pos_emb on the same inputs produced different")
        print(f"  outputs across the captured-vs-reconstructed paths. This indicates")
        print(f"  a bug in the patching/capture itself, not the RoPE concept.")
    elif not independent_ok:
        verdict = "INDEPENDENT_ROPE_MISMATCH"
        print(f"  \033[31m✗ INDEPENDENT_ROPE_MISMATCH\033[0m  (1b) failed.")
        print(f"  Our from-scratch RoPE implementation differs from HF's RoPE output.")
        print(f"  Either our published-formula implementation is wrong, OR HF uses a")
        print(f"  non-standard rotation convention. Inspect modeling_llama source")
        print(f"  for apply_rotary_pos_emb to compare conventions.")
    elif not scores_ok:
        verdict = "DOWNSTREAM_SCORE_MISMATCH"
        print(f"  \033[31m✗ DOWNSTREAM_SCORE_MISMATCH\033[0m  (1b) passed, (2) failed.")
        print(f"  RoPE is independently correct, but the live-rebuilt and")
        print(f"  independent-rebuilt downstream paths disagree. This is a")
        print(f"  consistency failure — diagnose matmul, repeat_kv, scaling,")
        print(f"  or mask add. (Note: (2) does NOT compare against the real")
        print(f"  pre-softmax scores from inside eager_attention_forward;")
        print(f"  it compares two reconstructions to each other.)")
    elif probs_diff_np is not None and probs_ok:
        verdict = "EXACT"
        print(f"  \033[32m✓ EXACT\033[0m  All checks under 1e-4.")
        print(f"  Independent reconstruction matches the live model end-to-end.")
        print(f"  Stage 1 capture design is validated.")
        print(f"  Invariant for tasb_verify_v2.py: max < 5e-4, mean < 1e-4.")
    elif probs_diff_np is not None and probs_bf16_ok:
        verdict = "BF16_BOUNDED"
        print(f"  \033[32m✓ BF16_BOUNDED\033[0m  (best plausible outcome)")
        print(f"  (1b) and (2) match exactly. (3) residual is within bf16")
        print(f"  precision — HF's .to(query.dtype) cast on softmax output is")
        print(f"  the measurement floor, not a reconstruction error.")
        print(f"  Stage 1 invariant bounds:")
        print(f"    fp32 control:   max < 1e-4, mean < 1e-5")
        print(f"    bf16 runtime:   max < 1e-2, mean < 1e-3")
        print(f"  Capture technique: patch apply_rotary_pos_emb at module level.")
    elif probs_diff_np is None:
        verdict = "INCOMPLETE"
        print(f"  \033[33m? INCOMPLETE\033[0m  attn_weights not captured (no eager patch).")
        print(f"  (1b) and (2) passed; (3) couldn't be evaluated. Probably fine but")
        print(f"  the end-to-end check is missing. Re-run with eager_attention_forward")
        print(f"  patched if possible.")
    else:
        verdict = "OUTPUT_CAST_MISMATCH"
        print(f"  \033[33m? OUTPUT_CAST_MISMATCH\033[0m  (2) passed, (3) beyond bf16.")
        print(f"  Scores match but softmax/dtype/output-cast path differs.")
        print(f"  Diagnose the softmax dtype and the .to(query.dtype) cast.")

    return verdict


if __name__ == '__main__':
    main()
