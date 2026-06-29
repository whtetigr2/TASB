"""
test_capture_v2.py — Invariant tests for tasb_capture_v2.py
==============================================================================
Proves that the production capture path satisfies every claim made in
tasb_capture_v2.py. This is Stage 1 acceptance test #1 (capture_invariant).

TESTS
-----
T1 single-layer single-step capture:
    Capture L18 on one short prompt. Verify:
    - capture invariant max < 5e-4
    - shapes are (B, n_q, S, head_dim) for Q, (B, n_kv, S, head_dim) for K
    - attn_weights shape (B, n_q, S, S)
    - mask present and shape (B, 1, S, S)
    - scaling ≈ 1/√128

T2 multi-layer capture:
    Capture L0, L6, L12, L18, L24, L27 simultaneously on one prompt.
    Verify every layer satisfies the invariant, layer_idx fields match
    requested layers.

T3 layer subset filtering:
    Capture only L18 on a forward that touches all 28 layers.
    Verify exactly one capture present and it's L18.

T4 context manager cleanup:
    Verify patches are restored after `with capturer.capture():` even if
    body raises. Test by running a capture that raises, then running a
    second forward and confirming HF behaves vanilla (no leaked patches).

T5 strict_verify failure path:
    Manually corrupt a capture and confirm verify_capture() raises.

T6 layer identification via stack walk:
    Capture L18 only on a forward that hits L0..L27 in sequence. Confirm
    layer_idx == 18 was identified correctly (not 0, not 17, not 19).

T7 signature-adaptive patching (Bug #9 production version):
    Confirm the patched functions don't forward kwargs that aren't in the
    original signature. Test by inspecting the filter_kwargs helper.

T8 non-reentrant guard:
    Two nested `with capture.capture():` blocks should raise.

This file is NOT a benchmark or a stress test. It's the canary that fails
loudly when something foundational breaks. Run before every Stage 1 change.
==============================================================================
"""

import numpy as np
import torch

from thermobridge import LlamaAttentionCapture, LayerCapture
from thermobridge.capture import _ROTARY_PARAMS, _EAGER_PARAMS


def _c(code, t): return f"\033[{code}m{t}\033[0m"
def green(t): return _c("32", t)
def red(t): return _c("31", t)
def yellow(t): return _c("33", t)
def bold(t): return _c("1", t)


def load_model():
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
    print("[setup] Loading LLaMA 3.2-3B (4-bit)...", flush=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),
        attn_implementation='eager', device_map='auto')
    model.eval()
    print(f"[setup] Ready on {next(model.parameters()).device}\n")
    return model, tok


# ── Test functions ────────────────────────────────────────────────────────

def test_t1_single_layer_single_step(model, tok):
    """T1: Capture L18 on one prompt. All invariants must hold."""
    print(bold("T1: single-layer single-step capture (L18)"))
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=True)

    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    cap = capturer.get_capture(18)
    print(f"  Q shape: {tuple(cap.q_post_rope.shape)}")
    print(f"  K shape: {tuple(cap.k_post_rope.shape)}")
    print(f"  attn_weights shape: {tuple(cap.attn_weights.shape)}")
    print(f"  attention_mask: "
          f"{tuple(cap.attention_mask.shape) if cap.attention_mask is not None else None}")
    print(f"  scaling: {cap.scaling:.6f} (expected ≈ {1/np.sqrt(128):.6f})")

    # Verify invariant explicitly
    results = capturer.verify_capture()
    diag = results[18]
    print(f"  capture invariant: max={diag['max_diff']:.2e}, "
          f"mean={diag['mean_diff']:.2e}")

    # Assertions
    assert cap.q_post_rope.shape[1] == 24, "expected 24 Q heads"
    assert cap.k_post_rope.shape[1] == 8, "expected 8 KV heads"
    assert cap.attn_weights.shape[1] == 24, "expected 24 Q heads in attn_weights"
    assert cap.attention_mask is not None, "expected live mask captured"
    assert abs(cap.scaling - 1.0/np.sqrt(128)) < 1e-6, "scaling != 1/√128"
    assert diag['ok'], (
        f"capture invariant failed: max={diag['max_diff']:.2e} "
        f"> tol={diag['atol']:.0e}")
    print(green("  ✓ T1 PASS\n"))
    return True


def test_t2_multi_layer(model, tok):
    """T2: Capture 6 layers simultaneously."""
    print(bold("T2: multi-layer capture (L0, L6, L12, L18, L24, L27)"))
    layers = [0, 6, 12, 18, 24, 27]
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=layers, strict_verify=True)

    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    captured = capturer.get_captured()
    print(f"  captured layers: {sorted(captured.keys())}")
    results = capturer.verify_capture()
    for L in layers:
        diag = results[L]
        print(f"  L{L:>2}: max={diag['max_diff']:.2e}, "
              f"mean={diag['mean_diff']:.2e}  "
              f"{'✓' if diag['ok'] else '✗'}")

    assert set(captured.keys()) == set(layers), (
        f"missing layers: requested {layers}, got {sorted(captured.keys())}")
    for L in layers:
        assert results[L]['ok'], f"L{L} invariant failed"
        assert captured[L].layer_idx == L, (
            f"layer_idx mismatch: stored {captured[L].layer_idx} != requested {L}")
    print(green("  ✓ T2 PASS\n"))
    return True


def test_t3_layer_subset(model, tok):
    """T3: Request only L18, confirm nothing else is captured."""
    print(bold("T3: layer subset filtering"))
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=True)
    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    captured = capturer.get_captured()
    print(f"  captured layers: {list(captured.keys())}")
    print(f"  total rope calls: {capturer.n_rope_calls} "
          f"(expected 28, one per layer)")
    print(f"  total eager calls: {capturer.n_eager_calls} (expected 28)")

    assert list(captured.keys()) == [18], (
        f"subset filter failed: got {list(captured.keys())}")
    assert capturer.n_rope_calls == 28, (
        f"expected 28 rope calls, got {capturer.n_rope_calls}")
    print(green("  ✓ T3 PASS\n"))
    return True


def test_t4_context_manager_cleanup(model, tok):
    """T4: Patches must be restored even if body raises."""
    print(bold("T4: context manager cleanup (patch restoration)"))
    from transformers.models.llama import modeling_llama
    orig_rope = modeling_llama.apply_rotary_pos_emb
    orig_eager = getattr(modeling_llama, 'eager_attention_forward', None)

    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=False)

    # Force a raise inside the body
    inputs = tok("Hi", return_tensors='pt').to(model.device)
    try:
        with capturer.capture():
            _ = model(**inputs, use_cache=False)
            raise ValueError("simulated body failure")
    except ValueError:
        pass

    # Patches must be restored despite the exception
    rope_restored = modeling_llama.apply_rotary_pos_emb is orig_rope
    if orig_eager is not None:
        eager_restored = modeling_llama.eager_attention_forward is orig_eager
    else:
        eager_restored = True
    print(f"  apply_rotary_pos_emb restored: {rope_restored}")
    print(f"  eager_attention_forward restored: {eager_restored}")

    assert rope_restored, "patched apply_rotary_pos_emb leaked!"
    assert eager_restored, "patched eager_attention_forward leaked!"

    # Run a second forward without capture; it should behave vanilla
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    print(green("  ✓ T4 PASS\n"))
    return True


def test_t5_strict_verify_failure_path(model, tok):
    """T5: A corrupted capture must cause verify_capture to raise."""
    print(bold("T5: strict_verify failure path"))
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18],
        strict_verify=False,   # we'll verify manually
        raise_on_verify_fail=True)
    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    # Verify passes initially
    results = capturer.verify_capture()
    assert results[18]['ok'], "initial verify failed"

    # Corrupt the capture: scale attn_weights by 1.5
    cap = capturer.captured[18]
    cap.attn_weights = cap.attn_weights * 1.5

    raised = False
    try:
        capturer.verify_capture()
    except RuntimeError as e:
        raised = True
        print(f"  verify_capture raised as expected:")
        print(f"  {str(e).splitlines()[0]}")

    assert raised, "verify_capture did not raise on corrupted state!"
    print(green("  ✓ T5 PASS\n"))
    return True


def test_t6_layer_identification(model, tok):
    """T6: layer_idx in stored capture matches requested layer."""
    print(bold("T6: layer identification via stack walk"))
    # Request a non-trivial layer to confirm we're not just always returning 0
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[17, 18, 19], strict_verify=True)
    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    for L in [17, 18, 19]:
        cap = capturer.get_capture(L)
        assert cap.layer_idx == L, (
            f"layer_idx mismatch: stored {cap.layer_idx} != requested {L}")
        print(f"  requested L{L}, stored layer_idx={cap.layer_idx} ✓")
    print(green("  ✓ T6 PASS\n"))
    return True


def test_t7_signature_adaptive_patching():
    """T7: Bug #9 production version — verify filter_kwargs drops unknowns.

    No model needed; this is a pure-Python test of the filtering logic.
    """
    print(bold("T7: signature-adaptive patching (Bug #9 production)"))
    # apply_rotary_pos_emb in some versions has no position_ids kwarg
    print(f"  apply_rotary signature params: {sorted(_ROTARY_PARAMS)}")
    print(f"  eager signature params:        {sorted(_EAGER_PARAMS)}")

    filtered = LlamaAttentionCapture._filter_kwargs(
        {'position_ids': None, 'unsqueeze_dim': 1, 'bogus_kwarg': 42},
        _ROTARY_PARAMS)
    print(f"  filtered kwargs (showing only those in signature): "
          f"{sorted(filtered.keys())}")
    assert 'bogus_kwarg' not in filtered, "filter failed to drop unknown kwarg"
    if 'unsqueeze_dim' in _ROTARY_PARAMS:
        assert 'unsqueeze_dim' in filtered, (
            "filter dropped a valid kwarg")
    print(green("  ✓ T7 PASS\n"))
    return True


def test_t8_non_reentrant_guard(model, tok):
    """T8: Nested captures should raise."""
    print(bold("T8: non-reentrant guard"))
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=False)
    inputs = tok("Hi", return_tensors='pt').to(model.device)

    raised = False
    try:
        with capturer.capture():
            with capturer.capture():  # should raise
                _ = model(**inputs, use_cache=False)
    except RuntimeError as e:
        raised = True
        print(f"  raised as expected: {e}")

    assert raised, "nested capture did not raise!"
    print(green("  ✓ T8 PASS\n"))
    return True


def test_t9_loud_fail_on_none_layer_idx(model, tok):
    """T9 (Bug #9a): if the stack-walk returns None during capture, the capture
    must raise immediately instead of silently dropping the layer."""
    print(bold("T9: capture loud-fails when _find_owning_layer_idx returns None"))
    inputs = tok("Hi", return_tensors='pt').to(model.device)
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=False)

    # Simulate a transformers upgrade breaking the stack-walk.
    orig_finder = LlamaAttentionCapture._find_owning_layer_idx
    LlamaAttentionCapture._find_owning_layer_idx = staticmethod(lambda: None)
    raised = False
    try:
        with capturer.capture():
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
    except RuntimeError as e:
        raised = True
        print(f"  raised as expected: {str(e)[:88]}...")
    finally:
        LlamaAttentionCapture._find_owning_layer_idx = staticmethod(orig_finder)

    assert raised, "capture did NOT raise on None layer_idx (silent skip!)"
    # Patches must be restored (no leak) -> a clean vanilla forward still works.
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    print(green("  ✓ T9 PASS\n"))
    return True


def test_t10_injector_loud_fail_on_bad_signature(model, tok):
    """T10 (Bug #9b): a target layer whose eager call can't supply value=args[3]
    must raise instead of silently skipping injection."""
    print(bold("T10: injector loud-fails on <4 positional args for a target layer"))
    import thermobridge.inject as inj_mod
    from thermobridge import LlamaAttentionInjector, DispatchEntry

    class FakeModule:
        layer_idx = 18
        num_key_value_groups = 1

    # Minimal valid LayerCapture (DispatchEntry type-checks it). The Bug #9b
    # guard fires before capture/value are ever read, so contents don't matter.
    cap = LayerCapture(
        layer_idx=18,
        q_post_rope=torch.zeros(1, 1, 2, 8),
        k_post_rope=torch.zeros(1, 1, 2, 8),
        attention_mask=None,
        scaling=1.0,
        attn_weights=torch.zeros(1, 1, 2, 2),
        seq_len=2,
        dtype=torch.float32,
    )
    entry = DispatchEntry(capture=cap,
                          p_thermo=torch.zeros(1, 1, 2, 2),
                          alpha=0.3)
    injector = LlamaAttentionInjector(dispatch_table={18: entry})

    # Guard fires before orig() is ever called, so a stub is never exercised;
    # ensure the closure can be built even if _ORIG_EAGER were unset.
    saved = inj_mod._ORIG_EAGER
    if inj_mod._ORIG_EAGER is None:
        inj_mod._ORIG_EAGER = lambda *a, **k: (None, None)
    raised = False
    try:
        patched = injector._make_patched_eager()
        assert patched is not None, "could not build patched eager closure"
        try:
            patched(FakeModule())   # only 1 positional arg -> len(args) < 4
        except RuntimeError as e:
            raised = True
            print(f"  raised as expected: {str(e)[:88]}...")
    finally:
        inj_mod._ORIG_EAGER = saved

    assert raised, "injector did NOT raise on <4 args for a target layer!"
    print(green("  ✓ T10 PASS\n"))
    return True


# ── Test runner ───────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  test_capture_v2.py — Stage 1 acceptance tests for tasb_capture_v2")
    print("=" * 78)
    print()

    # T7 doesn't need a model; run it first as a smoke test
    test_t7_signature_adaptive_patching()

    model, tok = load_model()

    tests = [
        ("T1", test_t1_single_layer_single_step),
        ("T2", test_t2_multi_layer),
        ("T3", test_t3_layer_subset),
        ("T4", test_t4_context_manager_cleanup),
        ("T5", test_t5_strict_verify_failure_path),
        ("T6", test_t6_layer_identification),
        ("T8", test_t8_non_reentrant_guard),
        ("T9", test_t9_loud_fail_on_none_layer_idx),
        ("T10", test_t10_injector_loud_fail_on_bad_signature),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        try:
            ok = fn(model, tok)
            if ok:
                passed += 1
        except AssertionError as e:
            failed.append((name, str(e)))
            print(red(f"  ✗ {name} FAIL: {e}\n"))
        except Exception as e:
            failed.append((name, f"unexpected exception: {type(e).__name__}: {e}"))
            print(red(f"  ✗ {name} ERROR: {type(e).__name__}: {e}\n"))

    print("=" * 78)
    print(f"  RESULTS: {passed + 1}/{len(tests) + 1} passed "
          f"(T7 always runs first)")
    if failed:
        for name, reason in failed:
            print(red(f"    {name}: {reason}"))
        sys.exit(1)
    else:
        print(green("  ✓ All Stage 1 capture invariants pass."))
        print("  tasb_capture_v2.py is validated. Proceed to tasb_sampler_v2.")
    print("=" * 78)


if __name__ == "__main__":
    main()
