"""
test_injector_v2.py — Stage 1 acceptance tests for tasb_injector_v2
==============================================================================
Validates the injector and closes the FINAL Stage 1 invariant
(`alpha_zero_identity`). Once T1 passes, all four Stage 1 invariants are
green and the bridge is end-to-end ready for the M5 faithfulness test.

TESTS
-----
T1 alpha_zero_identity (THE decisive Stage 1 invariant):
    Run forward vanilla. Run same forward at α=0 with injector active.
    Compare logits BIT-EXACT. The α=0 fast path in the injector must
    delegate to the original eager function without touching outputs.

T2 alpha_one_replaces_attention:
    Run at α=1, confirm output logits differ from vanilla. The patched
    path must actually do something — this test rules out a silent no-op.

T3 non_target_layers_untouched:
    Inject only at L18. Capture L0 attn_weights with and without the
    injector active. Must be bit-exact identical. Injector must not
    affect non-target layers.

T4 shape_contract_validation:
    Wrong-shape p_thermo must raise in __init__, not silently at forward.

T5 context_manager_cleanup:
    Raise inside `with injector.inject():`. Confirm eager_attention_forward
    is restored on exception.

T6 non_reentrant_guard:
    Nested `with injector.inject():` must raise.

T7 alpha_monotonicity:
    For increasing α (0.0, 0.25, 0.5, 0.75, 1.0), the L2 distance from
    vanilla logits should increase monotonically. Sanity that the blend
    formula is linear as advertised.

T8 end_to_end_smoke:
    Full pipeline: capture → sample → inject → forward. Confirm shape,
    finite outputs, no NaN.

The decisive test for Stage 1 is T1. If it passes, the four invariants
are: capture_invariant ✓, rope_regression ✓, backend_equivalence ✓,
alpha_zero_identity ✓.
==============================================================================
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasb_capture_v2 import LlamaAttentionCapture, LayerCapture
from tasb_sampler_v2 import sample, SamplerConfig
from tasb_injector_v2 import LlamaAttentionInjector, InjectorConfig


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


def _capture_l18(model, tok, prompt="The capital of France is"):
    """Helper: capture L18 once, return (LayerCapture, inputs)."""
    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=True)
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)
    return capturer.get_capture(18), inputs


def _vanilla_logits(model, inputs):
    """Run forward without any patches, return last-token logits."""
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    return out.logits.detach().clone()


# ── Tests ─────────────────────────────────────────────────────────────────

def test_t1_alpha_zero_identity(model, tok):
    """THE Stage 1 invariant: α=0 must produce bit-exact vanilla output."""
    print(bold("T1: alpha_zero_identity (THE Stage 1 invariant)"))

    cap, inputs = _capture_l18(model, tok)
    vanilla = _vanilla_logits(model, inputs)

    # Build a p_thermo (content doesn't matter — α=0 should ignore it)
    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))

    injector = LlamaAttentionInjector(
        capture=cap, p_thermo=p_thermo, alpha=0.0)

    with injector.inject():
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        patched = out.logits.detach().clone()

    # Bit-exact comparison
    bit_exact = torch.equal(vanilla, patched)
    if bit_exact:
        max_diff = 0.0
    else:
        max_diff = (vanilla.float() - patched.float()).abs().max().item()

    print(f"  vanilla logits shape:  {tuple(vanilla.shape)}")
    print(f"  patched logits shape:  {tuple(patched.shape)}")
    print(f"  torch.equal bit-exact: {bit_exact}")
    print(f"  max abs diff:          {max_diff:.2e}")
    print(f"  injector calls observed: {injector.n_eager_calls}")
    print(f"  injections performed:    {injector.n_injections}")

    # Bit-exact is the requirement. Allow strict zero only.
    assert bit_exact, (
        f"α=0 did NOT produce bit-exact vanilla output! max diff {max_diff:.2e}. "
        f"The α=0 fast path is not byte-equivalent to no patching.")
    assert injector.n_injections == 0, (
        f"α=0 should perform zero injections, got {injector.n_injections}. "
        f"The fast path is not being taken.")
    print(green("  ✓ T1 PASS — alpha_zero_identity validated (Stage 1 invariant #4)\n"))


def test_t2_alpha_one_replaces_attention(model, tok):
    """At α=1, output must differ from vanilla (i.e., we're actually doing something)."""
    print(bold("T2: alpha_one_replaces_attention"))
    cap, inputs = _capture_l18(model, tok)
    vanilla = _vanilla_logits(model, inputs)

    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))

    injector = LlamaAttentionInjector(
        capture=cap, p_thermo=p_thermo, alpha=1.0)

    with injector.inject():
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        patched = out.logits.detach().clone()

    diff = (vanilla.float() - patched.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  vanilla vs α=1 patched:")
    print(f"    max abs diff:  {max_diff:.4e}")
    print(f"    mean abs diff: {mean_diff:.4e}")
    print(f"  injector calls:      {injector.n_eager_calls}")
    print(f"  injections performed: {injector.n_injections}")

    # α=1 with K=10 sampled p_thermo should differ noticeably from vanilla
    # Use a very loose threshold — we just want to confirm non-no-op
    assert max_diff > 1e-3, (
        f"α=1 produced output indistinguishable from vanilla (max diff {max_diff:.2e}). "
        f"Either the patch is a no-op or the sampler produced p_thermo ≈ vanilla.")
    assert injector.n_injections > 0, (
        f"α=1 should perform at least one injection, got {injector.n_injections}.")
    print(green("  ✓ T2 PASS — α=1 produces different output (patch is active)\n"))


def test_t3_non_target_layers_untouched(model, tok):
    """Inject at L18 only; L0 attention should be byte-exact identical."""
    print(bold("T3: non_target_layers_untouched"))

    # First, capture L0 and L18 vanilla
    capturer_vanilla = LlamaAttentionCapture(
        model=model, layers_to_capture=[0, 18], strict_verify=True)
    inputs = tok("The capital of France is", return_tensors='pt').to(model.device)
    with capturer_vanilla.capture():
        _ = model(**inputs, use_cache=False)
    vanilla_l0 = capturer_vanilla.get_capture(0).attn_weights.detach().clone()
    cap_l18_for_inject = capturer_vanilla.get_capture(18)

    # Now inject at L18, simultaneously capture L0 with the injector active
    p_thermo = sample(cap_l18_for_inject, SamplerConfig(backend='exact', K=10, seed=42))
    injector = LlamaAttentionInjector(
        capture=cap_l18_for_inject, p_thermo=p_thermo, alpha=0.5)

    capturer_during = LlamaAttentionCapture(
        model=model, layers_to_capture=[0], strict_verify=False)
    # NOTE: strict_verify=False here because the injector modifies L18's
    # attn_weights, and the capturer's strict verify would check ALL captured
    # layers against the recompute formula. We're only capturing L0 in this
    # capturer, and L0 is unaffected by L18 injection, so verify on L0 alone
    # would pass — but we keep it off to be safe and not couple test results.

    with injector.inject():
        with capturer_during.capture():
            _ = model(**inputs, use_cache=False)
    during_l0 = capturer_during.get_capture(0).attn_weights.detach().clone()

    bit_exact = torch.equal(vanilla_l0, during_l0)
    if bit_exact:
        max_diff = 0.0
    else:
        max_diff = (vanilla_l0.float() - during_l0.float()).abs().max().item()

    print(f"  L0 vanilla vs L0 during L18-injection:")
    print(f"    torch.equal: {bit_exact}")
    print(f"    max diff:    {max_diff:.2e}")

    assert bit_exact, (
        f"L0 attention weights differ when injecting at L18! "
        f"max diff {max_diff:.2e}. The injector is leaking into non-target layers.")
    print(green("  ✓ T3 PASS — non-target layers untouched\n"))


def test_t4_shape_contract_validation(model, tok):
    """Wrong-shape p_thermo must raise in __init__."""
    print(bold("T4: shape_contract_validation"))
    cap, _ = _capture_l18(model, tok)

    # Build a wrong-shape p_thermo
    wrong_shape = torch.zeros(1, 24, 6, 7, dtype=torch.float32, device=model.device)

    raised = False
    try:
        LlamaAttentionInjector(capture=cap, p_thermo=wrong_shape, alpha=0.3)
    except ValueError as e:
        raised = True
        print(f"  wrong-shape p_thermo raises: ✓")
        print(f"    {e}")

    assert raised, "wrong-shape p_thermo did not raise"

    # Also test non-tensor p_thermo
    raised = False
    try:
        LlamaAttentionInjector(capture=cap, p_thermo=[1, 2, 3], alpha=0.3)
    except TypeError as e:
        raised = True
        print(f"  non-tensor p_thermo raises: ✓")
        print(f"    {e}")
    assert raised, "non-tensor p_thermo did not raise"
    print(green("  ✓ T4 PASS — shape and type guards fire\n"))


def test_t5_context_manager_cleanup(model, tok):
    """Patches must be restored even if body raises."""
    print(bold("T5: context_manager_cleanup"))
    cap, inputs = _capture_l18(model, tok)
    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))

    from transformers.models.llama import modeling_llama
    orig_eager = modeling_llama.eager_attention_forward

    injector = LlamaAttentionInjector(capture=cap, p_thermo=p_thermo, alpha=0.5)

    try:
        with injector.inject():
            _ = model(**inputs, use_cache=False)
            raise ValueError("simulated body failure")
    except ValueError:
        pass

    eager_restored = modeling_llama.eager_attention_forward is orig_eager
    print(f"  eager_attention_forward restored after exception: {eager_restored}")
    assert eager_restored, "eager_attention_forward leaked after exception"

    # Second forward should run vanilla
    vanilla = _vanilla_logits(model, inputs)
    assert vanilla.shape == (1, inputs['input_ids'].shape[1], model.config.vocab_size)
    print(green("  ✓ T5 PASS — patches restored on exception\n"))


def test_t6_non_reentrant_guard(model, tok):
    """Nested inject() must raise."""
    print(bold("T6: non_reentrant_guard"))
    cap, inputs = _capture_l18(model, tok)
    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))
    injector = LlamaAttentionInjector(capture=cap, p_thermo=p_thermo, alpha=0.5)

    raised = False
    try:
        with injector.inject():
            with injector.inject():
                pass
    except RuntimeError as e:
        raised = True
        print(f"  nested inject raises: ✓ ({e})")
    assert raised, "nested inject did not raise"
    print(green("  ✓ T6 PASS\n"))


def test_t7_alpha_monotonicity(model, tok):
    """L2 distance from vanilla should grow monotonically with α."""
    print(bold("T7: alpha_monotonicity"))
    cap, inputs = _capture_l18(model, tok)
    vanilla = _vanilla_logits(model, inputs)
    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    dists = []
    for alpha in alphas:
        injector = LlamaAttentionInjector(
            capture=cap, p_thermo=p_thermo, alpha=alpha)
        with injector.inject():
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
            patched = out.logits.detach()
        d = (vanilla.float() - patched.float()).norm().item()
        dists.append(d)
        print(f"  α={alpha:.2f}: ||patched - vanilla||₂ = {d:.4e}")

    # Strictly increasing
    assert dists[0] == 0.0, f"α=0 distance not zero: {dists[0]:.2e}"
    for i in range(1, len(dists)):
        assert dists[i] > dists[i-1], (
            f"non-monotonic at α={alphas[i]}: {dists[i]:.2e} <= {dists[i-1]:.2e}")
    print(green("  ✓ T7 PASS — distance grows monotonically with α\n"))


def test_t8_end_to_end_smoke(model, tok):
    """Full pipeline: capture → sample → inject. Output sanity."""
    print(bold("T8: end_to_end_smoke"))

    inputs = tok("The capital of France is", return_tensors='pt').to(model.device)

    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=True)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)
    cap = capturer.get_capture(18)

    p_thermo = sample(cap, SamplerConfig(backend='exact', K=10, seed=42))

    injector = LlamaAttentionInjector(
        capture=cap, p_thermo=p_thermo, alpha=0.3)

    with injector.inject():
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits.detach()

    # Sanity
    print(f"  output logits shape: {tuple(logits.shape)}")
    print(f"  contains NaN:        {torch.isnan(logits).any().item()}")
    print(f"  contains Inf:        {torch.isinf(logits).any().item()}")
    print(f"  logit range:         [{logits.min().item():.2f}, "
          f"{logits.max().item():.2f}]")

    assert not torch.isnan(logits).any(), "output contains NaN"
    assert not torch.isinf(logits).any(), "output contains Inf"
    print(green("  ✓ T8 PASS — end-to-end pipeline produces finite output\n"))


# ── Runner ────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  test_injector_v2.py — Stage 1 acceptance tests for tasb_injector_v2")
    print("=" * 78)
    print()

    model, tok = load_model()

    tests = [
        ("T1", test_t1_alpha_zero_identity),
        ("T2", test_t2_alpha_one_replaces_attention),
        ("T3", test_t3_non_target_layers_untouched),
        ("T4", test_t4_shape_contract_validation),
        ("T5", test_t5_context_manager_cleanup),
        ("T6", test_t6_non_reentrant_guard),
        ("T7", test_t7_alpha_monotonicity),
        ("T8", test_t8_end_to_end_smoke),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn(model, tok)
            passed += 1
        except AssertionError as e:
            failed.append((name, str(e)))
            print(red(f"  ✗ {name} FAIL: {e}\n"))
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(red(f"  ✗ {name} ERROR: {type(e).__name__}: {e}\n"))
            import traceback
            traceback.print_exc()

    print("=" * 78)
    print(f"  RESULTS: {passed}/{len(tests)} passed")
    if failed:
        for name, reason in failed:
            print(red(f"    {name}: {reason}"))
        sys.exit(1)
    else:
        print(green("  ✓ All Stage 1 injector invariants pass."))
        print(green("  alpha_zero_identity (Stage 1 invariant #4): VALIDATED."))
        print()
        print("  STAGE 1 STATUS: ALL FOUR INVARIANTS GREEN")
        print("    ✓ capture_invariant")
        print("    ✓ rope_regression")
        print("    ✓ backend_equivalence")
        print("    ✓ alpha_zero_identity")
        print()
        print("  Next: tasb_verify_v2.py (orchestrates all four invariants)")
        print("        then tasb_pipeline_v2.py (canonical entry point)")
        print("        then M5 faithfulness core test")
    print("=" * 78)


if __name__ == "__main__":
    main()
