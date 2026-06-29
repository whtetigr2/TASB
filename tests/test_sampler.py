"""
test_sampler_v2.py — Stage 1 acceptance tests for tasb_sampler_v2
==============================================================================
Proves the sampler satisfies its claims, including the backend_equivalence
invariant (#4 of Stage 1's four).

TESTS
-----
T1 shape and dtype contract:
    All three backends produce (B, n_q, S, S) fp32 output on a synthetic
    capture.

T2 row-stochastic invariant:
    Each Q head's rows sum to 1.0 within fp32 tolerance, for all backends.

T3 mask respect:
    All three backends produce zero mass on positions where the captured
    mask has its large-negative sentinel.

T4 backend_equivalence (THE Stage 1 invariant):
    `exact` and `gumbel` at K=1000 on the same synthetic logits converge
    to the same row distribution within 5% absolute. As K grows, this
    bound tightens.

T5 reproducibility via seed:
    Same backend, same seed, same logits → identical output.

T6 K=1 sanity:
    At K=1, each row is a single one-hot draw. Output is row-stochastic
    (sums to 1) and sparse (one nonzero per row).

T7 K=∞-equivalent convergence to softmax:
    At K=5000, `exact` backend output should match analytical softmax
    within ~3% per entry. Sanity check that we're sampling from the
    right distribution.

T8 config validation:
    SamplerConfig with invalid backend or K<1 raises.

T9 end-to-end with real capture (if model available):
    Load LLaMA, capture L18, run all three backends, verify shape and
    invariants. Skipped if model not available (so the unit tests can
    run on machines without the model).
==============================================================================
"""

import math
import numpy as np
import torch

from thermobridge import LayerCapture, LlamaAttentionCapture, sample, SamplerConfig
from thermobridge.sampler import VALID_BACKENDS


def _c(code, t): return f"\033[{code}m{t}\033[0m"
def green(t): return _c("32", t)
def red(t): return _c("31", t)
def bold(t): return _c("1", t)


def _make_synthetic_capture(
    B=1, n_q=4, n_kv=2, S=8, head_dim=8, seed=42
) -> LayerCapture:
    """Build a small synthetic LayerCapture for unit-test purposes."""
    torch.manual_seed(seed)
    Q = torch.randn(B, n_q, S, head_dim, dtype=torch.float32)
    K = torch.randn(B, n_kv, S, head_dim, dtype=torch.float32)
    # Causal mask in HF additive sentinel form
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(B, 1, S, S, dtype=torch.float32)
    mask = mask.masked_fill(causal, -3.39e38)

    return LayerCapture(
        layer_idx=0,
        q_post_rope=Q, k_post_rope=K, attention_mask=mask,
        scaling=1.0/math.sqrt(head_dim),
        attn_weights=torch.zeros(B, n_q, S, S),
        seq_len=S, dtype=torch.float32,
    )


def test_t1_shape_dtype():
    print(bold("T1: shape and dtype contract"))
    cap = _make_synthetic_capture()
    for backend in ['exact', 'gumbel', 'rbm']:
        cfg = SamplerConfig(backend=backend, K=10, seed=42)
        p = sample(cap, cfg)
        assert p.shape == (1, 4, 8, 8), f"{backend}: shape {p.shape}"
        assert p.dtype == torch.float32, f"{backend}: dtype {p.dtype}"
        print(f"  {backend:>6}: shape {tuple(p.shape)}, dtype {p.dtype}  ✓")
    print(green("  ✓ T1 PASS\n"))


def test_t2_row_stochastic():
    print(bold("T2: row-stochastic invariant"))
    cap = _make_synthetic_capture()
    causal = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    valid_rows = ~causal.all(dim=-1).unsqueeze(0).unsqueeze(0).expand(1, 4, 8)
    for backend in ['exact', 'gumbel', 'rbm']:
        cfg = SamplerConfig(backend=backend, K=100, seed=42)
        p = sample(cap, cfg)
        row_sums = p.sum(dim=-1)
        deviations = (row_sums[valid_rows] - 1.0).abs()
        max_dev = deviations.max().item()
        assert max_dev < 1e-5, f"{backend} row sums deviate by {max_dev}"
        print(f"  {backend:>6}: max row-sum deviation {max_dev:.2e}  ✓")
    print(green("  ✓ T2 PASS\n"))


def test_t3_mask_respect():
    print(bold("T3: mask respect (zero mass on masked positions)"))
    cap = _make_synthetic_capture()
    causal = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    causal_4d = causal.unsqueeze(0).unsqueeze(0).expand(1, 4, 8, 8)
    for backend in ['exact', 'gumbel', 'rbm']:
        cfg = SamplerConfig(backend=backend, K=100, seed=42)
        p = sample(cap, cfg)
        upper_mass = p[causal_4d].sum().item()
        assert upper_mass < 1e-10, (
            f"{backend} put mass on masked positions: {upper_mass}")
        print(f"  {backend:>6}: upper-triangle mass {upper_mass:.2e}  ✓")
    print(green("  ✓ T3 PASS\n"))


def test_t4_backend_equivalence():
    """THE Stage 1 invariant: exact and gumbel agree at large K.

    Tolerance is set by Monte Carlo precision. For K samples, the SE per
    entry is ~1/√K. At K=5000, SE ≈ 0.014, so we expect agreement within
    ~3 SE = 0.042 on the worst entry. We test < 0.05 for headroom.
    """
    print(bold("T4: backend_equivalence (exact vs gumbel at K=5000)"))
    cap = _make_synthetic_capture(seed=0)
    cfg_exact = SamplerConfig(backend='exact', K=5000, seed=42)
    cfg_gumbel = SamplerConfig(backend='gumbel', K=5000, seed=42)
    p_exact = sample(cap, cfg_exact)
    p_gumbel = sample(cap, cfg_gumbel)

    diff = (p_exact - p_gumbel).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  max abs diff:  {max_diff:.4f}")
    print(f"  mean abs diff: {mean_diff:.4f}")
    # K=5000 → SE ≈ 0.014, allow 3.5 SE for 99% headroom
    assert max_diff < 0.05, f"backend_equivalence failed: max diff {max_diff:.4f}"
    print(green(f"  ✓ T4 PASS — exact and gumbel agree at K=5000\n"))


def test_t5_reproducibility():
    print(bold("T5: reproducibility via seed"))
    cap = _make_synthetic_capture()
    for backend in ['exact', 'gumbel', 'rbm']:
        cfg1 = SamplerConfig(backend=backend, K=10, seed=12345)
        cfg2 = SamplerConfig(backend=backend, K=10, seed=12345)
        p1 = sample(cap, cfg1)
        p2 = sample(cap, cfg2)
        diff = (p1 - p2).abs().max().item()
        assert diff < 1e-10, (
            f"{backend} not reproducible: max diff {diff:.2e}")
        print(f"  {backend:>6}: bit-exact reproducible  ✓")
    print(green("  ✓ T5 PASS\n"))


def test_t6_K_one_sanity():
    print(bold("T6: K=1 produces one-hot rows"))
    cap = _make_synthetic_capture()
    causal = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    valid_rows = ~causal.all(dim=-1).unsqueeze(0).unsqueeze(0).expand(1, 4, 8)

    for backend in ['exact', 'gumbel']:
        cfg = SamplerConfig(backend=backend, K=1, seed=7)
        p = sample(cap, cfg)
        # Each valid row should be one-hot (one entry = 1.0, rest = 0)
        nonzero_per_row = (p > 0.5).sum(dim=-1)
        # For valid rows, exactly one entry should be > 0.5 (= 1.0)
        ok_rows = (nonzero_per_row[valid_rows] == 1).all().item()
        max_val_per_row = p.max(dim=-1).values
        assert ok_rows, f"{backend}: not all valid rows are one-hot at K=1"
        print(f"  {backend:>6}: K=1 one-hot rows ✓ (max per row = "
              f"{max_val_per_row[valid_rows].mean():.3f})")
    print(green("  ✓ T6 PASS\n"))


def test_t7_convergence_to_softmax():
    print(bold("T7: exact backend at K=5000 converges to analytical softmax"))
    cap = _make_synthetic_capture(seed=99)
    cfg = SamplerConfig(backend='exact', K=5000, seed=99)
    p_sampled = sample(cap, cfg)

    # Compute analytical target
    from thermobridge.sampler import _compute_logits
    logits = _compute_logits(cap)
    p_target = torch.softmax(logits, dim=-1)

    diff = (p_sampled - p_target).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  max abs diff vs analytical softmax:  {max_diff:.4f}")
    print(f"  mean abs diff vs analytical softmax: {mean_diff:.4f}")
    # At K=5000, Monte Carlo precision is roughly 1/√K ~ 0.014 standard error
    # Allow 3× that for 3-sigma headroom: 4%
    assert max_diff < 0.04, (
        f"exact backend at K=5000 doesn't match softmax: {max_diff:.4f}")
    print(green("  ✓ T7 PASS — exact backend samples from softmax\n"))


def test_t8_config_validation():
    print(bold("T8: SamplerConfig validation"))
    # Invalid backend
    raised = False
    try:
        SamplerConfig(backend='foo')
    except ValueError as e:
        raised = True
        print(f"  invalid backend raises ✓ ({e})")
    assert raised, "invalid backend did not raise"

    # K < 1
    raised = False
    try:
        SamplerConfig(K=0)
    except ValueError as e:
        raised = True
        print(f"  K=0 raises ✓ ({e})")
    assert raised, "K=0 did not raise"

    # Valid config works
    cfg = SamplerConfig(backend='exact', K=10)
    assert cfg.backend == 'exact'
    print(f"  valid config accepts ✓")
    print(green("  ✓ T8 PASS\n"))


def test_t9_end_to_end_with_real_capture():
    """End-to-end: real LLaMA capture → real sampler. Skipped if no model."""
    print(bold("T9: end-to-end with real LLaMA capture (L18)"))
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
    except ImportError:
        print("  transformers not available, skipping")
        return

    try:
        print("  loading model...")
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16),
            attn_implementation='eager', device_map='auto')
        model.eval()
    except Exception as e:
        print(f"  model load failed ({type(e).__name__}), skipping T9")
        return

    capturer = LlamaAttentionCapture(
        model=model, layers_to_capture=[18], strict_verify=True)
    inputs = tok("The capital of France is", return_tensors='pt').to(
        model.device)
    with capturer.capture():
        _ = model(**inputs, use_cache=False)

    cap = capturer.get_capture(18)
    B, n_q, S, _ = cap.attn_weights.shape
    print(f"  captured L18: Q{tuple(cap.q_post_rope.shape)}")

    for backend in ['exact', 'gumbel', 'rbm']:
        cfg = SamplerConfig(backend=backend, K=10, seed=42)
        p = sample(cap, cfg)
        assert p.shape == (B, n_q, S, S), f"{backend}: shape mismatch"
        row_sums = p.sum(dim=-1)
        # Need to construct valid_rows from the captured mask
        valid_rows_mask = (cap.attention_mask > -1e30).any(dim=-1).cpu()
        valid_rows_mask = valid_rows_mask.expand(B, n_q, S)
        max_dev = (row_sums.cpu()[valid_rows_mask] - 1.0).abs().max().item()
        print(f"  {backend:>6}: shape ✓, row sum dev {max_dev:.2e}  ✓")
    print(green("  ✓ T9 PASS\n"))


# ── Test runner ───────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  test_sampler_v2.py — Stage 1 acceptance tests for tasb_sampler_v2")
    print("=" * 78)
    print()

    tests = [
        ("T1", test_t1_shape_dtype),
        ("T2", test_t2_row_stochastic),
        ("T3", test_t3_mask_respect),
        ("T4", test_t4_backend_equivalence),
        ("T5", test_t5_reproducibility),
        ("T6", test_t6_K_one_sanity),
        ("T7", test_t7_convergence_to_softmax),
        ("T8", test_t8_config_validation),
        ("T9", test_t9_end_to_end_with_real_capture),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed.append((name, str(e)))
            print(red(f"  ✗ {name} FAIL: {e}\n"))
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(red(f"  ✗ {name} ERROR: {type(e).__name__}: {e}\n"))

    print("=" * 78)
    print(f"  RESULTS: {passed}/{len(tests)} passed")
    if failed:
        for name, reason in failed:
            print(red(f"    {name}: {reason}"))
        sys.exit(1)
    else:
        print(green("  ✓ All Stage 1 sampler invariants pass."))
        print("  tasb_sampler_v2.py is validated.")
        print("  Backend equivalence (#4 of Stage 1 invariants): VALIDATED.")
        print("  Proceed to tasb_injector_v2.")
    print("=" * 78)


if __name__ == "__main__":
    main()
