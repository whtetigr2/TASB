"""
tests/test_multilayer_v1.py — Phase 2 regression tests for multi-layer refactor
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Validates the tasb_injector_v2 + tasb_pipeline_v2 multi-layer refactor
against all 15 requirements in tasb_multilayer_spec_v5.md.

All 15 tests must pass before running M7 sub-sweep 5. If any fail, the
API change has a bug and M7-5 numbers would be untrustworthy.

TEST MAP (spec requirement -> test method):
  Req  1: test_01_scalar_m5_frozen_bitexact
  Req  2: test_02_single_element_list_equals_scalar
  Req  3: test_03_list_alpha_zero_scalar_equals_vanilla
  Req  4: test_04_list_alpha_zero_list_equals_vanilla
  Req  5: test_05_scalar_alpha_broadcast_equals_list_alpha_uniform_2layer
  Req  6: test_06_scalar_alpha_broadcast_equals_list_alpha_uniform_3layer
  Req  7: test_07_unsorted_input_order_independence
  Req  8: test_08_duplicate_layer_raises
  Req  9: test_09_alpha_length_mismatch_raises
  Req 10: test_10_per_layer_seeds_stable_and_unique
  Req 11: test_11_return_intermediates_dict_fields_contract
  Req 12: test_12_m5_frozen_scalar_regression  (alias / belt-and-suspenders)
  Req 13: test_13_multilayer_gumbel_rbm_supported_thrml_blocked  (updated: Bug #2)
  Req 14: test_14_scalar_bridgeresult_scalar_fields_populated
  Req 15: test_15_per_layer_seeds_order_independent
  +Test16: test_16_multilayer_gumbel_rbm_isolation  (Bug #2 coverage, beyond spec)

CONFIDENT BUCKET DEFINITION (cite alongside any zero-flip claim):
  confident: prob_gap >= 0.5
  moderate:  0.1 <= prob_gap < 0.5
  ambiguous: prob_gap < 0.1

SETUP
-----
Tests that require the model load it once via setUpClass. The model is
loaded in 4-bit quantization to match M5/M6/M7 harness conditions.

Run with:
    python -m pytest tests/test_multilayer_v1.py -v
or:
    python tests/test_multilayer_v1.py
==============================================================================
"""

import sys
import unittest

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
LAYER_IDX  = 18          # production layer
LAYER_IDX2 = 24          # second layer for multi-layer tests
ALPHA      = 0.3         # production alpha
BASE_SEED  = 42          # matches M5/M6/M7 harnesses
K          = 10          # production K


# ---------------------------------------------------------------------------
# Shared model fixture
# ---------------------------------------------------------------------------
def _load_model():
    """Load LLaMA 3.2-3B in 4-bit, eager attention. Matches harness config."""
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
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
    return model, tok


def _tokenize(tok, prompt, device):
    return tok(prompt, return_tensors='pt').to(device)['input_ids']


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestMultilayerRefactor(unittest.TestCase):
    """15 regression tests for the multi-layer injector/pipeline refactor."""

    # Shared prompt battery (4 prompts, matches M7 sweep harness)
    PROMPTS = [
        "The capital of France is",
        "In thermodynamics, entropy is a measure of",
        "The quick brown fox jumps over",
        "Transformer models use attention mechanisms to",
    ]

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\n  Loading LLaMA 3.2-3B (4-bit)...", flush=True)
        cls.model, cls.tok = _load_model()
        cls.device = next(cls.model.parameters()).device
        print(f"  Model ready on {cls.device}\n", flush=True)

        # Pre-tokenize all prompts
        cls.input_ids_list = [
            _tokenize(cls.tok, p, cls.device) for p in cls.PROMPTS
        ]

    # ── Imports ─────────────────────────────────────────────────────────
    def _get_bridge_forward(self):
        from thermobridge import bridge_forward
        return bridge_forward

    def _get_seed_for_layer(self):
        from thermobridge import seed_for_layer
        return seed_for_layer

    # ── Helpers ──────────────────────────────────────────────────────────
    def _run_scalar(self, input_ids, alpha=ALPHA, layer=LAYER_IDX,
                    seed=BASE_SEED, return_intermediates=True):
        bf = self._get_bridge_forward()
        return bf(
            self.model, self.tok,
            input_ids=input_ids,
            layer_idx=layer,
            alpha=alpha,
            backend='exact',
            K=K,
            seed=seed,
            return_intermediates=return_intermediates,
        )

    def _run_list(self, input_ids, layers, alpha=ALPHA,
                  seed=BASE_SEED, return_intermediates=True,
                  backend='exact'):
        bf = self._get_bridge_forward()
        return bf(
            self.model, self.tok,
            input_ids=input_ids,
            layer_idx=layers,
            alpha=alpha,
            backend=backend,
            K=K,
            seed=seed,
            return_intermediates=return_intermediates,
        )

    def _vanilla_logits(self, input_ids):
        """Run an unpatched forward and return logits."""
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits.detach().clone()

    # ── Test 1: scalar path bit-exact against M5_FROZEN ─────────────────
    def test_01_scalar_m5_frozen_bitexact(self):
        """Req 1: scalar bridge_forward produces identical output to
        pre-change version. Proxy: alpha=0 must be bit-exact vanilla,
        and alpha=0.3 result must be finite and differ from vanilla.
        Full M5_FROZEN CSV diff is a separate offline check."""
        print("\n  Test 01: scalar path sanity (M5_FROZEN proxy)")
        for i, ids in enumerate(self.input_ids_list):
            result = self._run_scalar(ids, alpha=0.0)
            self.assertTrue(
                torch.equal(result.logits, result.vanilla_logits),
                f"prompt {i}: alpha=0 scalar path not bit-exact vanilla")
            result_03 = self._run_scalar(ids, alpha=0.3)
            diff = (result_03.logits.float() -
                    result_03.vanilla_logits.float()).abs()
            self.assertGreater(
                diff.max().item(), 1e-4,
                f"prompt {i}: alpha=0.3 scalar produced near-vanilla output")
            self.assertFalse(
                torch.isnan(result_03.logits).any(),
                f"prompt {i}: NaN in alpha=0.3 scalar output")
        print("    PASS: scalar path alpha=0 bit-exact, alpha=0.3 finite+differs")

    # ── Test 2: single-element list == scalar ────────────────────────────
    def test_02_single_element_list_equals_scalar(self):
        """Req 2: bridge_forward(layer_idx=[18]) == bridge_forward(layer_idx=18)."""
        print("\n  Test 02: single-element list equals scalar")
        for i, ids in enumerate(self.input_ids_list):
            scalar_logits = self._run_scalar(
                ids, alpha=ALPHA, return_intermediates=False)
            list_logits = self._run_list(
                ids, layers=[LAYER_IDX], alpha=ALPHA,
                return_intermediates=False)
            self.assertTrue(
                torch.equal(scalar_logits, list_logits),
                f"prompt {i}: [18] list != scalar 18 logits")
        print("    PASS: single-element list is bit-exact scalar")

    # ── Test 3: list alpha=0 scalar == vanilla ───────────────────────────
    def test_03_list_alpha_zero_scalar_equals_vanilla(self):
        """Req 3: bridge_forward(layer_idx=[18,24], alpha=0.0) == vanilla."""
        print("\n  Test 03: list alpha=0.0 (scalar) equals vanilla")
        for i, ids in enumerate(self.input_ids_list):
            vanilla = self._vanilla_logits(ids)
            result = self._run_list(
                ids, layers=[LAYER_IDX, LAYER_IDX2], alpha=0.0)
            diff = (result.logits.float() - vanilla.float()).abs()
            self.assertEqual(
                diff.max().item(), 0.0,
                f"prompt {i}: list alpha=0 scalar max_abs_diff="
                f"{diff.max().item():.2e} (expected 0.00e+00)")
        print("    PASS: list alpha=0.0 (scalar) is bit-exact vanilla")

    # ── Test 4: list alpha=0 list == vanilla ─────────────────────────────
    def test_04_list_alpha_zero_list_equals_vanilla(self):
        """Req 4: bridge_forward(layer_idx=[18,24], alpha=[0.0,0.0]) == vanilla."""
        print("\n  Test 04: list alpha=[0.0, 0.0] equals vanilla")
        for i, ids in enumerate(self.input_ids_list):
            vanilla = self._vanilla_logits(ids)
            result = self._run_list(
                ids, layers=[LAYER_IDX, LAYER_IDX2], alpha=[0.0, 0.0])
            diff = (result.logits.float() - vanilla.float()).abs()
            self.assertEqual(
                diff.max().item(), 0.0,
                f"prompt {i}: list alpha=[0,0] max_abs_diff="
                f"{diff.max().item():.2e} (expected 0.00e+00)")
        print("    PASS: list alpha=[0.0, 0.0] is bit-exact vanilla")

    # ── Test 5: scalar alpha broadcast == list alpha uniform (2-layer) ───
    def test_05_scalar_alpha_broadcast_equals_list_alpha_uniform_2layer(self):
        """Req 5: alpha=0.3 scalar == alpha=[0.3,0.3] list at 2 layers."""
        print("\n  Test 05: scalar alpha broadcast == list alpha uniform (2-layer)")
        for i, ids in enumerate(self.input_ids_list):
            scalar_broadcast = self._run_list(
                ids, layers=[LAYER_IDX, LAYER_IDX2],
                alpha=0.3, return_intermediates=False)
            list_uniform = self._run_list(
                ids, layers=[LAYER_IDX, LAYER_IDX2],
                alpha=[0.3, 0.3], return_intermediates=False)
            self.assertTrue(
                torch.equal(scalar_broadcast, list_uniform),
                f"prompt {i}: scalar broadcast != list uniform at 2 layers")
        print("    PASS: scalar alpha broadcasts correctly at 2 layers")

    # ── Test 6: scalar alpha broadcast == list alpha uniform (3-layer) ───
    def test_06_scalar_alpha_broadcast_equals_list_alpha_uniform_3layer(self):
        """Req 6: alpha=0.3 scalar == alpha=[0.3,0.3,0.3] list at 3 layers."""
        print("\n  Test 06: scalar alpha broadcast == list alpha uniform (3-layer)")
        layers_3 = [15, LAYER_IDX, LAYER_IDX2]
        for i, ids in enumerate(self.input_ids_list):
            scalar_broadcast = self._run_list(
                ids, layers=layers_3,
                alpha=0.3, return_intermediates=False)
            list_uniform = self._run_list(
                ids, layers=layers_3,
                alpha=[0.3, 0.3, 0.3], return_intermediates=False)
            self.assertTrue(
                torch.equal(scalar_broadcast, list_uniform),
                f"prompt {i}: scalar broadcast != list uniform at 3 layers")
        print("    PASS: scalar alpha broadcasts correctly at 3 layers")

    # ── Test 7: unsorted input order independence ────────────────────────
    def test_07_unsorted_input_order_independence(self):
        """Req 7: [24,18]/[0.5,0.3] == [18,24]/[0.3,0.5]."""
        print("\n  Test 07: unsorted input order independence")
        for i, ids in enumerate(self.input_ids_list):
            fwd = self._run_list(
                ids, layers=[LAYER_IDX2, LAYER_IDX],
                alpha=[0.5, 0.3], return_intermediates=False)
            rev = self._run_list(
                ids, layers=[LAYER_IDX, LAYER_IDX2],
                alpha=[0.3, 0.5], return_intermediates=False)
            self.assertTrue(
                torch.equal(fwd, rev),
                f"prompt {i}: [24,18]/[0.5,0.3] != [18,24]/[0.3,0.5]")
        print("    PASS: output is order-independent")

    # ── Test 8: duplicate layer raises ValueError ────────────────────────
    def test_08_duplicate_layer_raises(self):
        """Req 8: bridge_forward(layer_idx=[18,18]) raises ValueError."""
        print("\n  Test 08: duplicate layer raises ValueError")
        bf = self._get_bridge_forward()
        ids = self.input_ids_list[0]
        with self.assertRaises(ValueError) as ctx:
            bf(self.model, self.tok, input_ids=ids,
               layer_idx=[LAYER_IDX, LAYER_IDX],
               alpha=0.3, backend='exact', K=K, seed=BASE_SEED)
        self.assertIn(str(LAYER_IDX), str(ctx.exception))
        print(f"    PASS: raised ValueError: {ctx.exception}")

    # ── Test 9: alpha length mismatch raises ValueError ──────────────────
    def test_09_alpha_length_mismatch_raises(self):
        """Req 9: layer_idx=[18,24], alpha=[0.3] raises ValueError."""
        print("\n  Test 09: alpha length mismatch raises ValueError")
        bf = self._get_bridge_forward()
        ids = self.input_ids_list[0]
        with self.assertRaises(ValueError) as ctx:
            bf(self.model, self.tok, input_ids=ids,
               layer_idx=[LAYER_IDX, LAYER_IDX2],
               alpha=[0.3], backend='exact', K=K, seed=BASE_SEED)
        msg = str(ctx.exception)
        self.assertIn("1", msg)   # alpha length
        self.assertIn("2", msg)   # layer_idx length
        print(f"    PASS: raised ValueError: {ctx.exception}")

    # ── Test 10: per_layer_seeds stable and unique ───────────────────────
    def test_10_per_layer_seeds_stable_and_unique(self):
        """Req 10: same base_seed -> same per_layer_seeds; layers get distinct seeds."""
        print("\n  Test 10: per_layer_seeds stable and unique")
        ids = self.input_ids_list[0]

        result_a = self._run_list(
            ids, layers=[LAYER_IDX, LAYER_IDX2], alpha=ALPHA)
        result_b = self._run_list(
            ids, layers=[LAYER_IDX, LAYER_IDX2], alpha=ALPHA)

        # Stability: same call twice -> same seeds
        self.assertEqual(
            result_a.per_layer_seeds, result_b.per_layer_seeds,
            "per_layer_seeds not stable across identical calls")

        # Uniqueness: different layers -> different seeds
        seed_18 = result_a.per_layer_seeds[LAYER_IDX]
        seed_24 = result_a.per_layer_seeds[LAYER_IDX2]
        self.assertIsNotNone(seed_18, "seed for L18 is None")
        self.assertIsNotNone(seed_24, "seed for L24 is None")
        self.assertNotEqual(
            seed_18, seed_24,
            f"L18 and L24 got identical seeds ({seed_18})")

        print(f"    PASS: L{LAYER_IDX} seed={seed_18}, "
              f"L{LAYER_IDX2} seed={seed_24}, stable across calls")

    # ── Test 11: return_intermediates dict field contract ────────────────
    def test_11_return_intermediates_dict_fields_contract(self):
        """Req 11: list call with return_intermediates=True returns correct
        BridgeResult field contract. Scalar fields are None; dict fields
        populated with correct keys."""
        print("\n  Test 11: return_intermediates dict fields contract")
        from thermobridge import LayerCapture

        ids = self.input_ids_list[0]
        layers = [LAYER_IDX, LAYER_IDX2]
        result = self._run_list(ids, layers=layers, alpha=ALPHA)

        # Scalar fields must be None on list call
        self.assertIsNone(result.capture,
            "result.capture should be None on list call")
        self.assertIsNone(result.p_thermo,
            "result.p_thermo should be None on list call")
        self.assertIsNone(result.alpha,
            "result.alpha should be None on list call")
        self.assertIsNone(result.layer_idx,
            "result.layer_idx should be None on list call")

        # Dict fields must be populated with correct keys
        expected_keys = set(layers)
        for field_name, field_val in [
            ('layer_captures',    result.layer_captures),
            ('p_thermo_by_layer', result.p_thermo_by_layer),
            ('alpha_by_layer',    result.alpha_by_layer),
            ('per_layer_seeds',   result.per_layer_seeds),
        ]:
            self.assertIsNotNone(field_val,
                f"result.{field_name} is None on list call")
            self.assertEqual(
                set(field_val.keys()), expected_keys,
                f"result.{field_name} keys {set(field_val.keys())} "
                f"!= expected {expected_keys}")

        # Type checks
        for li in layers:
            self.assertIsInstance(
                result.layer_captures[li], LayerCapture,
                f"layer_captures[{li}] is not LayerCapture")
            self.assertIsInstance(
                result.p_thermo_by_layer[li], torch.Tensor,
                f"p_thermo_by_layer[{li}] is not Tensor")
            self.assertIsInstance(
                result.alpha_by_layer[li], float,
                f"alpha_by_layer[{li}] is not float")
            self.assertIsInstance(
                result.per_layer_seeds[li], int,
                f"per_layer_seeds[{li}] is not int")

        # Keys must match across all four dicts
        keys_list = [
            set(result.layer_captures.keys()),
            set(result.p_thermo_by_layer.keys()),
            set(result.alpha_by_layer.keys()),
            set(result.per_layer_seeds.keys()),
        ]
        self.assertEqual(
            len(set(frozenset(k) for k in keys_list)), 1,
            f"Dict field keys are inconsistent: {keys_list}")

        print(f"    PASS: all dict fields populated with keys {expected_keys}, "
              f"scalar fields None")

    # ── Test 12: M5_FROZEN bit-exact scalar regression ───────────────────
    def test_12_m5_frozen_scalar_regression(self):
        """Req 12: belt-and-suspenders scalar alpha=0 identity across all
        prompts. Complements test_01; ensures no regression crept in."""
        print("\n  Test 12: M5_FROZEN scalar regression (alpha=0 identity)")
        for i, ids in enumerate(self.input_ids_list):
            result = self._run_scalar(ids, alpha=0.0)
            max_diff = (result.logits.float() -
                        result.vanilla_logits.float()).abs().max().item()
            self.assertEqual(
                max_diff, 0.0,
                f"prompt {i}: scalar alpha=0 max_abs_diff={max_diff:.2e} "
                f"(expected 0.00e+00)")
        print("    PASS: scalar alpha=0 is bit-exact vanilla on all prompts")

    # ── Test 13: multi-layer gumbel/rbm supported; thrml blocked ─────────
    def test_13_multilayer_gumbel_rbm_supported_thrml_blocked(self):
        """Req 13 (updated, Bug #2): multi-layer now supports gumbel/rbm via
        scoped per-layer RNG isolation. thrml multi-layer stays blocked."""
        print("\n  Test 13: multi-layer gumbel/rbm supported; thrml blocked")
        ids = self.input_ids_list[0]
        layers = [LAYER_IDX, LAYER_IDX2]
        vanilla = self._vanilla_logits(ids)
        for be in ('gumbel', 'rbm'):
            logits = self._run_list(ids, layers=layers, alpha=ALPHA,
                                    backend=be, return_intermediates=False)
            self.assertEqual(logits.shape, vanilla.shape,
                             f"{be} multi-layer logits shape mismatch")
            self.assertFalse(torch.isnan(logits).any(),
                             f"{be} multi-layer produced NaN")
            print(f"    {be}: ran multi-layer, finite logits {tuple(logits.shape)}")
        with self.assertRaises(ValueError) as ctx:
            self._run_list(ids, layers=layers, alpha=ALPHA, backend='thrml')
        self.assertIn('thrml', str(ctx.exception))
        print(f"    thrml correctly blocked: {ctx.exception}")
        print("    PASS")

    # ── Test 14: scalar BridgeResult scalar fields populated ─────────────
    def test_14_scalar_bridgeresult_scalar_fields_populated(self):
        """Req 14: scalar call populates scalar fields AND single-entry
        dict fields. No field should be None on a scalar call."""
        print("\n  Test 14: scalar BridgeResult field contract")
        from thermobridge import LayerCapture

        ids = self.input_ids_list[0]
        result = self._run_scalar(ids, alpha=ALPHA)

        # Scalar fields must be populated
        self.assertIsNotNone(result.capture,   "result.capture is None on scalar call")
        self.assertIsNotNone(result.p_thermo,  "result.p_thermo is None on scalar call")
        self.assertIsNotNone(result.alpha,     "result.alpha is None on scalar call")
        self.assertIsNotNone(result.layer_idx, "result.layer_idx is None on scalar call")
        self.assertEqual(result.layer_idx, LAYER_IDX)
        self.assertEqual(result.alpha, ALPHA)
        self.assertIsInstance(result.capture, LayerCapture)

        # Dict fields must be single-entry dicts with correct key
        for field_name, field_val in [
            ('layer_captures',    result.layer_captures),
            ('p_thermo_by_layer', result.p_thermo_by_layer),
            ('alpha_by_layer',    result.alpha_by_layer),
            ('per_layer_seeds',   result.per_layer_seeds),
        ]:
            self.assertIsNotNone(field_val,
                f"result.{field_name} is None on scalar call")
            self.assertEqual(
                list(field_val.keys()), [LAYER_IDX],
                f"result.{field_name} keys {list(field_val.keys())} "
                f"!= [{LAYER_IDX}] on scalar call")

        print(f"    PASS: scalar fields populated, dict fields are "
              f"single-entry {{18: ...}}")

    # ── Test 15: per_layer_seeds order independent ───────────────────────
    def test_15_per_layer_seeds_order_independent(self):
        """Req 15: per_layer_seeds is identical for [24,18] and [18,24]
        input orderings. Guards against enumerate-indexing footgun."""
        print("\n  Test 15: per_layer_seeds order independence")
        ids = self.input_ids_list[0]

        result_fwd = self._run_list(
            ids, layers=[LAYER_IDX2, LAYER_IDX], alpha=ALPHA)
        result_rev = self._run_list(
            ids, layers=[LAYER_IDX, LAYER_IDX2], alpha=ALPHA)

        self.assertEqual(
            result_fwd.per_layer_seeds,
            result_rev.per_layer_seeds,
            f"per_layer_seeds differ by input order:\n"
            f"  [24,18]: {result_fwd.per_layer_seeds}\n"
            f"  [18,24]: {result_rev.per_layer_seeds}")

        print(f"    PASS: per_layer_seeds identical regardless of input order "
              f"{result_fwd.per_layer_seeds}")

    # ── Test 16: multi-layer gumbel/rbm isolation + reproducibility ──────
    def test_16_multilayer_gumbel_rbm_isolation(self):
        """Bug #2 coverage (beyond the original 15-req spec): multi-layer
        gumbel/rbm are reproducible (same seed -> identical), per-layer seeds
        are distinct, and alpha=0 is bit-exact vanilla -- the RNG isolation
        that unblocks open-loop gumbel/rbm composition."""
        print("\n  Test 16: multi-layer gumbel/rbm isolation + reproducibility")
        ids = self.input_ids_list[0]
        layers = [LAYER_IDX, LAYER_IDX2]
        vanilla = self._vanilla_logits(ids)
        for be in ('gumbel', 'rbm'):
            r0 = self._run_list(ids, layers=layers, alpha=0.0, backend=be)
            max_diff = (r0.logits.float() - vanilla.float()).abs().max().item()
            self.assertEqual(max_diff, 0.0,
                f"{be} multi-layer alpha=0 not bit-exact vanilla (max_diff={max_diff:.2e})")
            a = self._run_list(ids, layers=layers, alpha=ALPHA, backend=be)
            b = self._run_list(ids, layers=layers, alpha=ALPHA, backend=be)
            self.assertTrue(torch.equal(a.logits, b.logits),
                f"{be} multi-layer not reproducible across identical calls")
            s1 = a.per_layer_seeds[LAYER_IDX]
            s2 = a.per_layer_seeds[LAYER_IDX2]
            self.assertNotEqual(s1, s2, f"{be} per-layer seeds identical ({s1})")
            self.assertFalse(torch.isnan(a.logits).any(), f"{be} produced NaN")
            print(f"    {be}: alpha=0 bit-exact, reproducible, seeds distinct ({s1}, {s2})")
        print("    PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Verbose output so every test's print() is visible alongside pass/fail
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # preserve definition order
    suite = loader.loadTestsFromTestCase(TestMultilayerRefactor)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
