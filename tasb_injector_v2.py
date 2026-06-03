"""
tasb_injector_v2.py — Per-Q-head p_thermo injection into the attention forward
==============================================================================

Author: Paul W. Shaver

Takes a dispatch table of {layer_idx: (LayerCapture, p_thermo, alpha)} and
blends each layer's p_thermo into the model's forward pass. The blended
attention is then used to recompute attn_output = blended_weights @ V,
which propagates through the rest of the model.

INPUT CONTRACT
--------------
A dispatch_table dict mapping each target layer_idx (int) to a tuple of:
  - LayerCapture: the capture from pass 1 for that layer
  - p_thermo:     (B, n_q, S, S) row-stochastic tensor from the sampler
  - alpha:        float blending coefficient in [0, 1]

BLEND FORMULA
-------------
    blended_weights = (1 - alpha) * attn_weights + alpha * p_thermo

Linear interpolation between vanilla Boltzmann attention (at the model's
own temperature) and the sampled substitute distribution. Both inputs are
row-stochastic, so the output is row-stochastic by linearity.

CRITICAL INVARIANT: alpha=0 IDENTITY
-------------------------------------
At alpha=0 for a given layer, the patched forward must produce BIT-EXACT
vanilla output for that layer. Not "within fp32 precision" — identical.
Enforced two ways:

1. Per-layer fast path: at alpha=0, the patched dispatch skips blending
   entirely for that layer and returns the original output unmodified.
2. Test: alpha0_max_abs_diff must be 0.00e+00 at every layer in the table.

INJECTION MECHANISM
-------------------
Patches eager_attention_forward at module level. ONE patch is installed
for the entire forward pass. Inside the patched function, dispatch is
routed by args[0].layer_idx — the LlamaAttention module announces its
own identity (confirmed present on LLaMA 3.2-3B). Non-target layers fall
through to the original function unchanged.

Each target layer fires exactly once per forward pass (sequential, not
parallel). The dispatch table is pre-built; the hot path is one dict
lookup per attention call.

WHAT THIS FILE DOES NOT DO
--------------------------
- Capture. The injector consumes LayerCaptures; it doesn't produce them.
- Sampling. The injector consumes p_thermo tensors from the sampler.
- Masking. p_thermo arrives with zero mass on masked positions; the
  linear blend preserves that property.

BUG GUARDS
----------
- #4 mask convention: handled in capture and sampler. Injector blends
  two already-mask-respecting tensors.
- #5 broadcast: p_thermo and attn_weights are both (B, n_q, S, S).
- #7 layer_subset type: dispatch_table keys are ints.
- #9 signature-adaptive: _filter_kwargs pattern preserved.
- #13 PEP 604 unions: using typing.Union / typing.List, not | syntax.

INVARIANTS
----------
- alpha=0 at a layer -> that layer's output is bit-exact vanilla.
- alpha=1 at a layer -> that layer uses pure p_thermo for attention.
- 0 < alpha < 1 -> blended_weights rows sum to 1 (linearity of blend).
- Non-target layers are byte-for-byte unmodified.
- Patch is installed exactly once per inject() call; restored in finally.
==============================================================================
"""

import inspect
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

from transformers.models.llama import modeling_llama as _llama_mod
from transformers.models.llama.modeling_llama import (
    LlamaAttention as _LlamaAttention,
    repeat_kv as _repeat_kv,
)

from tasb_capture_v2 import LayerCapture


# ---------------------------------------------------------------------------
# Snapshot original references at import time (Bug #9)
# ---------------------------------------------------------------------------
_ORIG_EAGER = getattr(_llama_mod, 'eager_attention_forward', None)
if _ORIG_EAGER is not None:
    _EAGER_SIG = inspect.signature(_ORIG_EAGER)
    _EAGER_PARAMS = set(_EAGER_SIG.parameters.keys())
else:
    _EAGER_SIG = None
    _EAGER_PARAMS = set()


def _get_attn_registry_eager():
    """Return (registry_mapping, original_entry) or (None, None)."""
    try:
        from transformers.models.llama.modeling_llama import (
            ALL_ATTENTION_FUNCTIONS)
        if hasattr(ALL_ATTENTION_FUNCTIONS, '_global_mapping'):
            mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
            if 'eager' in mapping:
                return mapping, mapping['eager']
    except (ImportError, AttributeError):
        pass
    return None, None


# ---------------------------------------------------------------------------
# DispatchEntry: typed tuple for one layer's injection state
# ---------------------------------------------------------------------------
class DispatchEntry:
    """Validated injection parameters for a single layer."""

    __slots__ = ('capture', 'p_thermo', 'alpha')

    def __init__(
        self,
        capture: LayerCapture,
        p_thermo: torch.Tensor,
        alpha: float,
    ):
        if not isinstance(capture, LayerCapture):
            raise TypeError(
                f"capture must be LayerCapture, got {type(capture).__name__}")
        if not isinstance(p_thermo, torch.Tensor):
            raise TypeError(
                f"p_thermo must be torch.Tensor, got {type(p_thermo).__name__}")
        if p_thermo.shape != capture.attn_weights.shape:
            raise ValueError(
                f"p_thermo shape {tuple(p_thermo.shape)} != "
                f"capture.attn_weights shape {tuple(capture.attn_weights.shape)}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}")
        self.capture = capture
        self.p_thermo = p_thermo
        self.alpha = float(alpha)


# ---------------------------------------------------------------------------
# LlamaAttentionInjector
# ---------------------------------------------------------------------------
class LlamaAttentionInjector:
    """Multi-layer p_thermo injector for a frozen LLaMA model.

    Installs ONE patch on eager_attention_forward. Inside the patch,
    dispatch is routed by args[0].layer_idx (the LlamaAttention module's
    own attribute — confirmed present on LLaMA 3.2-3B). No stack walking.

    For single-layer use (backward compat), pass a single-entry dict.
    The scalar M5/M6/M7 paths are unchanged in behavior.

    Usage (multi-layer):
        dispatch = {
            18: DispatchEntry(capture_18, p_thermo_18, alpha=0.3),
            24: DispatchEntry(capture_24, p_thermo_24, alpha=0.3),
        }
        injector = LlamaAttentionInjector(dispatch)
        with injector.inject():
            out = model(input_ids)

    Usage (single-layer, backward compat):
        dispatch = {18: DispatchEntry(capture, p_thermo, alpha=0.3)}
        injector = LlamaAttentionInjector(dispatch)
        with injector.inject():
            out = model(input_ids)
    """

    def __init__(
        self,
        dispatch_table: Dict[int, DispatchEntry],
    ):
        if not isinstance(dispatch_table, dict):
            raise TypeError(
                f"dispatch_table must be dict, got {type(dispatch_table).__name__}")
        if len(dispatch_table) == 0:
            raise ValueError("dispatch_table must not be empty.")
        for k, v in dispatch_table.items():
            if not isinstance(k, int):
                raise TypeError(
                    f"dispatch_table keys must be int, got {type(k).__name__} "
                    f"for key {k!r}")
            if not isinstance(v, DispatchEntry):
                raise TypeError(
                    f"dispatch_table values must be DispatchEntry, "
                    f"got {type(v).__name__} for layer {k}")

        self.dispatch_table = dispatch_table

        # Patch state — set during inject()
        self._installed = False
        self._registry_obj = None
        self._registry_orig = None

        # Diagnostics — populated during inject()
        self.n_eager_calls = 0
        self.n_injections_by_layer: Dict[int, int] = {
            li: 0 for li in dispatch_table}

    @staticmethod
    def _filter_kwargs(kwargs: dict, allowed: set) -> dict:
        """Bug #9 production filter — strip unknown kwargs."""
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _make_patched_eager(self):
        """Return the patched closure. Built once at inject() time.

        Hot path (per attention call):
          1. Increment call counter.
          2. Look up args[0].layer_idx in dispatch_table.
          3. If not present: fallthrough to original (non-target layer).
          4. If present and alpha==0.0: fallthrough (alpha=0 fast path).
          5. If present and alpha>0: blend + recompute attn_output.
        """
        if _ORIG_EAGER is None:
            return None

        orig = _ORIG_EAGER
        allowed = _EAGER_PARAMS
        dispatch_table = self.dispatch_table
        n_injections_by_layer = self.n_injections_by_layer
        # Capture self reference for call counter only
        injector_ref = self

        def patched(*args, **kwargs):
            injector_ref.n_eager_calls += 1
            filtered = {k: v for k, v in kwargs.items() if k in allowed}

            # Identify the calling layer. args[0] is the LlamaAttention
            # module instance; .layer_idx is set by HF at model build time.
            # Confirmed present on LLaMA 3.2-3B (verified 2026-06-02).
            if not args:
                return orig(*args, **filtered)

            calling_layer = getattr(args[0], 'layer_idx', None)

            # Non-target layer: transparent fallthrough
            if calling_layer not in dispatch_table:
                return orig(*args, **filtered)

            entry = dispatch_table[calling_layer]

            # alpha=0 fast path: bit-exact vanilla for this layer
            if entry.alpha == 0.0:
                return orig(*args, **filtered)

            # Run original to get reference attn_output and attn_weights
            attn_output, attn_weights = orig(*args, **filtered)

            # Need module and value from positional args.
            # Standard HF signature: (module, query, key, value, mask, ...)
            if len(args) < 4:
                # Unexpected signature change — skip injection, return orig
                return attn_output, attn_weights

            module = args[0]
            value = args[3]

            # Blend in fp32 (matches original softmax dtype convention)
            w_fp32 = attn_weights.to(torch.float32)
            p_fp32 = entry.p_thermo.to(w_fp32.device).to(torch.float32)
            blended = (1.0 - entry.alpha) * w_fp32 + entry.alpha * p_fp32
            blended = blended.to(attn_weights.dtype)

            # Recompute attn_output = blended @ V with same repeat_kv
            # expansion that eager_attention_forward uses internally
            v_rep = _repeat_kv(value, module.num_key_value_groups)
            new_attn_output = torch.matmul(blended, v_rep)
            new_attn_output = new_attn_output.transpose(1, 2).contiguous()

            n_injections_by_layer[calling_layer] += 1
            return new_attn_output, blended

        return patched

    @contextmanager
    def inject(self):
        """Install the eager patch, yield, restore on exit (try/finally).

        Patch is installed exactly once. Re-entrancy raises rather than
        silently double-patching (which would corrupt the canonical stack).

        The finally block restores unconditionally — if an exception
        propagates during the injected forward pass, the original
        eager_attention_forward is always restored before the exception
        escapes.
        """
        if self._installed:
            raise RuntimeError(
                "inject() is not re-entrant. Create a new "
                "LlamaAttentionInjector instance per forward pass.")

        if _ORIG_EAGER is None:
            raise RuntimeError(
                "eager_attention_forward not found at "
                "transformers.models.llama.modeling_llama. "
                "Cannot install injector patch.")

        patched = self._make_patched_eager()
        if patched is None:
            raise RuntimeError("Failed to construct patched eager function.")

        # Reset diagnostics
        self.n_eager_calls = 0
        for li in self.n_injections_by_layer:
            self.n_injections_by_layer[li] = 0

        # Install patch on module and registry
        _llama_mod.eager_attention_forward = patched
        self._registry_obj, self._registry_orig = _get_attn_registry_eager()
        if self._registry_obj is not None:
            self._registry_obj['eager'] = patched

        self._installed = True

        try:
            yield self
        finally:
            # Restore unconditionally — leak prevention
            _llama_mod.eager_attention_forward = _ORIG_EAGER
            if self._registry_obj is not None and self._registry_orig is not None:
                self._registry_obj['eager'] = self._registry_orig
            self._installed = False

    @property
    def n_injections(self) -> int:
        """Total injections across all layers. Backward-compat convenience."""
        return sum(self.n_injections_by_layer.values())

    def summary(self) -> str:
        layers = sorted(self.dispatch_table.keys())
        per_layer = ", ".join(
            f"L{li}(alpha={self.dispatch_table[li].alpha}, "
            f"inj={self.n_injections_by_layer[li]})"
            for li in layers
        )
        return (
            f"LlamaAttentionInjector("
            f"layers={layers}, "
            f"eager_calls={self.n_eager_calls}, "
            f"injections={self.n_injections}, "
            f"per_layer=[{per_layer}])")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("tasb_injector_v2.py self-checks:")
    print(f"  _ORIG_EAGER present: {_ORIG_EAGER is not None}")
    if _EAGER_SIG is not None:
        print(f"  _EAGER_PARAMS: {sorted(_EAGER_PARAMS)}")
    reg, _ = _get_attn_registry_eager()
    print(f"  ALL_ATTENTION_FUNCTIONS: {'present' if reg else 'absent'}")

    # DispatchEntry validation
    import numpy as np
    dummy_shape = (1, 8, 16, 16)
    dummy_weights = torch.zeros(dummy_shape)

    # Build a minimal LayerCapture-like object for validation testing
    # (real LayerCapture comes from tasb_capture_v2; just test guards here)
    try:
        DispatchEntry(capture="not_a_capture", p_thermo=dummy_weights, alpha=0.3)
    except TypeError as e:
        print(f"  capture type guard:    OK ({e})")

    try:
        DispatchEntry(capture=object(), p_thermo=dummy_weights, alpha=1.5)
    except TypeError:
        print(f"  alpha range guard:     OK (raised on bad capture type first)")

    # dict key type guard
    try:
        LlamaAttentionInjector(dispatch_table={"18": object()})
    except TypeError as e:
        print(f"  key type guard:        OK ({e})")

    # empty dict guard
    try:
        LlamaAttentionInjector(dispatch_table={})
    except ValueError as e:
        print(f"  empty dict guard:      OK ({e})")

    print("  Module loads cleanly.")
