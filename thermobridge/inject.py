"""
inject.py — Per-Q-head p_thermo injection into attention forward
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver. USPTO Provisional 64/019,999.

Takes a dispatch table of {layer_idx: DispatchEntry} and blends each
layer's p_thermo into the model's forward pass.

BLEND FORMULA
-------------
    blended_weights = (1 - alpha) * attn_weights + alpha * p_thermo

Linear interpolation between vanilla attention and the sampled Boltzmann
distribution. Both inputs are row-stochastic, so the output is row-stochastic.

ALPHA=0 IDENTITY
----------------
At alpha=0 for a layer, the patched forward returns the original output
unmodified — bit-exact vanilla. This is enforced via a fast path: when
alpha==0.0, the dispatch skips blending and falls through to the original.

INJECTION MECHANISM
-------------------
Patches eager_attention_forward at module level. One patch is installed for
the entire forward pass. Inside the patch, dispatch is routed by
args[0].layer_idx — the LlamaAttention module announces its own identity.
Non-target layers fall through to the original unchanged.

The patch is restored unconditionally in the finally block of inject(),
even if an exception propagates during the forward pass.
==============================================================================
"""

import inspect
from contextlib import contextmanager
from typing import Dict, Optional

import torch

from transformers.models.llama import modeling_llama as _llama_mod
from transformers.models.llama.modeling_llama import (
    LlamaAttention as _LlamaAttention,
    repeat_kv as _repeat_kv,
)

from thermobridge.capture import LayerCapture


# Snapshot original references at import time for clean restore
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
# DispatchEntry
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
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.capture = capture
        self.p_thermo = p_thermo
        self.alpha = float(alpha)


# ---------------------------------------------------------------------------
# LlamaAttentionInjector
# ---------------------------------------------------------------------------
class LlamaAttentionInjector:
    """Multi-layer p_thermo injector for a frozen LLaMA model.

    Installs one patch on eager_attention_forward. Dispatch is routed by
    args[0].layer_idx (the LlamaAttention module's own attribute, present
    on LLaMA 3.2-3B). No stack walking — one dict lookup per attention call.

    Usage (multi-layer):
        dispatch = {
            18: DispatchEntry(capture_18, p_thermo_18, alpha=0.3),
            24: DispatchEntry(capture_24, p_thermo_24, alpha=0.3),
        }
        injector = LlamaAttentionInjector(dispatch)
        with injector.inject():
            out = model(input_ids)

    Usage (single-layer):
        dispatch = {18: DispatchEntry(capture, p_thermo, alpha=0.3)}
        injector = LlamaAttentionInjector(dispatch)
        with injector.inject():
            out = model(input_ids)
    """

    def __init__(self, dispatch_table: Dict[int, DispatchEntry]):
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
        self._installed = False
        self._registry_obj = None
        self._registry_orig = None
        self.n_eager_calls = 0
        self.n_injections_by_layer: Dict[int, int] = {
            li: 0 for li in dispatch_table}

    @staticmethod
    def _filter_kwargs(kwargs: dict, allowed: set) -> dict:
        """Strip kwargs not in the original function signature."""
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _make_patched_eager(self):
        """Return the patched closure. Built once at inject() time.

        Hot path per attention call:
          1. Increment call counter.
          2. Look up args[0].layer_idx in dispatch_table.
          3. Not present: fallthrough to original (non-target layer).
          4. Present, alpha==0.0: fallthrough (alpha=0 fast path).
          5. Present, alpha>0: blend p_thermo and recompute attn_output.
        """
        if _ORIG_EAGER is None:
            return None

        orig = _ORIG_EAGER
        allowed = _EAGER_PARAMS
        dispatch_table = self.dispatch_table
        n_injections_by_layer = self.n_injections_by_layer
        injector_ref = self

        def patched(*args, **kwargs):
            injector_ref.n_eager_calls += 1
            filtered = {k: v for k, v in kwargs.items() if k in allowed}

            if not args:
                return orig(*args, **filtered)

            calling_layer = getattr(args[0], 'layer_idx', None)

            if calling_layer not in dispatch_table:
                return orig(*args, **filtered)

            entry = dispatch_table[calling_layer]

            if entry.alpha == 0.0:
                return orig(*args, **filtered)

            # Target layer with alpha>0 — we need value=args[3] to recompute
            # the blended output. Fail loud if the HF signature changed.
            if len(args) < 4:
                raise RuntimeError(
                    f"LlamaAttentionInjector: target layer {calling_layer} "
                    f"eager_attention_forward had {len(args)} positional args (<4); "
                    "cannot read value=args[3] for blended output recompute. "
                    "The transformers signature changed — update the injector.")

            attn_output, attn_weights = orig(*args, **filtered)

            module = args[0]
            value = args[3]

            # Blend in fp32 to match softmax dtype convention
            w_fp32 = attn_weights.to(torch.float32)
            p_fp32 = entry.p_thermo.to(w_fp32.device).to(torch.float32)
            blended = (1.0 - entry.alpha) * w_fp32 + entry.alpha * p_fp32
            blended = blended.to(attn_weights.dtype)

            v_rep = _repeat_kv(value, module.num_key_value_groups)
            new_attn_output = torch.matmul(blended, v_rep)
            new_attn_output = new_attn_output.transpose(1, 2).contiguous()

            n_injections_by_layer[calling_layer] += 1
            return new_attn_output, blended

        return patched

    @contextmanager
    def inject(self):
        """Install the eager patch, yield, restore unconditionally on exit."""
        if self._installed:
            raise RuntimeError(
                "inject() is not re-entrant. Create a new "
                "LlamaAttentionInjector per forward pass.")

        if _ORIG_EAGER is None:
            raise RuntimeError(
                "eager_attention_forward not found at "
                "transformers.models.llama.modeling_llama.")

        patched = self._make_patched_eager()
        if patched is None:
            raise RuntimeError("Failed to construct patched eager function.")

        self.n_eager_calls = 0
        for li in self.n_injections_by_layer:
            self.n_injections_by_layer[li] = 0

        _llama_mod.eager_attention_forward = patched
        self._registry_obj, self._registry_orig = _get_attn_registry_eager()
        if self._registry_obj is not None:
            self._registry_obj['eager'] = patched

        self._installed = True

        try:
            yield self
        finally:
            _llama_mod.eager_attention_forward = _ORIG_EAGER
            if self._registry_obj is not None and self._registry_orig is not None:
                self._registry_obj['eager'] = self._registry_orig
            self._installed = False

    @property
    def n_injections(self) -> int:
        """Total injections across all layers."""
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
    print("inject.py self-checks:")
    print(f"  _ORIG_EAGER present: {_ORIG_EAGER is not None}")
    if _EAGER_SIG is not None:
        print(f"  _EAGER_PARAMS: {sorted(_EAGER_PARAMS)}")
    reg, _ = _get_attn_registry_eager()
    print(f"  ALL_ATTENTION_FUNCTIONS: {'present' if reg else 'absent'}")

    try:
        DispatchEntry(capture="not_a_capture", p_thermo=torch.zeros(1), alpha=0.3)
    except TypeError as e:
        print(f"  capture type guard:  OK ({e})")

    try:
        LlamaAttentionInjector(dispatch_table={"18": object()})
    except TypeError as e:
        print(f"  key type guard:      OK ({e})")

    try:
        LlamaAttentionInjector(dispatch_table={})
    except ValueError as e:
        print(f"  empty dict guard:    OK ({e})")

    print("  Module loads cleanly.")
