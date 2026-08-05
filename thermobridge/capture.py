"""
capture.py — RoPE-aware per-Q-head attention capture for LLaMA models
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver.

During a forward pass on a frozen LLaMA model, this module captures the
canonical attention state at one or more designated layers:

  * q_post_rope    — post-RoPE query  (B, n_q_heads,  Sq, head_dim)
  * k_post_rope    — post-RoPE key    (B, n_kv_heads, Sk, head_dim)
  * attention_mask — live HF mask     (B, 1, Sq, Sk)
  * scaling        — HF scaling factor (typically 1/√d_k)
  * attn_weights   — post-softmax weights (verify_capture only)

Sq == Sk during full-sequence reprocessing (prefill, or any use_cache=False
call). Sq == 1 < Sk during a single-token KV-cached decode step — one new
query row attending to the full cached key history. Nothing downstream
should assume Sq == Sk.

REWRITE (2026-07-04) — KV-cache correctness
--------------------------------------------
The original version patched TWO functions: `apply_rotary_pos_emb` (to grab
post-RoPE Q/K) and `eager_attention_forward` (to grab attention_mask/scaling/
attn_weights), merging the two scratch dicts after the forward pass, with
layer identification done by walking the Python call stack for the owning
LlamaAttention instance.

That design breaks under KV-cache decoding: `apply_rotary_pos_emb` fires on
only the NEWLY COMPUTED query/key for the current step (shape
(B, n, 1, head_dim) once a KV cache is in use) — but `eager_attention_forward`'s
`key`/`value` arguments are the FULL, already-cache-concatenated sequence
(HF's `LlamaAttention.forward` calls `past_key_value.update(...)` to get the
full K/V *before* calling the attention interface function). Capturing K from
the RoPE patch under caching would silently capture only the newest 1-token
slice instead of the full history the bridge actually needs — a correctness
bug, not merely an inefficiency. This was found and fixed in the
`Active_Dev/TASB` working copy (`tasb_capture_v2.py`) on 2026-07-03 as part of
root-causing a live-chat OOM (the OOM fix itself required `use_cache=True`,
which is exactly the mode this bug lived in); this file is the same fix
ported into the real installed package, which had not been touched.

Fix: capture Q and K directly from `eager_attention_forward`'s own
`query`/`key` arguments instead. Both are already post-RoPE (RoPE is always
applied before the attention interface is called) and already reflect
whatever the model actually attended over — correct in both full
reprocessing (Sq==Sk) and under KV-cache decoding (Sq=1, Sk=S_total). This
also eliminates the separate RoPE patch, the two-scratch-dict merge, and the
stack-walk-based layer lookup entirely — `eager_attention_forward`'s own
first positional argument, `module`, IS the owning LlamaAttention instance,
with `.layer_idx` set by HF at model-build time (the same `module.layer_idx`
lookup `inject.py` already used, correctly, from day one).

DESIGN
------
1. LlamaAttention.forward is not modified. Only eager_attention_forward is
   patched at the module level.

2. Layer identification via `module.layer_idx` (module is
   eager_attention_forward's own first argument — no stack walking).

3. Signature-adaptive patching: kwargs are filtered through the original
   function's signature at runtime, ensuring cross-version compatibility.

4. Context-manager interface. Patch install, teardown, and registry restore
   all happen via __enter__/__exit__. Patches cannot leak across forwards.

5. Capture invariant checked in strict mode:
       softmax(Q_post @ repeat_kv(K_post).T * scaling + mask) ≈ attn_weights
   Checked at every captured step, handling both Sq==Sk and Sq==1<Sk.
   Failure raises immediately — invalid science fails loud, not silently.

6. Per-layer subset: only layers in layers_to_capture are captured.

USAGE
-----
    from thermobridge.capture import LlamaAttentionCapture

    capturer = LlamaAttentionCapture(model, layers_to_capture=[18])
    with capturer.capture():
        out = model(input_ids)  # or model(new_token, past_key_values=pkv, use_cache=True)
    captured = capturer.get_capture(18)

MASK CONVENTION
---------------
Captured attention_mask is HF's actual mask (~+0 / ~-3.4e38 in bf16).
Downstream consumers must apply it as an ADDITION to logits in logit space,
not as masked_fill in probability space.
==============================================================================
"""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from transformers.models.llama import modeling_llama as _llama_mod
from transformers.models.llama.modeling_llama import (
    LlamaAttention as _LlamaAttention,
    repeat_kv as _repeat_kv,
)


# Snapshot original reference at import time
_ORIG_EAGER = getattr(_llama_mod, 'eager_attention_forward', None)
if _ORIG_EAGER is None:
    raise ImportError(
        "thermobridge.capture: transformers.models.llama.modeling_llama."
        "eager_attention_forward not found. This capture module requires "
        "an HF transformers version that exposes eager_attention_forward "
        "at module level (attn_implementation='eager'). Re-validate against "
        "the installed transformers version before use.")

_EAGER_SIG = inspect.signature(_ORIG_EAGER)
_EAGER_PARAMS = set(_EAGER_SIG.parameters.keys())


def _get_attn_registry_eager():
    """Return (registry_obj, original_entry_or_None)."""
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerCapture:
    """Per-layer captured state from one forward pass (or one decode step).

    All tensors are on the device they were captured on. Shapes:
      q_post_rope:    (B, n_q,  Sq, head_dim)
      k_post_rope:    (B, n_kv, Sk, head_dim)
      attention_mask: (B, 1, Sq, Sk) or None
      attn_weights:   (B, n_q, Sq, Sk)
    """
    layer_idx: int
    q_post_rope: torch.Tensor     # (B, n_q,  Sq, head_dim)
    k_post_rope: torch.Tensor     # (B, n_kv, Sk, head_dim)
    attention_mask: torch.Tensor | None  # (B, 1, Sq, Sk) or None
    scaling: float
    attn_weights: torch.Tensor    # (B, n_q, Sq, Sk)
    seq_len: int                  # key-side context length (Sk); use
                                   # q_post_rope.shape[-2] if you need Sq
    dtype: torch.dtype

    def to_numpy(self) -> dict[str, Any]:
        """Convert to numpy for analysis."""
        d = {
            'layer_idx': self.layer_idx,
            'q_post_rope': self.q_post_rope.detach().float().cpu().numpy(),
            'k_post_rope': self.k_post_rope.detach().float().cpu().numpy(),
            'scaling': self.scaling,
            'attn_weights': self.attn_weights.detach().float().cpu().numpy(),
            'seq_len': self.seq_len,
            'dtype': str(self.dtype),
        }
        if self.attention_mask is not None:
            d['attention_mask'] = (
                self.attention_mask.detach().float().cpu().numpy())
        else:
            d['attention_mask'] = None
        return d


@dataclass
class CaptureConfig:
    layers_to_capture: list[int]
    strict_verify: bool = True
    strict_atol: float = 5e-4
    strict_mean_atol: float = 1e-4
    raise_on_verify_fail: bool = True
    log_calls: bool = False


# ---------------------------------------------------------------------------
# LlamaAttentionCapture
# ---------------------------------------------------------------------------

class LlamaAttentionCapture:
    """RoPE-aware per-Q-head attention capture for LLaMA models, correct
    under both full-sequence reprocessing and KV-cache decoding.

    Patches transformers.models.llama.modeling_llama.eager_attention_forward
    at the module level. LlamaAttention.forward is not modified.

    Use as a context manager:

        capturer = LlamaAttentionCapture(model, layers_to_capture=[18])
        with capturer.capture():
            out = model(input_ids)
        captured = capturer.get_capture(18)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers_to_capture: list[int],
        strict_verify: bool = True,
        strict_atol: float = 5e-4,
        strict_mean_atol: float = 1e-4,
        raise_on_verify_fail: bool = True,
        log_calls: bool = False,
    ):
        if not isinstance(layers_to_capture, list):
            raise TypeError(
                f"layers_to_capture must be list[int], "
                f"got {type(layers_to_capture)}: {layers_to_capture}")
        if not all(isinstance(L, int) for L in layers_to_capture):
            raise TypeError(
                f"layers_to_capture entries must be int. Got: {layers_to_capture}")

        self.model = model
        self.config = CaptureConfig(
            layers_to_capture=list(layers_to_capture),
            strict_verify=strict_verify,
            strict_atol=strict_atol,
            strict_mean_atol=strict_mean_atol,
            raise_on_verify_fail=raise_on_verify_fail,
            log_calls=log_calls,
        )
        self.layer_set = set(self.config.layers_to_capture)
        self.captured: dict[int, LayerCapture] = {}
        self._installed = False
        self._registry_obj = None
        self._registry_orig = None
        self.n_eager_calls = 0
        self.n_captured_steps = 0

    @staticmethod
    def _filter_kwargs(kwargs: dict, allowed: set[str]) -> dict:
        """Return only kwargs accepted by the target function signature."""
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _make_patched_eager(self):
        """Build a closure that captures Q/K/mask/scaling/attn_weights
        directly from this call's own arguments — correct whether this call
        is a full-sequence prefill (Sq==Sk) or a single-token KV-cached
        decode step (Sq=1 < Sk)."""
        orig = _ORIG_EAGER
        allowed = _EAGER_PARAMS
        layer_set = self.layer_set
        captured = self.captured
        log = self.config.log_calls

        def patched(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            attn_output, attn_weights = orig(*args, **filtered_kwargs)

            self.n_eager_calls += 1

            if not args:
                return attn_output, attn_weights
            module = args[0]
            layer_idx = getattr(module, 'layer_idx', None)

            if log:
                print(f"  [eager #{self.n_eager_calls}] layer_idx={layer_idx}")

            if layer_idx is None:
                raise RuntimeError(
                    "LlamaAttentionCapture: eager_attention_forward's module "
                    f"argument has no layer_idx on call #{self.n_eager_calls}. "
                    "HF internals changed in a way that breaks layer "
                    "identification. Failing loud instead of silently "
                    "skipping capture.")

            if layer_idx in layer_set:
                if len(args) < 3:
                    raise RuntimeError(
                        f"LlamaAttentionCapture: target layer {layer_idx} "
                        f"eager_attention_forward call had {len(args)} "
                        "positional args (<3); cannot read query/key to "
                        "capture. HF signature changed.")
                query = args[1]
                key   = args[2]
                attention_mask = (args[4] if len(args) > 4
                                  else kwargs.get('attention_mask'))
                scaling_val = (args[5] if len(args) > 5
                               else kwargs.get('scaling'))
                if scaling_val is None:
                    raise RuntimeError(
                        f"LlamaAttentionCapture: target layer {layer_idx} "
                        "eager_attention_forward call had no resolvable "
                        "`scaling` value. Cannot capture without it.")

                captured[layer_idx] = LayerCapture(
                    layer_idx=layer_idx,
                    q_post_rope=query.detach().clone(),
                    k_post_rope=key.detach().clone(),
                    attention_mask=(attention_mask.detach().clone()
                                     if attention_mask is not None else None),
                    scaling=float(scaling_val),
                    attn_weights=attn_weights.detach().clone(),
                    seq_len=key.shape[-2],
                    dtype=query.dtype,
                )
                self.n_captured_steps += 1

            return attn_output, attn_weights

        return patched

    def _verify_one_layer(self, layer_idx: int,
                          cap: LayerCapture) -> tuple[bool, dict]:
        """Reconstruct softmax(Q@K.T * scale + mask) and compare to
        attn_weights. Handles both Sq==Sk (prefill) and Sq==1<Sk (cached
        decode) shapes."""
        Q = cap.q_post_rope
        K = cap.k_post_rope

        Sq = Q.shape[-2]
        Sk = K.shape[-2]

        n_q = Q.shape[1]
        n_kv = K.shape[1]
        if n_q % n_kv != 0:
            return False, {'error': f'n_q={n_q} not divisible by n_kv={n_kv}'}
        kv_groups = n_q // n_kv

        K_rep = _repeat_kv(K, kv_groups)
        scores = torch.matmul(Q, K_rep.transpose(-2, -1)) * cap.scaling

        if cap.attention_mask is not None:
            scores = scores + cap.attention_mask

        recon = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32).to(Q.dtype)

        diff = (recon.float() - cap.attn_weights.float()).abs()
        diff_np = diff.detach().cpu().numpy()

        # Build valid-position mask. Three cases:
        #  - A real attention_mask is present: use it directly (correct for
        #    any Sq/Sk shape).
        #  - No mask, Sq == Sk: standard full-sequence causal mask.
        #  - No mask, Sq == 1 < Sk: a single newly-appended query token,
        #    causally valid against every cached key position — no masking
        #    needed (HF supplies an all-zero mask for this case anyway).
        if cap.attention_mask is not None:
            valid = (cap.attention_mask > -1e30).detach().cpu().numpy()
            valid = np.broadcast_to(valid, diff_np.shape)
        elif Sq == Sk:
            causal = np.triu(np.ones((Sq, Sk), dtype=bool), 1)
            valid = np.broadcast_to(~causal, diff_np.shape)
        else:
            valid_2d = np.ones((Sq, Sk), dtype=bool)
            valid = np.broadcast_to(valid_2d, diff_np.shape)

        valid_diffs = diff_np[valid]
        max_diff = float(valid_diffs.max())
        mean_diff = float(valid_diffs.mean())

        ok = (max_diff < self.config.strict_atol
              and mean_diff < self.config.strict_mean_atol)
        return ok, {
            'layer_idx': layer_idx,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'atol': self.config.strict_atol,
            'mean_atol': self.config.strict_mean_atol,
            'shape': tuple(diff.shape),
        }

    def verify_capture(self) -> dict:
        """Run the capture invariant on every captured layer."""
        results = {}
        failed = []
        for layer_idx, cap in self.captured.items():
            ok, diag = self._verify_one_layer(layer_idx, cap)
            results[layer_idx] = {'ok': ok, **diag}
            if not ok:
                failed.append(layer_idx)

        if failed and self.config.raise_on_verify_fail:
            summary = "\n".join(
                f"  L{L}: max={results[L]['max_diff']:.2e} "
                f"(tol={results[L]['atol']:.0e}), "
                f"mean={results[L]['mean_diff']:.2e} "
                f"(tol={results[L]['mean_atol']:.0e})"
                for L in failed
            )
            raise RuntimeError(
                f"Capture invariant failed on {len(failed)} layer(s):\n"
                f"{summary}\n"
                "The captured state does not reproduce HF's attn_weights via "
                "softmax(Q@K.T * scale + mask). See capture.py _verify_one_layer.")

        return results

    @contextmanager
    def capture(self):
        """Install patch, yield, restore patch unconditionally in finally."""
        if self._installed:
            raise RuntimeError(
                "capture() is not re-entrant. Use a new LlamaAttentionCapture "
                "instance for each forward.")

        self.captured.clear()
        self.n_eager_calls = 0
        self.n_captured_steps = 0

        patched_eager = self._make_patched_eager()
        _llama_mod.eager_attention_forward = patched_eager

        self._registry_obj, self._registry_orig = _get_attn_registry_eager()
        if self._registry_obj is not None:
            self._registry_obj['eager'] = patched_eager

        self._installed = True

        try:
            yield self
        finally:
            _llama_mod.eager_attention_forward = _ORIG_EAGER
            if self._registry_obj is not None and self._registry_orig is not None:
                self._registry_obj['eager'] = self._registry_orig
            self._installed = False
            if self.config.strict_verify and self.captured:
                self.verify_capture()

    def get_captured(self) -> dict[int, LayerCapture]:
        """Return captured tensors keyed by layer_idx."""
        return dict(self.captured)

    def get_capture(self, layer_idx: int) -> LayerCapture:
        """Return capture for a single layer; raises KeyError if absent."""
        if layer_idx not in self.captured:
            raise KeyError(
                f"Layer {layer_idx} not in captured set "
                f"{list(self.captured.keys())}.")
        return self.captured[layer_idx]

    def summary(self) -> str:
        lines = [
            "LlamaAttentionCapture summary:",
            f"  layers_to_capture: {self.config.layers_to_capture}",
            f"  eager calls observed: {self.n_eager_calls}",
            f"  captured layers:      {list(self.captured.keys())}",
            f"  strict_verify:        {self.config.strict_verify}",
            f"  strict_atol:          {self.config.strict_atol}",
        ]
        for L, cap in self.captured.items():
            lines.append(
                f"  L{L}: Q{tuple(cap.q_post_rope.shape)} "
                f"K{tuple(cap.k_post_rope.shape)} "
                f"mask={'yes' if cap.attention_mask is not None else 'no'} "
                f"scaling={cap.scaling:.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("capture.py self-checks:")
    print(f"  eager_attention_forward params: {sorted(_EAGER_PARAMS)}")
    reg, _ = _get_attn_registry_eager()
    print(f"  ALL_ATTENTION_FUNCTIONS: {'present' if reg else 'absent'}")
    print("  Module loads cleanly.")
