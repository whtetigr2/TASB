"""
capture.py — RoPE-aware per-Q-head attention capture for LLaMA models
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver. USPTO Provisional 64/019,999.

During a forward pass on a frozen LLaMA model, this module captures the
canonical attention state at one or more designated layers:

  * q_post_rope    — post-RoPE query  (B, n_q_heads,  S, head_dim)
  * k_post_rope    — post-RoPE key    (B, n_kv_heads, S, head_dim)
  * attention_mask — live HF mask     (B, 1, S, S)
  * scaling        — HF scaling factor (typically 1/√d_k)
  * attn_weights   — post-softmax weights (verify_capture only)

DESIGN
------
1. LlamaAttention.forward is not modified. Only apply_rotary_pos_emb and
   eager_attention_forward are patched at the module level.

2. Layer identification via call-stack owner-discovery: when the patched
   function fires, walk the stack to find the LlamaAttention instance that
   owns the call. Robust to call ordering changes.

3. Signature-adaptive patching: kwargs are filtered through the original
   function's signature at runtime, ensuring cross-version compatibility.

4. Context-manager interface. Patch install, teardown, and registry restore
   all happen via __enter__/__exit__. Patches cannot leak across forwards.

5. Capture invariant checked in strict mode:
       softmax(Q_post @ repeat_kv(K_post).T * scaling + mask) ≈ attn_weights
   Checked at every captured step. Failure raises immediately — invalid
   science fails loud, not silently.

6. Per-layer subset: only layers in layers_to_capture are captured.

USAGE
-----
    from thermobridge.capture import LlamaAttentionCapture

    capturer = LlamaAttentionCapture(model, layers_to_capture=[18])
    with capturer.capture():
        out = model(input_ids)
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


# Snapshot original references once at import time
_ORIG_APPLY_ROTARY = _llama_mod.apply_rotary_pos_emb
_ORIG_EAGER = getattr(_llama_mod, 'eager_attention_forward', None)

_ROTARY_SIG = inspect.signature(_ORIG_APPLY_ROTARY)
_ROTARY_PARAMS = set(_ROTARY_SIG.parameters.keys())
if _ORIG_EAGER is not None:
    _EAGER_SIG = inspect.signature(_ORIG_EAGER)
    _EAGER_PARAMS = set(_EAGER_SIG.parameters.keys())
else:
    _EAGER_SIG = None
    _EAGER_PARAMS = set()


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
    """Per-layer captured state from one forward pass.

    All tensors are on the device they were captured on.
    """
    layer_idx: int
    q_post_rope: torch.Tensor     # (B, n_q,  S, head_dim)
    k_post_rope: torch.Tensor     # (B, n_kv, S, head_dim)
    attention_mask: torch.Tensor | None  # (B, 1, S, S) or None
    scaling: float
    attn_weights: torch.Tensor    # (B, n_q, S, S)
    seq_len: int
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
    """RoPE-aware per-Q-head attention capture for LLaMA models.

    Patches transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    and eager_attention_forward at the module level. LlamaAttention.forward
    is not modified.

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
        self._rope_scratch: dict[int, dict[str, torch.Tensor]] = {}
        self._eager_scratch: dict[int, dict[str, Any]] = {}
        self.captured: dict[int, LayerCapture] = {}
        self._installed = False
        self._registry_obj = None
        self._registry_orig = None
        self.n_rope_calls = 0
        self.n_eager_calls = 0
        self.n_captured_steps = 0

    @staticmethod
    def _find_owning_layer_idx() -> int | None:
        """Walk the call stack to find the LlamaAttention that owns this call."""
        frame = inspect.currentframe()
        if frame is None:
            return None
        try:
            frame = frame.f_back
            depth_limit = 16
            while frame is not None and depth_limit > 0:
                local_self = frame.f_locals.get('self')
                if isinstance(local_self, _LlamaAttention):
                    return getattr(local_self, 'layer_idx', None)
                frame = frame.f_back
                depth_limit -= 1
        finally:
            del frame
        return None

    @staticmethod
    def _filter_kwargs(kwargs: dict, allowed: set[str]) -> dict:
        """Return only kwargs accepted by the target function signature."""
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _make_patched_rope(self):
        orig = _ORIG_APPLY_ROTARY
        allowed = _ROTARY_PARAMS
        layer_set = self.layer_set
        scratch = self._rope_scratch
        log = self.config.log_calls

        def patched(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            q_out, k_out = orig(*args, **filtered_kwargs)

            layer_idx = LlamaAttentionCapture._find_owning_layer_idx()
            self.n_rope_calls += 1
            if log:
                print(f"  [rope #{self.n_rope_calls}] layer_idx={layer_idx}")

            # Fail loud if the stack-walk breaks — a silent drop would produce
            # incorrect captures without any indication.
            if layer_idx is None:
                raise RuntimeError(
                    "LlamaAttentionCapture: _find_owning_layer_idx() returned "
                    f"None on rope call #{self.n_rope_calls}. The HF call stack "
                    "changed; post-RoPE Q/K capture cannot be trusted. "
                    "Re-validate the capture patch against the installed "
                    "transformers version.")

            if layer_idx in layer_set:
                if len(args) >= 4:
                    q_pre, k_pre, cos, sin = args[0], args[1], args[2], args[3]
                    scratch[layer_idx] = {
                        'q_pre_rope':  q_pre.detach().clone(),
                        'k_pre_rope':  k_pre.detach().clone(),
                        'cos':         cos.detach().clone(),
                        'sin':         sin.detach().clone(),
                        'q_post_rope': q_out.detach().clone(),
                        'k_post_rope': k_out.detach().clone(),
                    }
                else:
                    raise RuntimeError(
                        f"LlamaAttentionCapture: target layer {layer_idx} rope "
                        f"call had {len(args)} positional args (<4); cannot "
                        "capture post-RoPE Q/K. HF signature changed.")

            return q_out, k_out

        return patched

    def _make_patched_eager(self):
        if _ORIG_EAGER is None:
            return None

        orig = _ORIG_EAGER
        allowed = _EAGER_PARAMS
        layer_set = self.layer_set
        eager_scratch = self._eager_scratch
        log = self.config.log_calls

        def patched(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            attn_output, attn_weights = orig(*args, **filtered_kwargs)

            layer_idx = LlamaAttentionCapture._find_owning_layer_idx()
            self.n_eager_calls += 1
            if log:
                print(f"  [eager #{self.n_eager_calls}] layer_idx={layer_idx}")

            if layer_idx is None:
                raise RuntimeError(
                    "LlamaAttentionCapture: _find_owning_layer_idx() returned "
                    f"None on eager call #{self.n_eager_calls}. The HF call "
                    "stack changed — capture cannot be trusted.")

            if layer_idx in layer_set:
                attention_mask = (args[4] if len(args) > 4
                                  else kwargs.get('attention_mask'))
                scaling_val = kwargs.get('scaling', None)
                if scaling_val is None and len(args) > 5:
                    scaling_val = args[5] if isinstance(args[5], float) else None
                eager_scratch[layer_idx] = {
                    'attention_mask': (attention_mask.detach().clone()
                                        if attention_mask is not None else None),
                    'scaling': (float(scaling_val) if scaling_val is not None
                                else None),
                    'attn_weights': attn_weights.detach().clone(),
                }

            return attn_output, attn_weights

        return patched

    def _verify_one_layer(self, layer_idx: int,
                          cap: LayerCapture) -> tuple[bool, dict]:
        """Reconstruct softmax(Q@K.T * scale + mask) and compare to attn_weights."""
        Q = cap.q_post_rope
        K = cap.k_post_rope

        n_q = Q.shape[1]
        n_kv = K.shape[1]
        if n_q % n_kv != 0:
            return False, {'error': f'n_q={n_q} not divisible by n_kv={n_kv}'}
        kv_groups = n_q // n_kv

        K_rep = _repeat_kv(K, kv_groups)
        scores = torch.matmul(Q, K_rep.transpose(-2, -1)) * cap.scaling

        # Additive mask (HF uses large-negative sentinel, not masked_fill)
        if cap.attention_mask is not None:
            scores = scores + cap.attention_mask

        recon = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32).to(Q.dtype)

        diff = (recon.float() - cap.attn_weights.float()).abs()
        diff_np = diff.detach().cpu().numpy()

        if cap.attention_mask is not None:
            valid = (cap.attention_mask > -1e30).detach().cpu().numpy()
            valid = np.broadcast_to(valid, diff_np.shape)
        else:
            S = cap.seq_len
            causal = np.triu(np.ones((S, S), dtype=bool), 1)
            valid = np.broadcast_to(~causal, diff_np.shape)

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

    def _finalize(self):
        """Merge rope and eager scratch into LayerCapture objects."""
        for layer_idx in self.config.layers_to_capture:
            if layer_idx not in self._rope_scratch:
                continue
            if layer_idx not in self._eager_scratch:
                continue

            rs = self._rope_scratch[layer_idx]
            es = self._eager_scratch[layer_idx]

            cap = LayerCapture(
                layer_idx=layer_idx,
                q_post_rope=rs['q_post_rope'],
                k_post_rope=rs['k_post_rope'],
                attention_mask=es['attention_mask'],
                scaling=es['scaling'],
                attn_weights=es['attn_weights'],
                seq_len=rs['q_post_rope'].shape[-2],
                dtype=rs['q_post_rope'].dtype,
            )
            self.captured[layer_idx] = cap
            self.n_captured_steps += 1

    @contextmanager
    def capture(self):
        """Install patches, yield, restore patches unconditionally in finally."""
        if self._installed:
            raise RuntimeError(
                "capture() is not re-entrant. Use a new LlamaAttentionCapture "
                "instance for each forward.")

        self._rope_scratch.clear()
        self._eager_scratch.clear()
        self.captured.clear()
        self.n_rope_calls = 0
        self.n_eager_calls = 0
        self.n_captured_steps = 0

        patched_rope = self._make_patched_rope()
        patched_eager = self._make_patched_eager()

        _llama_mod.apply_rotary_pos_emb = patched_rope
        if patched_eager is not None and _ORIG_EAGER is not None:
            _llama_mod.eager_attention_forward = patched_eager

        self._registry_obj, self._registry_orig = _get_attn_registry_eager()
        if self._registry_obj is not None and patched_eager is not None:
            self._registry_obj['eager'] = patched_eager

        self._installed = True

        try:
            yield self
        finally:
            _llama_mod.apply_rotary_pos_emb = _ORIG_APPLY_ROTARY
            if _ORIG_EAGER is not None:
                _llama_mod.eager_attention_forward = _ORIG_EAGER
            if self._registry_obj is not None and self._registry_orig is not None:
                self._registry_obj['eager'] = self._registry_orig
            self._installed = False
            self._finalize()
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
            f"  rope calls observed:  {self.n_rope_calls}",
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
    print(f"  apply_rotary_pos_emb params: {sorted(_ROTARY_PARAMS)}")
    if _ORIG_EAGER is not None:
        print(f"  eager_attention_forward params: {sorted(_EAGER_PARAMS)}")
    else:
        print("  eager_attention_forward: not found (older transformers?)")
    reg, _ = _get_attn_registry_eager()
    print(f"  ALL_ATTENTION_FUNCTIONS: {'present' if reg else 'absent'}")
    print("  Module loads cleanly.")
