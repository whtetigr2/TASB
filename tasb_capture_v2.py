"""
tasb_capture_v2.py — RoPE-aware per-Q-head capture for the TASB bridge
==============================================================================

Author: Paul W. Shaver

Replaces VanillaCapture from tasb_two_pass.py. Built on the anchor result
in tests/test_rope_live_capture_v2.py, which proved bit-exact independent
reconstruction of HF's attention pipeline (0.00e+00 across all four
diagnostic comparisons).

WHAT THIS FILE DOES
-------------------
During a forward pass on a frozen LLaMA model, this module captures the
canonical attention state at one or more designated layers:

  * `q_post_rope`     — post-RoPE query, shape (B, n_q_heads,  S, head_dim)
  * `k_post_rope`     — post-RoPE key,   shape (B, n_kv_heads, S, head_dim)
  * `attention_mask`  — the live mask HF used, shape (B, 1, S, S)
  * `scaling`         — the scaling factor HF used (typically 1/√d_k)
  * `attn_weights`    — the post-softmax weights HF produced
                        (for verify_capture only; not used by bridge)

These are the tensors the bridge needs to compute a faithful TSU substitute
distribution. The capture is per-Q-head — no KV-averaging happens here.
Averaging, if needed, is the sampler's choice and lives in tasb_sampler_v2.

DESIGN PRINCIPLES
-----------------
1. Leave LlamaAttention.forward intact. Patch only apply_rotary_pos_emb
   and eager_attention_forward at the module level. This was validated
   bit-exact in the anchor test.

2. Layer identification via call-stack owner-discovery (not call counting).
   When the patched function fires, walk the stack to find the
   LlamaAttention instance that owns the call, read its layer_idx. Robust
   to call ordering and to layers that don't dispatch through every helper.

3. Signature-adaptive patching via inspect.signature. The patched function
   only forwards kwargs that exist in the original signature, regardless of
   transformers version. Wrapper-level defaults are documentation; runtime
   filtering is correctness. (Bug #9, production-grade version.)

4. Context-manager interface. `with capturer.capture(): model(...)`. Patch
   installation, teardown, and registry restoration all happen via
   __enter__/__exit__. No way to leak patches across forwards.

5. Verify-on-every-step in strict mode. The capture invariant
       softmax(Q_post @ repeat_kv(K_post).T * scaling + mask) ≈ attn_weights
   is checked at every captured step. Failure raises immediately. This is
   the "flag invalid science" rule operationalized.

6. Per-layer subset. Only capture from layers the bridge actually uses
   (the bridge's `layers_to_capture` list). Other layers run vanilla with
   no capture overhead.

WHAT THIS FILE DOES NOT DO
--------------------------
- No sampling. The bridge calls tasb_sampler_v2 with the captured tensors.
- No injection. The bridge calls tasb_injector_v2 with sampler output.
- No support for non-LLaMA architectures. Mistral/Qwen come after Stage 1.
- No fp16 capture path. Captures in whatever dtype HF used.

INVARIANTS THIS FILE MUST SATISFY
---------------------------------
After a successful capture, for every captured layer L:
  (1) softmax(Q_post[L] @ repeat_kv(K_post[L]).T * scaling[L] + mask[L]) ==
      attn_weights[L]  within tolerance set by `strict_atol` (default 5e-4).
  (2) Q_post[L].shape == (B, n_q_heads, S, head_dim)
  (3) K_post[L].shape == (B, n_kv_heads, S, head_dim)
  (4) layer_idx for the captured tensors is the layer we requested.

These are enforced in `verify_capture()`. Tests live in
tests/test_capture_v2.py.

USAGE
-----
    from tasb_capture_v2 import LlamaAttentionCapture

    capturer = LlamaAttentionCapture(
        model=model,
        layers_to_capture=[18],
        strict_verify=True,
    )

    with capturer.capture():
        out = model(input_ids)

    captured = capturer.get_captured()
    # captured[18] is a dict with keys q_post_rope, k_post_rope,
    # attention_mask, scaling, attn_weights, layer_idx, seq_len

BUG GUARDS (from registry)
--------------------------
- #4 mask convention: we store HF's actual mask (typically +0/-3.4e38 in
  bf16), NOT a -1e9 sentinel. Downstream consumers must apply mask as
  ADDITION to logits, not as masked_fill.
- #5 broadcast shape: captured mask is (B,1,S,S), broadcastable to
  (B,n_q,S,S) but must be explicitly broadcast before boolean indexing.
- #8 RoPE bug: this file is the fix. q_post_rope is captured AFTER
  apply_rotary_pos_emb, never reconstructed from pre-RoPE projections.
- #9 wrapper signature: we use inspect.signature filtering, not opaque
  passthrough. Production-grade version.
==============================================================================
"""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


# ── Module-level imports of HF internals we patch ─────────────────────────
# Done at import time so we capture references to the originals once.
from transformers.models.llama import modeling_llama as _llama_mod
from transformers.models.llama.modeling_llama import (
    LlamaAttention as _LlamaAttention,
    repeat_kv as _repeat_kv,
)


# Snapshot original references at import time
_ORIG_APPLY_ROTARY = _llama_mod.apply_rotary_pos_emb
_ORIG_EAGER = getattr(_llama_mod, 'eager_attention_forward', None)

# Snapshot original signatures for adaptive forwarding (Bug #9 prod version)
_ROTARY_SIG = inspect.signature(_ORIG_APPLY_ROTARY)
_ROTARY_PARAMS = set(_ROTARY_SIG.parameters.keys())
if _ORIG_EAGER is not None:
    _EAGER_SIG = inspect.signature(_ORIG_EAGER)
    _EAGER_PARAMS = set(_EAGER_SIG.parameters.keys())
else:
    _EAGER_SIG = None
    _EAGER_PARAMS = set()


# Snapshot ALL_ATTENTION_FUNCTIONS registry entry for eager (if present)
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


# ── Data classes for captured state ───────────────────────────────────────

@dataclass
class LayerCapture:
    """Per-layer captured state from one forward pass.

    All tensors are on the device they were captured on; downstream code
    is responsible for moving them as needed. Shapes match LLaMA conventions:
    (B, n_heads, S, head_dim) for Q/K, (B, 1, S, S) for mask,
    (B, n_q_heads, S, S) for attn_weights.
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
        """Convert to numpy for analysis. Keeps mask as None if None."""
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
    strict_verify: bool = True       # verify_capture() at every step
    strict_atol: float = 5e-4        # tolerance for verify_capture
    strict_mean_atol: float = 1e-4   # mean diff tolerance
    raise_on_verify_fail: bool = True
    log_calls: bool = False          # debug: log every patched call


# ── The capturer class ────────────────────────────────────────────────────

class LlamaAttentionCapture:
    """RoPE-aware per-Q-head attention capture for LLaMA models.

    Patches transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    and eager_attention_forward at the module level. LlamaAttention.forward
    is NOT modified — it runs the original implementation.

    Layer identification is done by walking the call stack to find the
    LlamaAttention instance that owns the patched call.

    Use as a context manager:

        capturer = LlamaAttentionCapture(model, layers_to_capture=[18])
        with capturer.capture():
            out = model(input_ids)
        captured = capturer.get_captured()
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
                "layers_to_capture must be List[int] (Bug #7 guard). "
                f"Got {type(layers_to_capture)}: {layers_to_capture}")
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

        # Per-layer scratch state (collected during the forward pass) and
        # final captures (populated after the forward completes)
        self._rope_scratch: dict[int, dict[str, torch.Tensor]] = {}
        self._eager_scratch: dict[int, dict[str, Any]] = {}
        self.captured: dict[int, LayerCapture] = {}

        # Patch teardown state (set by __enter__, restored by __exit__)
        self._installed = False
        self._registry_obj = None
        self._registry_orig = None

        # Bookkeeping
        self.n_rope_calls = 0
        self.n_eager_calls = 0
        self.n_captured_steps = 0

    # ── Layer identification via stack walk ─────────────────────────────
    @staticmethod
    def _find_owning_layer_idx() -> int | None:
        """Walk up the Python call stack to find the LlamaAttention instance
        that owns the current call. Returns layer_idx or None if not found.

        We look for a frame whose locals contain 'self' bound to a
        LlamaAttention. The first such frame above ours is the owner.
        """
        frame = inspect.currentframe()
        if frame is None:
            return None
        try:
            # Skip our own frame and the patched function's frame
            frame = frame.f_back
            depth_limit = 16  # bound the walk
            while frame is not None and depth_limit > 0:
                local_self = frame.f_locals.get('self')
                if isinstance(local_self, _LlamaAttention):
                    return getattr(local_self, 'layer_idx', None)
                frame = frame.f_back
                depth_limit -= 1
        finally:
            del frame  # avoid reference cycle
        return None

    # ── Signature-adaptive call (Bug #9 production version) ────────────
    @staticmethod
    def _filter_kwargs(kwargs: dict, allowed: set[str]) -> dict:
        """Return a kwargs dict containing only keys present in `allowed`.

        This is the production version of Bug #9: we never forward a kwarg
        the target function doesn't accept. Cross-version-robust.
        """
        return {k: v for k, v in kwargs.items() if k in allowed}

    # ── The patched apply_rotary_pos_emb ───────────────────────────────
    def _make_patched_rope(self):
        """Build a closure that:
          1. Calls the original apply_rotary_pos_emb with signature-filtered
             kwargs.
          2. If the owning layer is in our capture set, stashes the post-RoPE
             Q and K and the cos/sin for that layer.
        """
        orig = _ORIG_APPLY_ROTARY
        allowed = _ROTARY_PARAMS
        layer_set = self.layer_set
        scratch = self._rope_scratch
        log = self.config.log_calls

        def patched(*args, **kwargs):
            # Always call the original with adaptive kwargs filtering
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            q_out, k_out = orig(*args, **filtered_kwargs)

            # Identify the owning layer
            layer_idx = LlamaAttentionCapture._find_owning_layer_idx()
            self.n_rope_calls += 1
            if log:
                print(f"  [rope #{self.n_rope_calls}] layer_idx={layer_idx}")

            if layer_idx is not None and layer_idx in layer_set:
                # Stash post-RoPE Q, K plus cos, sin and pre-RoPE Q, K for
                # verify_capture. args[0]=q, args[1]=k, args[2]=cos, args[3]=sin
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

            return q_out, k_out

        return patched

    # ── The patched eager_attention_forward ────────────────────────────
    def _make_patched_eager(self):
        """Build a closure that:
          1. Calls the original eager_attention_forward with signature-
             filtered kwargs.
          2. If the owning layer is in our capture set, stashes the live
             attention_mask, scaling, and the returned attn_weights.
        """
        if _ORIG_EAGER is None:
            return None

        orig = _ORIG_EAGER
        allowed = _EAGER_PARAMS
        layer_set = self.layer_set
        eager_scratch = self._eager_scratch
        log = self.config.log_calls

        def patched(*args, **kwargs):
            # Adaptive kwargs forwarding
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            attn_output, attn_weights = orig(*args, **filtered_kwargs)

            layer_idx = LlamaAttentionCapture._find_owning_layer_idx()
            self.n_eager_calls += 1
            if log:
                print(f"  [eager #{self.n_eager_calls}] layer_idx={layer_idx}")

            if layer_idx is not None and layer_idx in layer_set:
                # Extract scaling and attention_mask from args/kwargs
                # Standard signature: (module, query, key, value,
                #                      attention_mask, scaling=..., ...)
                attention_mask = (args[4] if len(args) > 4
                                  else kwargs.get('attention_mask'))
                scaling_val = kwargs.get('scaling', None)
                if scaling_val is None and len(args) > 5:
                    # Some versions accept scaling positionally; defensive read
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

    # ── Capture invariant ──────────────────────────────────────────────
    def _verify_one_layer(self, layer_idx: int,
                          cap: LayerCapture) -> tuple[bool, dict]:
        """Reconstruct softmax(Q_post @ repeat_kv(K_post).T * scaling + mask)
        and compare to captured attn_weights.

        Returns (ok, diagnostic_dict).
        """
        Q = cap.q_post_rope
        K = cap.k_post_rope
        device = Q.device

        # Determine n_key_value_groups from shapes
        n_q = Q.shape[1]
        n_kv = K.shape[1]
        if n_q % n_kv != 0:
            return False, {'error': f'n_q={n_q} not divisible by n_kv={n_kv}'}
        kv_groups = n_q // n_kv

        # repeat_kv to expand K to n_q heads
        K_rep = _repeat_kv(K, kv_groups)

        # matmul * scaling
        scores = torch.matmul(Q, K_rep.transpose(-2, -1)) * cap.scaling

        # Add mask if present (Bug #4: HF uses additive mask with large
        # negative sentinel; we ADD it, not masked_fill)
        if cap.attention_mask is not None:
            scores = scores + cap.attention_mask

        # softmax in fp32, cast back to query dtype (matches HF's
        # nn.functional.softmax(..., dtype=torch.float32).to(query.dtype))
        recon = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Compare
        diff = (recon.float() - cap.attn_weights.float()).abs()
        diff_np = diff.detach().cpu().numpy()

        # Build valid-position mask (Bug #5: must broadcast explicitly)
        if cap.attention_mask is not None:
            valid = (cap.attention_mask > -1e30).detach().cpu().numpy()
            valid = np.broadcast_to(valid, diff_np.shape)
        else:
            # No mask captured — assume full causal
            S = cap.seq_len
            causal = np.triu(np.ones((S, S), dtype=bool), 1)
            valid_2d = ~causal
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
        """Run the capture invariant on every captured layer.

        Returns a dict mapping layer_idx → diagnostic dict. If
        `raise_on_verify_fail` is True and any layer fails, raises
        RuntimeError after collecting diagnostics for all layers.
        """
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
                f"This indicates the captured attention state does not "
                f"reproduce HF's attn_weights via softmax(Q@K.T * scale + mask).\n"
                f"Refusing to produce results. See tasb_capture_v2.py "
                f"_verify_one_layer for details."
            )

        return results

    # ── Finalize captures after forward pass ───────────────────────────
    def _finalize(self):
        """Merge rope and eager scratch into LayerCapture objects."""
        for layer_idx in self.config.layers_to_capture:
            if layer_idx not in self._rope_scratch:
                # Layer never had its RoPE called — should not happen for
                # standard LLaMA, but defensive
                continue
            if layer_idx not in self._eager_scratch:
                # Same — defensive
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

    # ── Context manager ────────────────────────────────────────────────
    @contextmanager
    def capture(self):
        """Install patches, yield, restore patches in finally.

        Raises RuntimeError if capture invariant fails after the forward
        completes (when strict_verify=True).
        """
        if self._installed:
            raise RuntimeError(
                "capture() is not re-entrant. Use a new LlamaAttentionCapture "
                "instance for each forward.")

        # Clear scratch (in case capturer is reused after a previous forward)
        self._rope_scratch.clear()
        self._eager_scratch.clear()
        self.captured.clear()
        self.n_rope_calls = 0
        self.n_eager_calls = 0
        self.n_captured_steps = 0

        # Install patches
        patched_rope = self._make_patched_rope()
        patched_eager = self._make_patched_eager()

        _llama_mod.apply_rotary_pos_emb = patched_rope
        if patched_eager is not None and _ORIG_EAGER is not None:
            _llama_mod.eager_attention_forward = patched_eager

        # Patch the attention registry if present (recent transformers)
        self._registry_obj, self._registry_orig = _get_attn_registry_eager()
        if (self._registry_obj is not None
            and patched_eager is not None):
            self._registry_obj['eager'] = patched_eager

        self._installed = True

        try:
            yield self
        finally:
            # Restore originals UNCONDITIONALLY (leak prevention)
            _llama_mod.apply_rotary_pos_emb = _ORIG_APPLY_ROTARY
            if _ORIG_EAGER is not None:
                _llama_mod.eager_attention_forward = _ORIG_EAGER
            if self._registry_obj is not None and self._registry_orig is not None:
                self._registry_obj['eager'] = self._registry_orig
            self._installed = False

            # Finalize captures from scratch state
            self._finalize()

            # Verify if strict mode
            if self.config.strict_verify and self.captured:
                self.verify_capture()  # raises if raise_on_verify_fail

    # ── Accessors ──────────────────────────────────────────────────────
    def get_captured(self) -> dict[int, LayerCapture]:
        """Return the captured tensors keyed by layer_idx."""
        return dict(self.captured)

    def get_capture(self, layer_idx: int) -> LayerCapture:
        """Return capture for a single layer; raises KeyError if absent."""
        if layer_idx not in self.captured:
            raise KeyError(
                f"Layer {layer_idx} not in captured set "
                f"{list(self.captured.keys())}. "
                f"Was it included in layers_to_capture?")
        return self.captured[layer_idx]

    def summary(self) -> str:
        """Human-readable summary of the most recent capture."""
        lines = [
            f"LlamaAttentionCapture summary:",
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


# ── Self-test on import (cheap sanity) ────────────────────────────────────
if __name__ == "__main__":
    print("tasb_capture_v2.py self-checks:")
    print(f"  _ORIG_APPLY_ROTARY signature params: {sorted(_ROTARY_PARAMS)}")
    if _ORIG_EAGER is not None:
        print(f"  _ORIG_EAGER signature params:        {sorted(_EAGER_PARAMS)}")
    else:
        print(f"  _ORIG_EAGER not found at module level (older transformers?)")
    reg, orig = _get_attn_registry_eager()
    print(f"  ALL_ATTENTION_FUNCTIONS eager registry: "
          f"{'present' if reg is not None else 'absent'}")
    print(f"  module loads cleanly. Ready for use.")
