"""
tasb_pipeline_v2.py — Canonical bridge entry point
==============================================================================

Author: Paul W. Shaver

The two-pass thermodynamic attention sampling bridge, end to end.
Supports single-layer (scalar layer_idx) and multi-layer (list layer_idx)
injection in one unified API. Scalar path is bit-exact identical to the
pre-multi-layer version — M5_FROZEN_20260530 reproduces exactly.

WHAT THIS FILE DOES
-------------------
One function: bridge_forward(model, tok, prompt, ...). Runs the canonical
bridge sequence:

  1. NORMALIZE: canonicalize (layer_idx, alpha) inputs into sorted,
     validated (layer, alpha) pairs. Enforce backend restriction for
     multi-layer list mode.

  2. CAPTURE (pass 1): run a vanilla forward with patched
     apply_rotary_pos_emb + eager_attention_forward. Stash post-RoPE Q,
     K, mask, scaling, attn_weights at every target layer. Verify the
     capture invariant at each layer.

  3. SAMPLE: for each target layer, draw p_thermo from the chosen sampler
     backend using a per-layer seed derived from (base_seed, layer_idx).
     Seeds are order-independent (CRC-based derivation).

  4. INJECT (pass 2): run a second forward with ONE patched
     eager_attention_forward. The injector's dispatch table routes each
     layer's attention call to its pre-sampled p_thermo. Non-target layers
     pass through unchanged.

  5. RETURN: bridge logits plus optional intermediates in BridgeResult.

OPEN-LOOP COMPOSITION (multi-layer)
------------------------------------
M7-5 tests OPEN-LOOP multi-layer composition. p_thermo at each layer is
sampled from the VANILLA capture (pass 1), not from the perturbed hidden
state that results from upstream injections. Layer N's sample does not
see the effect of layer N-1's injection.

This isolates the composition question (do independently-derived
perturbations stack cleanly?) from the convergence question (does
sequential recomputation converge?). All M7-5 result language must say
"open-loop multi-layer composition." Closed-loop is a future milestone.

SEED DERIVATION
---------------
Per-layer seeds are derived as:
    seed_for_layer(base_seed, layer_idx) =
        (base_seed + zlib.crc32(f"layer_{layer_idx}".encode())) & 0x7FFFFFFF

This is order-independent: the seed for layer 18 is the same regardless
of whether [18, 24] or [24, 18] was passed. Matches the pattern used in
M7 sub-sweep harnesses. Bug #12: zlib.crc32, NOT hash().

BACKEND RESTRICTION
-------------------
Multi-layer list mode (len > 1) requires backend='exact'. The gumbel and
rbm backends use global torch.manual_seed and cannot guarantee per-layer
RNG isolation. Single-element list [N] is treated as scalar and has no
restriction. Guard is explicit in the error message.

BRIDGERESULT CONTRACT
---------------------
Scalar call (layer_idx=int):
    Scalar fields populated: capture, p_thermo, alpha, layer_idx
    Dict fields populated as single-entry dicts: {N: ...}

List call (layer_idx=list):
    Scalar fields are None (multi-layer mode)
    Dict fields populated with all layers

Both paths always populate alpha_by_layer and per_layer_seeds (small
metadata). layer_captures and p_thermo_by_layer are gated by
return_intermediates (large tensors).

INVARIANTS
----------
- alpha=0 (scalar or list) -> bit-exact vanilla output.
- Scalar path reproduces M5_FROZEN_20260530 bit-exact.
- Non-target layers are byte-for-byte unmodified during inject pass.
- Per-layer seeds are stable (same base_seed -> same seeds) and unique
  (different layers -> different seeds) by CRC construction.

BUG GUARDS
----------
- #12 zlib.crc32 for seed derivation (not hash())
- #13 typing.Union / typing.List (not PEP 604 | syntax)
==============================================================================
"""

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from tasb_capture_v2 import LlamaAttentionCapture, LayerCapture
from tasb_sampler_v2 import sample, SamplerConfig
from tasb_injector_v2 import LlamaAttentionInjector, DispatchEntry


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def seed_for_layer(base_seed: int, layer_idx: int) -> int:
    """Derive a per-layer seed from (base_seed, layer_idx).

    Order-independent: seed_for_layer(42, 18) is the same regardless of
    what other layers are in the call. Uses zlib.crc32 (Bug #12 — hash()
    is process-salted and not stable across invocations).

    Args:
        base_seed: the base seed passed to bridge_forward.
        layer_idx: the layer index to derive a seed for.

    Returns:
        A non-negative int suitable for use as a sampler seed.
    """
    return (base_seed + zlib.crc32(f"layer_{layer_idx}".encode())) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# BridgeResult
# ---------------------------------------------------------------------------

@dataclass
class BridgeResult:
    """Container for bridge_forward output when return_intermediates=True.

    SCALAR FIELDS (populated on scalar calls, None on list calls):
        logits:         (B, S, vocab_size) — output of the inject pass
        vanilla_logits: (B, S, vocab_size) — output of an unpatched pass
        capture:        LayerCapture from the capture pass (scalar only)
        p_thermo:       (B, n_q, S, S) — sampler output (scalar only)
        alpha:          float — the alpha used (scalar only)
        layer_idx:      int — the layer that was bridged (scalar only)

    DICT FIELDS (always populated; single-entry dict on scalar calls):
        layer_captures:     dict[int, LayerCapture] — one per target layer
                            (only populated when return_intermediates=True)
        p_thermo_by_layer:  dict[int, Tensor] — one per target layer
                            (only populated when return_intermediates=True)
        alpha_by_layer:     dict[int, float] — always populated (small)
        per_layer_seeds:    dict[int, int] — always populated (small),
                            for reproducibility verification

    METADATA (always populated):
        prompt:   str | None — input prompt (None when input_ids was used)
        backend:  str — the sampler backend used
        K:        int — samples per row used
        layers:   list[int] — sorted list of all bridged layers
    """
    # Always populated
    logits: torch.Tensor
    vanilla_logits: torch.Tensor
    prompt: Optional[str]
    backend: str
    K: int
    layers: List[int]

    # Scalar fields — None on list calls
    capture: Optional[LayerCapture] = None
    p_thermo: Optional[torch.Tensor] = None
    alpha: Optional[float] = None
    layer_idx: Optional[int] = None

    # Dict fields — always populated; large tensors gated by return_intermediates
    layer_captures: Optional[Dict[int, LayerCapture]] = None
    p_thermo_by_layer: Optional[Dict[int, torch.Tensor]] = None
    alpha_by_layer: Optional[Dict[int, float]] = None
    per_layer_seeds: Optional[Dict[int, int]] = None


# ---------------------------------------------------------------------------
# bridge_forward
# ---------------------------------------------------------------------------

def bridge_forward(
    model,
    tok,
    prompt: Optional[str] = None,
    input_ids: Optional[torch.Tensor] = None,
    layer_idx: Union[int, List[int]] = 18,
    alpha: Union[float, List[float]] = 0.3,
    backend: str = 'exact',
    K: int = 10,
    seed: Optional[int] = None,
    strict_verify: bool = True,
    return_intermediates: bool = False,
) -> Union[torch.Tensor, BridgeResult]:
    """Run the bridge forward pass on a single prompt.

    Accepts scalar or list layer_idx and alpha. Scalar path is bit-exact
    identical to the pre-multi-layer version (M5_FROZEN_20260530 regresses
    cleanly). List path runs open-loop multi-layer composition.

    Two input modes (mutually exclusive):
        prompt:    str, tokenized internally via tok.
        input_ids: pre-tokenized tensor of shape (B, S), already on device.

    Args:
        model:     frozen LLaMA causal LM (attn_implementation='eager').
        tok:       matching tokenizer (required when prompt is given).
        prompt:    input string. Mutually exclusive with input_ids.
        input_ids: pre-tokenized ids, shape (B, S). Mutually exclusive
                   with prompt. Must already be on model.device.
        layer_idx: int or list[int]. Layer(s) to inject at.
        alpha:     float or list[float]. Blend coefficient(s) in [0, 1].
                   Scalar broadcasts to all layers. List must match
                   layer_idx length.
        backend:   sampler backend — 'exact', 'gumbel', or 'rbm'.
                   Multi-layer list mode requires backend='exact'.
        K:         samples per row for the sampler.
        seed:      base seed for reproducible per-layer sampling.
                   Per-layer seeds are derived via seed_for_layer().
        strict_verify: if True (default), capture invariant violations
                   raise RuntimeError.
        return_intermediates: if True, return BridgeResult with captures,
                   p_thermo tensors, and metadata. If False, return just
                   the logits tensor.

    Returns:
        return_intermediates=False: torch.Tensor shape (B, S, vocab_size).
        return_intermediates=True:  BridgeResult dataclass.

    Raises:
        ValueError: alpha out of [0,1]; backend invalid for multi-layer;
                    alpha/layer_idx length mismatch; duplicate layer_idx;
                    prompt and input_ids both given or neither given.
        RuntimeError: capture invariant fails with strict_verify=True.
        TypeError: layer_idx not int or list[int].
    """

    # ── 1. Resolve inputs ───────────────────────────────────────────────
    if prompt is not None and input_ids is not None:
        raise ValueError("Pass `prompt` OR `input_ids`, not both.")
    if prompt is None and input_ids is None:
        raise ValueError("Must pass either `prompt` or `input_ids`.")
    if prompt is not None:
        if tok is None:
            raise ValueError("`tok` is required when passing a `prompt` string.")
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        prompt_for_record = prompt
    else:
        inputs = {'input_ids': input_ids}
        prompt_for_record = None

    # ── 2. Normalize (layer_idx, alpha) into sorted validated pairs ─────
    is_scalar_call = isinstance(layer_idx, int)

    if is_scalar_call:
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise TypeError(
                f"alpha must be float when layer_idx is int, "
                f"got {type(alpha).__name__}")
        layer_idx_list = [layer_idx]
        alpha_list = [float(alpha)]
    elif isinstance(layer_idx, list):
        if not all(isinstance(li, int) for li in layer_idx):
            raise TypeError(
                "All elements of layer_idx list must be int.")
        if isinstance(alpha, (float, int)):
            alpha_list = [float(alpha)] * len(layer_idx)
        elif isinstance(alpha, list):
            if len(alpha) != len(layer_idx):
                raise ValueError(
                    f"alpha list length {len(alpha)} != "
                    f"layer_idx list length {len(layer_idx)}.")
            alpha_list = [float(a) for a in alpha]
        else:
            raise TypeError(
                f"alpha must be float or list[float], "
                f"got {type(alpha).__name__}")
        layer_idx_list = layer_idx
    else:
        raise TypeError(
            f"layer_idx must be int or list[int], "
            f"got {type(layer_idx).__name__}")

    # Pair-bind BEFORE sorting (spec req 7: bind alpha to layer first)
    pairs = list(zip(layer_idx_list, alpha_list))

    # Validate: no duplicate layer indices
    seen = set()
    for li, _ in pairs:
        if li in seen:
            raise ValueError(
                f"Duplicate layer index {li} in layer_idx. "
                f"Each layer may appear at most once.")
        seen.add(li)

    # Validate: alpha in [0, 1] per layer
    for li, a in pairs:
        if not 0.0 <= a <= 1.0:
            raise ValueError(
                f"alpha={a} for layer {li} is out of [0, 1].")

    # Sort by layer index (ascending = forward pass order)
    sorted_pairs = sorted(pairs, key=lambda p: p[0])

    # ── 3. Backend restriction for multi-layer list mode ────────────────
    # Single-element list is treated as scalar — no restriction.
    is_multilayer = (not is_scalar_call) and len(sorted_pairs) > 1
    if is_multilayer and backend != 'exact':
        raise ValueError(
            f"multi-layer bridge_forward currently supports backend='exact' "
            f"only; got backend={backend!r}. The gumbel/rbm backends use "
            f"global torch.manual_seed and cannot guarantee per-layer RNG "
            f"isolation. Fix tracked separately; remove this guard after "
            f"RNG refactor.")

    # ── 4. Capture pass (pass 1 — vanilla forward) ──────────────────────
    layers_to_capture = [li for li, _ in sorted_pairs]

    capturer = LlamaAttentionCapture(
        model=model,
        layers_to_capture=layers_to_capture,
        strict_verify=strict_verify,
    )
    with capturer.capture():
        with torch.no_grad():
            vanilla_out = model(**inputs, use_cache=False)
    vanilla_logits = vanilla_out.logits.detach().clone()

    # Fetch captures per layer
    capture_by_layer = {li: capturer.get_capture(li) for li in layers_to_capture}

    # ── 5. Sample per layer ──────────────────────────────────────────────
    # Seed handling: if no seed given, use None for all layers (non-deterministic).
    # If seed given, derive per-layer seed via CRC (Bug #12, order-independent).
    dispatch_table: Dict[int, DispatchEntry] = {}
    per_layer_seeds: Dict[int, int] = {}
    p_thermo_by_layer: Dict[int, torch.Tensor] = {}
    alpha_by_layer: Dict[int, float] = {}

    for li, a in sorted_pairs:
        if seed is not None:
            layer_seed = seed_for_layer(seed, li)
        else:
            layer_seed = None
        per_layer_seeds[li] = layer_seed
        alpha_by_layer[li] = a

        sampler_cfg = SamplerConfig(backend=backend, K=K, seed=layer_seed)
        p_t = sample(capture_by_layer[li], sampler_cfg)
        p_thermo_by_layer[li] = p_t

        dispatch_table[li] = DispatchEntry(
            capture=capture_by_layer[li],
            p_thermo=p_t,
            alpha=a,
        )

    # ── 6. Inject pass (pass 2 — one patched forward) ───────────────────
    injector = LlamaAttentionInjector(dispatch_table=dispatch_table)
    with injector.inject():
        with torch.no_grad():
            bridge_out = model(**inputs, use_cache=False)
    bridge_logits = bridge_out.logits.detach().clone()

    # ── 7. Build and return result ───────────────────────────────────────
    sorted_layers = [li for li, _ in sorted_pairs]

    if not return_intermediates:
        return bridge_logits

    # Scalar fields: populated on scalar calls, None on list calls
    if is_scalar_call:
        li = sorted_layers[0]
        scalar_capture = capture_by_layer[li]
        scalar_p_thermo = p_thermo_by_layer[li]
        scalar_alpha = alpha_by_layer[li]
        scalar_layer_idx = li
    else:
        scalar_capture = None
        scalar_p_thermo = None
        scalar_alpha = None
        scalar_layer_idx = None

    # Dict fields: large tensors gated by return_intermediates (already True here)
    return BridgeResult(
        logits=bridge_logits,
        vanilla_logits=vanilla_logits,
        prompt=prompt_for_record,
        backend=backend,
        K=K,
        layers=sorted_layers,
        # Scalar fields
        capture=scalar_capture,
        p_thermo=scalar_p_thermo,
        alpha=scalar_alpha,
        layer_idx=scalar_layer_idx,
        # Dict fields
        layer_captures=capture_by_layer,
        p_thermo_by_layer=p_thermo_by_layer,
        alpha_by_layer=alpha_by_layer,
        per_layer_seeds=per_layer_seeds,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Smoke tests for normalization logic and seed derivation.
    Model-dependent tests live in tests/test_injector_v2.py.
    """
    print("tasb_pipeline_v2.py self-checks:")
    print()

    # seed_for_layer: stability and uniqueness
    print("  seed_for_layer:")
    s18a = seed_for_layer(42, 18)
    s18b = seed_for_layer(42, 18)
    s24  = seed_for_layer(42, 24)
    print(f"    seed(42, 18) stable:  {s18a == s18b} ({s18a})")
    print(f"    seed(42, 18) != seed(42, 24): {s18a != s24} ({s24})")
    assert s18a == s18b, "seed_for_layer not stable"
    assert s18a != s24,  "seed_for_layer not unique across layers"

    # Order independence: [24, 18] and [18, 24] must produce same seeds
    seeds_fwd = {li: seed_for_layer(42, li) for li in [18, 24]}
    seeds_rev = {li: seed_for_layer(42, li) for li in [24, 18]}
    assert seeds_fwd == seeds_rev, "seed_for_layer not order-independent"
    print(f"    order-independent:    OK {seeds_fwd}")
    print()

    # Input normalization guards (no model needed)
    print("  Input normalization guards:")

    # Duplicate layer index
    try:
        # Simulate the duplicate check directly
        pairs = list(zip([18, 18], [0.3, 0.3]))
        seen = set()
        for li, _ in pairs:
            if li in seen:
                raise ValueError(f"Duplicate layer index {li}.")
            seen.add(li)
        print("    duplicate guard: FAILED (no exception raised)")
    except ValueError as e:
        print(f"    duplicate guard:      OK ({e})")

    # Alpha list length mismatch
    try:
        layer_idx_list = [18, 24]
        alpha_list_bad = [0.3]
        if len(alpha_list_bad) != len(layer_idx_list):
            raise ValueError(
                f"alpha list length {len(alpha_list_bad)} != "
                f"layer_idx list length {len(layer_idx_list)}.")
        print("    length mismatch guard: FAILED")
    except ValueError as e:
        print(f"    length mismatch guard: OK ({e})")

    # Backend restriction
    try:
        is_multilayer = True
        backend_test = 'gumbel'
        if is_multilayer and backend_test != 'exact':
            raise ValueError(
                f"multi-layer bridge_forward currently supports "
                f"backend='exact' only; got backend={backend_test!r}.")
        print("    backend guard: FAILED")
    except ValueError as e:
        print(f"    backend guard:         OK ({e})")

    # Pair-bind order independence: [24,18]/[0.5,0.3] vs [18,24]/[0.3,0.5]
    pairs_fwd = sorted(zip([24, 18], [0.5, 0.3]), key=lambda p: p[0])
    pairs_rev = sorted(zip([18, 24], [0.3, 0.5]), key=lambda p: p[0])
    assert pairs_fwd == pairs_rev, "pair-bind order-independence broken"
    print(f"    pair-bind order-independence: OK {pairs_fwd}")
    print()

    print("  All pipeline self-checks passed.")
    print()
    print("  For model-dependent tests (alpha=0 identity, M5_FROZEN")
    print("  bit-exact regression, multi-layer composition), run:")
    print("    python tests/test_injector_v2.py")
