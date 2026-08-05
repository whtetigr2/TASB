"""
bridge.py — Canonical two-pass thermodynamic attention sampling bridge
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver.

Two-pass protocol:
  1. CAPTURE  — vanilla forward with patched apply_rotary_pos_emb and
                eager_attention_forward; stashes post-RoPE Q, K, mask,
                scaling, and attn_weights at every target layer.
  2. SAMPLE   — for each target layer, draw p_thermo from the chosen
                backend using a per-layer seed derived from (base_seed, layer).
  3. INJECT   — second forward with one patched eager_attention_forward;
                dispatch table routes each layer to its pre-sampled p_thermo.

Supports scalar layer_idx (single layer) and list layer_idx (multi-layer).
Scalar path is backward compatible with all prior milestone results.

OPEN-LOOP COMPOSITION (multi-layer)
------------------------------------
p_thermo at each layer is sampled from the vanilla capture (pass 1), not
from the perturbed hidden state resulting from upstream injections.
Layer N's sample does not see the effect of layer N-1's injection.
Closed-loop (sequential recomputation) is future work.

SEED DERIVATION
---------------
Per-layer seeds are derived as:
    seed_for_layer(base, layer) =
        (base + zlib.crc32(f"layer_{layer}".encode())) & 0x7FFFFFFF

Order-independent (same base_seed + layer always gives the same seed,
regardless of what other layers are in the call). zlib.crc32 is used
because hash() is process-salted in CPython and not stable across runs.

BACKEND RESTRICTION
-------------------
Multi-layer list mode (len > 1) requires backend in ('exact', 'gumbel', 'rbm').
These backends use a scoped per-layer torch.Generator, guaranteeing RNG
isolation. The thrml backend is single-layer only until multi-layer is
validated.
==============================================================================
"""

import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from thermobridge.capture import LlamaAttentionCapture, LayerCapture
from thermobridge.sampler import sample, SamplerConfig
from thermobridge.inject import LlamaAttentionInjector, DispatchEntry


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def seed_for_layer(base_seed: int, layer_idx: int) -> int:
    """Derive a stable, order-independent per-layer seed.

    Args:
        base_seed: base seed passed to bridge_forward.
        layer_idx: layer to derive a seed for.

    Returns:
        Non-negative int suitable for use as a sampler seed.
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
        alpha_by_layer:     dict[int, float] — always populated
        per_layer_seeds:    dict[int, int] — always populated

    METADATA (always populated):
        prompt:   str | None — input prompt (None when input_ids was used)
        backend:  str — the sampler backend used
        K:        int — samples per row used
        layers:   list[int] — sorted list of all bridged layers
    """
    logits: torch.Tensor
    vanilla_logits: torch.Tensor
    prompt: Optional[str]
    backend: str
    K: int
    layers: List[int]

    capture: Optional[LayerCapture] = None
    p_thermo: Optional[torch.Tensor] = None
    alpha: Optional[float] = None
    layer_idx: Optional[int] = None

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

    Scalar layer_idx runs single-layer injection (backward compatible).
    List layer_idx runs open-loop multi-layer composition.

    Two input modes (mutually exclusive):
        prompt:    str, tokenized internally via tok.
        input_ids: pre-tokenized tensor (B, S), already on model.device.

    Args:
        model:     frozen LLaMA causal LM (attn_implementation='eager').
        tok:       matching tokenizer (required when prompt is given).
        prompt:    input string. Mutually exclusive with input_ids.
        input_ids: pre-tokenized ids (B, S). Mutually exclusive with prompt.
        layer_idx: int or list[int]. Layer(s) to inject at.
        alpha:     float or list[float]. Blend coefficient(s) in [0, 1].
                   Scalar broadcasts to all layers.
        backend:   'exact', 'gumbel', 'rbm', or 'thrml'.
                   Multi-layer list mode requires 'exact', 'gumbel', or 'rbm'.
        K:         samples per row for the sampler.
        seed:      base seed for reproducible per-layer sampling.
        strict_verify: if True, capture invariant violations raise RuntimeError.
        return_intermediates: if True, return BridgeResult. If False, return
                   just the logits tensor (B, S, vocab_size).

    Returns:
        return_intermediates=False: torch.Tensor (B, S, vocab_size).
        return_intermediates=True:  BridgeResult.

    Raises:
        ValueError: invalid alpha, backend, or layer_idx arguments.
        RuntimeError: capture invariant failure (strict_verify=True).
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
        if not isinstance(alpha, (float, int)):
            raise TypeError(
                f"alpha must be float when layer_idx is int, "
                f"got {type(alpha).__name__}")
        layer_idx_list = [layer_idx]
        alpha_list = [float(alpha)]
    elif isinstance(layer_idx, list):
        if not all(isinstance(li, int) for li in layer_idx):
            raise TypeError("All elements of layer_idx list must be int.")
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
                f"alpha must be float or list[float], got {type(alpha).__name__}")
        layer_idx_list = layer_idx
    else:
        raise TypeError(
            f"layer_idx must be int or list[int], got {type(layer_idx).__name__}")

    # Bind alpha to layer before sorting (order-independent)
    pairs = list(zip(layer_idx_list, alpha_list))

    seen: set = set()
    for li, _ in pairs:
        if li in seen:
            raise ValueError(f"Duplicate layer index {li} in layer_idx.")
        seen.add(li)

    for li, a in pairs:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha={a} for layer {li} is out of [0, 1].")

    sorted_pairs = sorted(pairs, key=lambda p: p[0])

    # ── 3. Backend restriction for multi-layer list mode ────────────────
    is_multilayer = (not is_scalar_call) and len(sorted_pairs) > 1
    MULTILAYER_BACKENDS = ('exact', 'gumbel', 'rbm')
    if is_multilayer and backend not in MULTILAYER_BACKENDS:
        raise ValueError(
            f"multi-layer bridge_forward supports backend in "
            f"{MULTILAYER_BACKENDS} (scoped per-layer RNG isolation); "
            f"got backend={backend!r}. thrml multi-layer is not yet "
            f"validated — use single-layer for thrml.")

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

    capture_by_layer = {li: capturer.get_capture(li) for li in layers_to_capture}

    # ── 5. Sample per layer ──────────────────────────────────────────────
    dispatch_table: Dict[int, DispatchEntry] = {}
    per_layer_seeds: Dict[int, int] = {}
    p_thermo_by_layer: Dict[int, torch.Tensor] = {}
    alpha_by_layer: Dict[int, float] = {}

    for li, a in sorted_pairs:
        layer_seed = seed_for_layer(seed, li) if seed is not None else None
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

    return BridgeResult(
        logits=bridge_logits,
        vanilla_logits=vanilla_logits,
        prompt=prompt_for_record,
        backend=backend,
        K=K,
        layers=sorted_layers,
        capture=scalar_capture,
        p_thermo=scalar_p_thermo,
        alpha=scalar_alpha,
        layer_idx=scalar_layer_idx,
        layer_captures=capture_by_layer,
        p_thermo_by_layer=p_thermo_by_layer,
        alpha_by_layer=alpha_by_layer,
        per_layer_seeds=per_layer_seeds,
    )


# ---------------------------------------------------------------------------
# Self-test (no model required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("bridge.py self-checks:")

    s18a = seed_for_layer(42, 18)
    s18b = seed_for_layer(42, 18)
    s24  = seed_for_layer(42, 24)
    assert s18a == s18b, "seed_for_layer not stable"
    assert s18a != s24,  "seed_for_layer not unique across layers"
    seeds_fwd = {li: seed_for_layer(42, li) for li in [18, 24]}
    seeds_rev = {li: seed_for_layer(42, li) for li in [24, 18]}
    assert seeds_fwd == seeds_rev, "seed_for_layer not order-independent"
    print(f"  seed_for_layer: stable, unique, order-independent  OK {seeds_fwd}")

    try:
        pairs = list(zip([18, 18], [0.3, 0.3]))
        seen: set = set()
        for li, _ in pairs:
            if li in seen:
                raise ValueError(f"Duplicate layer index {li}.")
            seen.add(li)
    except ValueError as e:
        print(f"  duplicate guard:  OK ({e})")

    print("  All self-checks passed.")
