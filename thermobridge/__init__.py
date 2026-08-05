"""
thermobridge — thermodynamic attention sampling for frozen transformer models
==============================================================================
Author: Paul W. Shaver
© 2026 Paul W. Shaver.

Quick start:

    from thermobridge import bridge_forward

    logits = bridge_forward(model, tok, "Hello, world!")

    # with diagnostics
    from thermobridge import bridge_forward, BridgeResult
    result = bridge_forward(model, tok, "Hello, world!", return_intermediates=True)
    assert isinstance(result, BridgeResult)
==============================================================================
"""

from thermobridge.bridge import bridge_forward, BridgeResult, seed_for_layer
from thermobridge.capture import LlamaAttentionCapture, LayerCapture
from thermobridge.sampler import sample, SamplerConfig
from thermobridge.inject import LlamaAttentionInjector, DispatchEntry

__version__ = "0.1.0"
__author__ = "Paul W. Shaver"

__all__ = [
    "bridge_forward",
    "BridgeResult",
    "seed_for_layer",
    "LlamaAttentionCapture",
    "LayerCapture",
    "sample",
    "SamplerConfig",
    "LlamaAttentionInjector",
    "DispatchEntry",
]
