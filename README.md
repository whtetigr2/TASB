# TASB — Thermodynamic Attention Sampling Bridge

**A no-retrain inference bridge between frozen pretrained transformers and stochastic-compute hardware substrates.**

*Paul W. Shaver — 2026*

> **Push note (2026-06-19):** Docs cleanup — README trimmed to "what it is + how to run it." Full results, characterization tables, and the operating envelope moved to MILESTONES.md; key citations moved to CITATIONS.md. All faithfulness numbers unchanged.

---

## What this is

TASB lets a frozen, pretrained transformer run its attention on stochastic-compute hardware (e.g. an Extropic-style TSU) without any retraining, weight changes, fine-tuning, distillation, or adapter layers. It works because softmax attention at scale factor `1/√d_k` is mathematically identical to a Boltzmann distribution at temperature `T = √d_k`: a GPU computes that distribution deterministically and takes a weighted average, while a TSU samples from the same distribution physically. TASB is the translation layer that lets the second regime stand in for the first.

## What this is not

Not a new attention architecture, a training method, a KV-cache eviction policy (e.g. Carnot Attention — unrelated work), or interpretability tooling. The model is frozen and its weights are unchanged; TASB only substitutes equivalent samples from the distribution the model already defines.

---

## Quickstart

Clone, install, and run the live four-backend chat demo:

```bash
git clone https://github.com/whtetigr2/TASB
cd TASB
pip install torch transformers bitsandbytes thrml

python tasb_llama32_chat_runtime.py --backend exact
```

Switch samplers, layers, alpha, and K mid-conversation with slash commands:

```
/backend thrml      switch sampler (exact | gumbel | rbm | thrml | vanilla)
/alpha 1.0          full TSU participation
/layer 18           single-layer injection
/layers 15 18 21    multi-layer composition
/k 50               Boltzmann samples per position
/telemetry          show full per-token pipeline (CAPTURE → SAMPLE → INJECT → TOKEN)
/sweep              alpha dose-response on last prompt
/stats              session metrics summary
```

Each turn shows a HUD with the active backend, KL from vanilla, top-1 agreement, confident-flip count, VRAM, and tokens/sec.

Full setup details — CPU-only path, HuggingFace model access, and troubleshooting — are in [QUICKSTART.md](QUICKSTART.md).

---

## Programmatic usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tasb_pipeline_v2 import bridge_forward

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16),
    attn_implementation='eager', device_map='auto')
model.eval()

# Single-layer bridge at production config
result = bridge_forward(
    model, tok,
    prompt="The relationship between thermodynamics and computation",
    layer_idx=18, alpha=0.3, backend='exact', K=10, seed=42,
    return_intermediates=True)

print(f"Top-1 agree: {result.logits.argmax(-1).eq(result.vanilla_logits.argmax(-1)).float().mean():.3f}")
print(f"Mean KL: {torch.nn.functional.kl_div(result.logits.log_softmax(-1), result.vanilla_logits.softmax(-1), reduction='batchmean'):.5f}")
```

Multi-layer composition and the Extropic THRML reference backend are covered in [QUICKSTART.md](QUICKSTART.md).

---

## Regression tests

```bash
python tests/test_multilayer_v1.py
python tests/test_capture_v2.py
python tests/test_sampler_v2.py
python tests/test_injector_v2.py
python tests/test_thrml_backend.py
```

---

## Production files

| File | Role |
|------|------|
| `tasb_capture_v2.py` | Post-RoPE Q/K capture; patches `apply_rotary_pos_emb` + `eager_attention_forward` |
| `tasb_sampler_v2.py` | Backend dispatch: `exact`, `gumbel`, `rbm`, `thrml` |
| `tasb_sampler_thrml.py` | THRML block-Gibbs Boltzmann backend; bridges to `thrml.models.discrete_ebm` |
| `tasb_injector_v2.py` | Dict-dispatch multi-layer injector; one patch, routes by `args[0].layer_idx` |
| `tasb_pipeline_v2.py` | `bridge_forward()` entry point; input normalization, seed derivation, `BridgeResult` |
| `tasb_llama32_chat_runtime.py` | Live console chat with HUD, slash commands, telemetry, CSV logging |

---

## Validated results

- **Zero confident-position flips through α=0.70** across every layer config (0–10L).
- **Bit-exact α=0 identity**, regression-tested per step in both scalar and multi-layer form.
- Single-layer L18 at α=0.3: **100% top-1, KL 0.00138, 0 confident flips** across 8,840 measured positions.

Full picture — the 8,840-position sweep, four-backend (exact/gumbel/rbm/thrml) comparison, operating envelope, and M5–M7 characterization — is in [MILESTONES.md](MILESTONES.md). Key citations are in [CITATIONS.md](CITATIONS.md).

---

## License

Research use permitted with attribution. Commercial licensing inquiries: contact the inventor. See [LICENSE.md](LICENSE.md).
