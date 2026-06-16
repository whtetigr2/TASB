# TASB — Thermodynamic Attention Sampling Bridge

**A no-retrain inference bridge between frozen pretrained transformers and stochastic-compute hardware substrates.**

*Paul W. Shaver — 2026*

> **Push note (2026-06-16):** THRML vmap patch — 8x throughput improvement (22s/token → 2.5s/token on T4). Repo restructured. All faithfulness numbers unchanged.

---

## What this is

Extropic's own literature states the problem plainly:

> *"You cannot just take a trained transformer or diffusion model that runs on a GPU and drop it onto a TSU. The architectures are too different."*

TASB is the software answer to that problem. It translates digital attention sampling into TSU-compatible Boltzmann attention sampling — without retraining the model, without modifying its weights, without fine-tuning, distillation, or adapter layers of any kind.

The bridge works because softmax attention at scale factor `1/√d_k` is mathematically identical to a Boltzmann distribution at temperature `T = √d_k`. A GPU computes this distribution deterministically and takes a weighted average. A TSU samples from the same distribution physically. TASB is the translation layer that lets the second regime stand in for the first.

---

## The result in one table

Measured on LLaMA 3.2-3B, teacher-forced, 13 alpha values × 8 layer configs × 4 prompts × 40 steps = **8,840 positions** across the full operating envelope:

| Config | Layers | α=0.3 Top-1 | α=0.3 KL | Confident flips (all α) |
|--------|--------|-------------|-----------|------------------------|
| 1L     | [18]   | **100.00%** | 0.00138   | 0 / 8,840 positions    |
| 5L     | [15,18,21,24,27] | **98.82%** | 0.00515 | 0 / 8,840 |
| 10L    | [10–27 even] | **95.29%** | 0.00874 | 4 / 8,840 (at α≥0.85 only) |

**Zero confident-position flips through α=0.70 across all layer configs.**
The 4 confident flips that appear at α≥0.85 occur only in the two heaviest configs (10L and 6L) — 0.045% of all measured positions at maximum TSU participation.

---

## Live four-backend demonstration

Interactive chat runtime with mid-conversation backend switching. Four independent Boltzmann samplers validated on the same frozen LLaMA 3.2-3B at α=1.0, single-layer L18, K=50, on the prompt *"Hello Llama! Are you ready to assist?"*:

| Backend | Sampler                                    | KL-Div    | Top-1   | Confident flips |
|---------|--------------------------------------------|-----------|---------|-----------------|
| exact   | `torch.multinomial` over softmax           | 0.00068   | 100.0%  | 0               |
| gumbel  | Gumbel-max in logit space                  | 0.00047   | 100.0%  | 0               |
| rbm     | Iterative RBM Gibbs                        | 0.00012   | 100.0%  | 0               |
| thrml   | Extropic THRML block-Gibbs Boltzmann       | 0.00185   | 100.0%  | 0               |

The `thrml` backend uses Extropic's reference Boltzmann sampler (`thrml.models.discrete_ebm.CategoricalEBMFactor` driven by `CategoricalGibbsConditional`). The substrate-agnostic claim — that TASB produces equivalent model behavior regardless of which Boltzmann sampler sits underneath — is demonstrated across four independent implementations.

### Try the live chat

```bash
python tasb_llama32_chat_runtime.py --backend thrml
```

Mid-conversation slash commands switch backends, layers, alpha, and K on the fly:

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

Every turn displays a HUD with the active backend, KL from vanilla, top-1 agreement, confident-flip count, VRAM, and tokens per second. The bridge state is fully reactive — the next turn after any slash command reflects the new configuration.

---

## Why this matters for hardware vendors

The Z1 chip doesn't need to replace the GPU overnight. TASB enables **progressive substrate adoption** — the chip takes over 1 layer, then 5, then 10, as hardware scales. The software bridge is the integration layer that makes that path possible without retraining the model at each step.

This is the "hard part" that current stochastic hardware roadmaps treat as an open problem. On the software side, it is solved.

---

## Architecture

```
                        bridge_forward()
                              │
               ┌──────────────┼──────────────┐
               │         normalize            │
               │   pair-bind · sort · guard   │
               └──────────────┬──────────────┘
                              │
               ┌──────────────┼──────────────┐
               │      Pass 1 — CAPTURE        │
               │  post-RoPE Q, K per layer    │
               │  attn_weights · mask · scale │
               │  capture invariant verified  │
               └──────────────┬──────────────┘
                              │
               ┌──────────────┼──────────────┐
               │         SAMPLE               │
               │  seed_for_layer(base, idx)   │
               │  one of: exact | gumbel |    │
               │          rbm   | thrml       │
               │  → p_thermo (B, n_q, S, S)  │
               └──────────────┬──────────────┘
                              │
               ┌──────────────┼──────────────┐
               │      Pass 2 — INJECT         │
               │  ONE patch on eager_attn     │
               │  dispatch by args[0].layer_idx│
               │  blend = (1-α)·w + α·p_thermo│
               │  non-target → passthrough    │
               └──────────────┬──────────────┘
                              │
                        BridgeResult
                  logits · KL · per_layer_seeds
```

**Key implementation insight:** LLaMA's `LlamaAttention` modules carry `self.layer_idx` as an attribute. The injector patches `eager_attention_forward` once and routes by `args[0].layer_idx` — no stack walking, no module lookup dict, one attribute read per attention call.

---

## Validated claims

All claims are backed by CSV artifacts in `results/`. Every number is reproducible.

### Faithfulness (M5 — sealed 2026-05-30)

On corrected post-RoPE capture, single-layer L18 TASB at α=0.3 preserves vanilla top-1 on **98.9%** of teacher-forced positions overall and **98.3%** on non-cycle-looped prompts, with mean KL **0.00118**. Across the full α sweep to α=1.0, zero flips at confident positions. Disagreements concentrated in ambiguous positions where vanilla had small top-1 margins.

### Production-realism (M6 — 2026-05-31)

Under realistic top-p sampling (shadow mode), top-1 agreement **96.9%**, mean KL **0.00149**. The bridge's trajectory divergence at α=0.3 is statistically indistinguishable from RNG-driven vanilla-vs-vanilla variance. Top-p sampling is the dominant source of trajectory chaos — not the bridge.

### Full characterization (M7 — 2026-06-03)

Seven sub-sweeps across the complete operating envelope:

| Sweep | Variable | Result |
|-------|----------|--------|
| Layer sweep | L0–L27 | Zero confident flips at every layer |
| K sweep | K=1–100 | Zero confident flips; KL drops monotonically |
| Seed variance | 12 seeds | Top-1 98.80%±0.56%; non-unanimous positions exclusively AMBIGUOUS |
| α fine sweep | α=0.0–1.0 | Zero confident flips across full range |
| Multi-layer composition | C1–C5 (1–5L) | Zero confident flips; KL saturates, does not compound |
| Scaling curve | 1L–10L | Zero confident flips through 10 layers; KL growth sub-linear |
| 2D sweep | 13α × 8 configs | Zero confident flips through α=0.70; boundary at α=0.85/10L |

### Substrate-agnostic bridge (live chat, 2026-06-10)

Four independent Boltzmann sampler implementations validated end-to-end through the chat runtime on real prompts with full conversation context. KL < 0.01 and zero confident flips on every backend at α=1.0, single layer. The bridge is sampler-implementation-independent: any backend that draws from `exp(J)/Z` at the attention-scale temperature satisfies the contract.

### Structural alignment finding

The bridge adds **4–7× more KL at ambiguous positions than at confident positions** across the full alpha range. Perturbation energy is geometrically concentrated in the low-certainty regions of the model's probability landscape. This is not a tuned behavior — it is a consequence of Boltzmann sampling at attention-scale temperature interacting with the model's existing confidence geometry.

### Long-context and generation (stress test)

- Zero OOM through 10 layers at 512-token context (peak VRAM 3.58GB on 4-bit 3B model)
- Zero speed penalty on capture-once generation: 5.9 tok/s with or without bridge at any layer count
- 5-paragraph side-by-side at 256-token context: word-for-word identical output between vanilla and 5L bridge at α=0.3

---

## Operating envelope

The measured safe operating regime for the demo slider:

```
α ∈ [0.00, 0.70]:  Zero confident flips at any layer config (0–10L)
α ∈ [0.70, 0.85]:  First confident flips appear at 10L and 6L only
α ∈ [0.85, 1.00]:  4 total confident flips across 8,840 positions
                    at the two heaviest configs only
```

**Production recommendation:** α=0.3, K=50 (for TSU silicon), any layer config.

---

## Quickstart

### Live chat with the bridge

```bash
git clone https://github.com/whtetigr2/TASB
cd TASB
pip install torch transformers bitsandbytes thrml

python tasb_llama32_chat_runtime.py --backend exact
# Then in chat: /backend thrml, /alpha 1.0, /telemetry, etc.
```

### Programmatic single-pass

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

# Same call with Extropic's THRML reference sampler
result_thrml = bridge_forward(
    model, tok,
    prompt="The relationship between thermodynamics and computation",
    layer_idx=18, alpha=0.3, backend='thrml', K=10, seed=42,
    return_intermediates=True)

# Multi-layer bridge (5 layers simultaneously)
result_5L = bridge_forward(
    model, tok,
    prompt="The relationship between thermodynamics and computation",
    layer_idx=[15, 18, 21, 24, 27], alpha=0.3,
    backend='exact', K=10, seed=42,
    return_intermediates=True)
```

---

## Regression tests

```bash
# All 15 multi-layer spec requirements
python tests/test_multilayer_v1.py

# Capture invariant + RoPE regression
python tests/test_capture_v2.py

# Sampler backend agreement (all four)
python tests/test_sampler_v2.py

# Injector invariants
python tests/test_injector_v2.py

# THRML backend standalone validation
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

## Key citations

- **Softmax = Boltzmann at T=√d_k:** Kim, "Thermodynamic Isomorphism of Transformers" (arXiv:2602.08216)
- **Attention = Hopfield energy minimization:** Ramsauer et al., "Hopfield Networks Is All You Need" (ICLR 2021)
- **α as QUBO-style thermodynamic penalty weight:** "Thermodynamic significance of QUBO encoding on quantum annealers" (arXiv:2601.04402)
- **p-bit substrate physics:** Camsari et al., "p-Bits for Probabilistic Spin Logic"; "Probabilistic Computing with p-Bits"
- **Transformer architecture:** Vaswani et al., "Attention Is All You Need"
- **THRML reference sampler:** Extropic AI, `extropic-ai/thrml` on GitHub

---

## What this is not

- Not a new attention architecture
- Not a KV-cache eviction policy
- Not mechanistic interpretability
- Not a training method
- Not "thermodynamic attention" in the KV-cache eviction sense (Carnot Attention, HuggingFace forum Feb 2026 — entirely different work)

TASB is a **sampling-regime translation layer.** The model is frozen. The weights are unchanged. The bridge substitutes equivalent samples from the same distribution the model defines — it just sources those samples from stochastic hardware (or any Boltzmann sampler) instead of a deterministic softmax.

---

## License and patent notice

Research use permitted with attribution. Commercial licensing inquiries: contact the inventor.

---

*Results are reproducible. Every claim has a CSV. Every CSV has a script that generated it.*
