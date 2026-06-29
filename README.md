# thermobridge

[![CI](https://github.com/whtetigr2/TASB/actions/workflows/ci.yml/badge.svg)](https://github.com/whtetigr2/TASB/actions/workflows/ci.yml)
[![Patent Pending](https://img.shields.io/badge/patent-pending-blue)](https://patents.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Thermodynamic attention sampling bridge for frozen transformer models. Replaces softmax attention weights with Boltzmann-sampled distributions drawn from the same energy landscape — no fine-tuning, no architectural changes, no retraining.

---

## Installation

```bash
pip install thermobridge
```

For the THRML hardware backend (Extropic thermodynamic simulation unit):

```bash
pip install "thermobridge[thrml]"
```

---

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from thermobridge import bridge_forward

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    attn_implementation="eager",
    torch_dtype="auto",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model.eval()

logits = bridge_forward(model, tok, "The thermodynamic basis of intelligence is")
```

### With diagnostics

```python
from thermobridge import bridge_forward

result = bridge_forward(
    model, tok,
    "The thermodynamic basis of intelligence is",
    layer_idx=18,
    alpha=0.3,
    backend="exact",
    K=10,
    return_intermediates=True,
)

print(f"KL(bridge || vanilla): {result.kl:.4f}")
print(f"Layer captured: {result.layer_idx}, seq_len: {result.capture.seq_len}")
```

---

## Backends

| Backend | Description | Status |
|---------|-------------|--------|
| `exact` | K multinomial draws from softmax(Q·Kᵀ·scale + mask). As K→∞, p\_thermo→softmax. | **Production** |
| `gumbel` | Gumbel-max trick: K perturbed argmaxes. Mathematically equivalent to `exact`; hardware-natural. | Production |
| `rbm` | Gibbs-style categorical sampling on the energy landscape. | Research |
| `thrml` | Extropic THRML block-Gibbs Boltzmann sampler. Requires `pip install thrml`. Hardware path for TSU. | Hardware-ready |

**Production config** (validated): `layer_idx=18, alpha=0.3, backend='exact', K=10`

---

## Validation

All results in `validation/results/`. Validated on LLaMA 3.2-3B and OLMoE-1B-7B.

| Test | Result | Threshold |
|------|--------|-----------|
| T1.C Per-head KL fidelity (4 backends × 4 prompts) | All PASS | std(KL/head) < 0.001 |
| T1.D K-convergence (KL ∝ 1/K, R²) | R² = 0.998 | > 0.95 |
| T2.B Gibbs chain mixing (R-hat) | R-hat = 1.0003 | < 1.01 |
| T2.C Detailed balance (chi-squared) | p = 0.43 | > 0.05 |
| T3 Perplexity on LLaMA 3.2-3B | ΔPPL = +0.0114 nats | < 0.05 |
| T5 Perplexity on OLMoE-1B-7B | ΔPPL = +0.0009 nats | < 0.05 |

---

## Citation

If you use thermobridge in your research, please cite:

```bibtex
@misc{shaver2026thermobridge,
  author    = {Shaver, Paul W.},
  title     = {thermobridge: Thermodynamic Attention Sampling for Frozen Transformers},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/whtetigr2/TASB}
}

@patent{shaver2026provisional,
  author    = {Shaver, Paul W.},
  title     = {Thermodynamic Attention Sampling Bridge for Transformer Models},
  number    = {64/019,999},
  type      = {U.S. Provisional Patent Application},
  year      = {2026},
  month     = {March}
}
```

---

## License

MIT — see [LICENSE](LICENSE)

Patent Pending: USPTO Provisional 64/019,999 (filed 2026-03-28).
