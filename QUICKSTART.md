# TASB Quickstart

Get the bridge running in under 10 minutes.

---

## Prerequisites

- Python 3.10+
- A NVIDIA GPU with 4GB+ VRAM (recommended) OR 8GB+ system RAM for CPU-only
- A HuggingFace account with access to LLaMA 3.2-3B

**AMD GPU / Windows users:** See the [CPU-only path](#cpu-only-no-gpu) below.
`bitsandbytes` requires CUDA. On AMD or CPU-only systems, run the model
in full precision without quantization.

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/whtetigr2/TASB
cd TASB
```

---

## Step 2 — Install PyTorch

Install PyTorch for your system from https://pytorch.org/get-started/locally/

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision
```

---

## Step 3 — Install remaining dependencies

**With NVIDIA GPU (4-bit quantization enabled):**
```bash
pip install transformers accelerate huggingface_hub bitsandbytes
```

**CPU only (no bitsandbytes):**
```bash
pip install transformers accelerate huggingface_hub
```

---

## Step 4 — Get LLaMA 3.2-3B access

LLaMA 3.2-3B is a gated model. You need to request access once:

1. Go to https://huggingface.co/meta-llama/Llama-3.2-3B
2. Click "Request access" and accept the license (approval is usually instant)
3. Log in from the terminal:

```bash
huggingface-cli login
```

Paste your HuggingFace token when prompted (get one at
https://huggingface.co/settings/tokens).

---

## Step 5 — Run the bridge

### NVIDIA GPU path (recommended)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tasb_pipeline_v2 import bridge_forward

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    ),
    attn_implementation='eager',
    device_map='auto',
)
model.eval()

# Single-layer bridge at production config
result = bridge_forward(
    model, tok,
    prompt="The relationship between thermodynamics and computation",
    layer_idx=18,
    alpha=0.3,
    backend='exact',
    K=10,
    seed=42,
    return_intermediates=True,
)

print(f"Top-1 agreement: {result.logits.argmax(-1).eq(result.vanilla_logits.argmax(-1)).float().mean():.3f}")
print(f"Mean KL:         {torch.nn.functional.kl_div(result.logits.log_softmax(-1), result.vanilla_logits.softmax(-1), reduction='batchmean'):.5f}")
print("Bridge is running.")
```

Expected output:
```
Top-1 agreement: 1.000
Mean KL:         0.00138
Bridge is running.
```

---

### CPU-only (no GPU) {#cpu-only-no-gpu}

Slower (~0.5 tok/s vs ~6 tok/s on GPU) but fully functional.
Needs ~8GB system RAM. Remove the quantization config:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tasb_pipeline_v2 import bridge_forward

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float32,
    attn_implementation='eager',
    device_map='cpu',
)
model.eval()

result = bridge_forward(
    model, tok,
    prompt="The relationship between thermodynamics and computation",
    layer_idx=18,
    alpha=0.3,
    backend='exact',
    K=10,
    seed=42,
    return_intermediates=True,
)

print(f"Top-1 agreement: {result.logits.argmax(-1).eq(result.vanilla_logits.argmax(-1)).float().mean():.3f}")
print("Bridge is running.")
```

---

## Step 6 — Run the tests

Confirms the full stack is working correctly on your hardware:

```bash
python tests/test_capture_v2.py
python tests/test_sampler_v2.py
python tests/test_injector_v2.py
python tests/test_multilayer_v1.py
```

Expected: all pass. The multilayer test loads the model and runs
15 invariant checks — takes ~30 seconds on GPU.

---

## Multi-layer bridge

Inject at multiple layers simultaneously:

```python
# 5-layer spread — production sweet spot
result = bridge_forward(
    model, tok,
    prompt="Your prompt here",
    layer_idx=[15, 18, 21, 24, 27],
    alpha=0.3,
    backend='exact',
    K=10,
    seed=42,
    return_intermediates=True,
)
print(f"Layers bridged: {result.layers}")
print(f"Per-layer seeds: {result.per_layer_seeds}")
```

**Backend restriction:** multi-layer mode requires `backend='exact'`.
The gumbel and rbm backends use global RNG and cannot guarantee
per-layer isolation.

---

## Alpha reference

The `alpha` parameter controls TSU participation:

| Alpha | Meaning | Notes |
|-------|---------|-------|
| 0.0 | Pure GPU (vanilla) | Bit-exact vanilla output |
| 0.3 | Production recommendation | Zero confident flips, all layer counts |
| 0.7 | Maximum safe zone | Zero confident flips, all layer counts |
| 1.0 | Full TSU substitution | 4 flips / 8,840 positions at 10L only |

See `results/tasb_2d_sweep_summary_*.csv` for the full measured
operating envelope across 13 alpha values × 8 layer configs.

---

## Common errors

**`ModuleNotFoundError: No module named 'bitsandbytes'`**
You're on CPU or AMD. Use the CPU-only path above.

**`OSError: meta-llama/Llama-3.2-3B is not a local folder`**
Run `huggingface-cli login` and make sure you've been granted access
at https://huggingface.co/meta-llama/Llama-3.2-3B.

**`RuntimeError: CUDA out of memory`**
Reduce context length or use a smaller batch. The model needs ~2.4GB
VRAM in 4-bit. At 512-token context with 10 layers, peak is ~3.6GB.

**`capture_invariant failed`**
The model version or HuggingFace transformers version changed the
internal attention implementation. Pin to `transformers==4.44.0`
which was the validated version.

---

## Environment

Validated on:
- LLaMA 3.2-3B (meta-llama/Llama-3.2-3B)
- Python 3.11
- PyTorch 2.3.0
- transformers 4.44.0
- bitsandbytes 0.43.3
- NVIDIA A100 (Lightning.ai) and NVIDIA T4

---

## Questions

Open an issue on GitHub or contact whtetigr2@gmail.com
