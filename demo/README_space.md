---
title: Thermobridge
emoji: 🌡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Thermodynamic attention sampling — Boltzmann vs softmax
---

# thermobridge

Thermodynamic attention sampling bridge for frozen transformer models.

Replaces softmax attention weights with Boltzmann-sampled distributions drawn
from the same energy landscape — no fine-tuning, no architectural changes.

**Tab 1 (CPU):** Synthetic demo — interactive exploration of the Boltzmann-softmax
equivalence on controlled energy matrices.

**Tab 2 (GPU):** Full pipeline on a real LLaMA 3.2-3B model. Requires GPU hardware
and HF token with LLaMA access — see the tab for setup instructions.

---

Patent Pending · USPTO Provisional 64/019,999 ·
[GitHub](https://github.com/whtetigr2/TASB)
