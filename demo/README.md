---
title: Thermobridge
emoji: 🌡️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: false
license: mit
short_description: Thermodynamic attention sampling — Boltzmann vs softmax
---

# thermobridge

Thermodynamic attention sampling bridge for frozen transformer models.

Replaces softmax attention weights with Boltzmann-sampled distributions drawn
from the same energy landscape — no fine-tuning, no architectural changes, no retraining.

**Tab 1 — Synthetic Demo (CPU):** Interactive Gibbs-chain animation and analysis plots
on controlled synthetic energy matrices. Watch Boltzmann sampling converge to softmax in real time.

**Tab 2 — Real Attention Viewer:** Actual per-token softmax attention matrices captured
from frozen LLaMA 3.2-3B on A100. Real tokenization — token strings on both axes.
Select prompt / layer / head to inspect exactly which tokens attend to which tokens.

**Tab 3 — Real LLaMA Profile:** Full thermodynamic landscape measured across all 28 layers
and 24 heads (3,360 observations: 5 prompts × 28 layers × 24 heads). Includes the key result:
r(Cv, KL) = 0.8241 — specific heat predicts sampling error (p = 6.1 × 10⁻²⁵).

**Tab 4 — About / Whitepaper:** Full theory, math, and citations.

---

[GitHub](https://github.com/whtetigr2/TASB)
