# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-29

### Added
- `thermobridge.bridge.bridge_forward` — canonical two-pass thermodynamic attention sampling
- Four sampling backends: `exact`, `gumbel`, `rbm`, `thrml`
- Multi-layer open-loop composition with order-independent per-layer seed derivation
- `LlamaAttentionCapture` — RoPE-aware post-softmax attention capture with capture invariant
- `LlamaAttentionInjector` — per-Q-head p\_thermo blend injection
- Full validation suite: T1.A–T1.D, T2.B, T2.C, T3, T5
- Validated on LLaMA 3.2-3B and OLMoE-1B-7B: ΔPPL ≤ +0.012 nats at α=0.3, K=10

### Notes
- THRML backend requires `pip install thrml` (Extropic hardware path)
- Multi-layer composition with `thrml` backend is single-layer only pending validation
