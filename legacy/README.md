# TASB Legacy Archive

Files archived from the pre-RoPE-bug-fix era. Preserved for provenance but
**must not be cited as bridge-faithfulness evidence**.

## What's in here

### `pre_rope_capture/` — the broken capture stack
The original TASB stack captured pre-RoPE `q_proj`/`k_proj` outputs and
reconstructed `QK^T` from them. HF LLaMA applies RoPE between projection
and matmul, so the reconstructed scores correspond to a different energy
landscape than the captured post-softmax weights.

Files: `tasb_two_pass_1_.py`, `tasb_core_1_.py`, `tasb_llama_config_2_.py`,
`tasb_llama_hook_1_.py`, `bridge_gating_2_.py`, and the four
`tasb_*_test.py` wrappers.

### `stress_wrappers/` — pre-refactor measurement wrappers
v2 deep ablation, v3 encoding comparison, v4 ironclad design.

### `results/` — CSV outputs from legacy runs
Pre-refactor `tasb_stress_v*` and `tasb_encoding_v*` CSVs.

## Status: NOT VALID AS BRIDGE-FAITHFULNESS EVIDENCE

Confirmed by `../tests/test_rope_regression.py`: pre-RoPE reconstruction
max diff vs HF attention weights = 0.0988 (structurally wrong).
Post-RoPE reconstruction in the new stack matches HF bit-exactly (0.00e+00,
verified by `../tests/test_rope_live_capture_v2.py`).

**Implication:** The 99.3% match rates and η scores in legacy CSVs are
NOT evidence of bridge faithfulness. They are evidence of model
robustness to small attention-layer perturbations at α=0.30 — a real
but much weaker finding.

## What CAN be cited
- **Perturbation robustness**: model preserves output 99%+ at α=0.30
  under structured noise injection.
- **Methodology**: multi-encoding test design, McNemar framework,
  engineering posture, bug-guard registry.

## What CANNOT be cited
- Bridge faithfulness on LLaMA 3.2-3B (use new stack results).
- TSU substitute distribution accuracy (sampled wrong distribution).
- Encoding equivalence (all encodings sampled from the same wrong
  distribution; doesn't translate to corrected capture).
