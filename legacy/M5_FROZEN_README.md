# M5 FROZEN — Faithfulness Core Test (2026-05-30)

**Status:** CLOSED. Codex sign-off received 2026-05-30.
**Model:** meta-llama/Llama-3.2-3B (4-bit nf4 quantization, eager attention)
**Layer:** L18 (single-layer injection)
**Sampler:** exact backend, K=10 samples per row
**Stack:** post-RoPE per-Q-head capture (corrected from pre-RoPE legacy)

---

## What this test measures

M5 is the **teacher-forced matched-context faithfulness test**. It answers Codex's question:

> "Does TSU perturbation preserve or predictably alter the next-token distribution relative to a matched baseline?"

**Teacher-forcing is not production inference.** In M5, both vanilla and TASB are repeatedly given the same context, advanced by *vanilla's* chosen token at each step. This prevents the two runs from drifting apart, so we can measure the bridge's per-step effect cleanly. It isolates the question "given the same input, does TASB preserve the model's next-token behavior?"

The production question — "when TASB is allowed to generate freely, does the whole trajectory remain useful/coherent/stable?" — is **M6**, not M5. M5 is a necessary precondition for M6 to mean anything, but it does not by itself demonstrate production readiness.

## What this test does NOT establish

- It does NOT establish production-realism. Free autoregressive generation under top-p/temperature sampling is M6 territory.
- It does NOT validate across layers, K values, seeds, or model families. That is M7 characterization.
- It does NOT prove TSU hardware behavior. The sampling here runs on GPU in software, simulating what a Boltzmann-faithful substrate would produce. Hardware validation requires Extropic silicon.
- It does NOT auto-detect arbitrary model architectures. That is M8 calibration.

## Canonical claim (Codex-approved wording, 2026-05-30)

> On corrected post-RoPE capture, single-layer L18 TASB at α=0.3 preserves vanilla top-1 on **98.9% of teacher-forced positions overall** and **98.3% on non-cycle-looped prompts**, with **mean KL 0.00118 overall / 0.00201 varied**. Across the full α sweep up to α=1.0, TASB produced **zero flips at confident positions**; disagreements concentrated in ambiguous/moderate positions where vanilla itself had small top-1 margins.

This is the wording for memos, pitch decks, Extropic positioning, and any external claim about M5 until M6 supersedes it.

## Aggregate results (cluster bootstrap, prompt-mean estimator)

| α | top1 (cluster) | top1 95% CI | top5 | mean KL | KL 95% CI |
|---|---|---|---|---|---|
| 0.0 | 100.0% | [100.0, 100.0] | 100.0% | 0.000000 | [0.00000, 0.00000] |
| 0.1 | 99.8% | [99.4, 100.0] | 100.0% | 0.000396 | [0.00025, 0.00056] |
| 0.2 | 99.6% | [99.1, 100.0] | 100.0% | 0.000666 | [0.00042, 0.00093] |
| 0.3 | 98.9% | [98.0, 99.6] | 100.0% | 0.001180 | [0.00066, 0.00179] |
| 0.5 | 98.3% | [96.9, 99.4] | 100.0% | 0.002513 | [0.00134, 0.00384] |
| 0.7 | 98.1% | [96.9, 99.4] | 100.0% | 0.004344 | [0.00229, 0.00663] |
| 1.0 | 97.6% | [95.6, 99.4] | 100.0% | 0.008969 | [0.00436, 0.01419] |

**α=0 identity (per-step measurement, worst case):** `max_abs_diff = 0.00e+00` — bit-exact vanilla via the injector's α=0 fast path. The alpha_zero_identity Stage 1 invariant holds end-to-end, every step, every prompt.

**KL monotonicity:** mean KL grows monotonically with α: 0.0004 → 0.0007 → 0.0012 → 0.0025 → 0.0043 → 0.0090. The bridge behaves predictably across the α range.

## Confidence-bucket recut (the structural finding)

Positions binned by vanilla prob_gap (top-1 prob − top-2 prob):

| bucket | gap range | positions | fraction |
|---|---|---|---|
| CONFIDENT | gap ≥ 0.50 | 360 | 66.7% |
| MODERATE | 0.10 ≤ gap < 0.50 | 105 | 19.4% |
| AMBIGUOUS | gap < 0.10 | 75 | 13.9% |

Top-1 agreement by bucket:

| α | CONFIDENT | MODERATE | AMBIGUOUS |
|---|---|---|---|
| 0.3 | **100.0%** | 99.0% | 93.3% |
| 1.0 | **100.0%** | 95.2% | 89.3% |

**Zero confident-position disagreements at any α from 0.1 to 1.0.** The bridge never flips a token the model predicts with confidence, even when attention is fully replaced (α=1.0).

Disagreement distribution at α=0.3: 5 ambiguous / 1 moderate / **0 confident**. At α=1.0: 8 ambiguous / 5 moderate / **0 confident**. The structural property holds across the entire perturbation range.

## Cycle-looping diagnostic

5 of 9 prompts (HC2, HC3, LC1, LC2, LX1) cycle-loop under teacher-forced greedy decoding — 4-gram repeat rates 61–86%, 8-gram repeat rates 55–85%, consecutive-token repeat rate 0% across all prompts. Looped positions have inflated mean prob_gap (~0.7) and trivially-easy top-1 predictions, inflating aggregate agreement.

The **honest aggregate excluding cycle-looped prompts** at α=0.3:
- Top-1 agreement: 98.3% (varied) vs 98.9% (all)
- Mean KL: 0.00201 (varied) vs 0.00118 (all)

The varied figure is what should be cited externally as the production-relevant number.

## Bug history (what was wrong, what was patched, why we trust this version)

This is M5's **second** run. The first (`tasb_m5_faithfulness_20260528_021614.csv`) had three issues caught in peer review:

**Patch 1 (KL clamp bug, P1).** Original KL/JS/entropy used `softmax(...).clamp(eps=1e-4)` before computing log. On LLaMA's 128k vocab, that clamp added ~12 units of artificial mass to both distributions identically, suppressing real KL by ~15–17×. Fixed by using `F.log_softmax`-based KL with no clamp. Verified on synthetic distributions: confident position 16.6× suppression, ambiguous position 14.6× suppression. The original headline (`mean KL 0.0001 at α=0.3`) was withdrawn; this CSV's 0.00118 is the true value.

**Patch 2 (α=0 sanity hardcoded).** Original code wrote `kl=0, top1_agree=1` for α=0 without running the comparison. Fixed by actually measuring α=0 per step against `bridge_forward`'s captured `vanilla_logits` and recording `alpha0_max_abs_diff` as a regression column. Result: bit-exact identity confirmed on every step of every prompt.

**Patch 3 (process-salted hash for seeds).** Original code used Python's `hash((prompt_id, step))` for sampler seeds. Python's `hash()` is salted per process via `PYTHONHASHSEED`, so `--seed 42` did not reproduce across separate invocations. Fixed by using `zlib.crc32` for stable cross-invocation seeds. Sampler-level reproducibility now holds given the same environment (Python, PyTorch, CUDA, transformers, GPU); bit-identical CSV across environments would require pinning the full stack and is not claimed.

**Patch 4 (cluster bootstrap).** Original CIs resampled individual rows independently, ignoring the prompt-level correlation structure. Fixed by cluster-bootstrap on per-prompt means (mean-of-prompt-means estimator). For equal-length prompts the row and cluster point estimates coincide; for variable-length future runs the cluster estimator is the correct one.

**Patch 5 (column naming, repetition metrics).** `logit_gap` was misnamed (it was probability-space gap); renamed `prob_gap` and `logit_margin` added separately. Original "looping" diagnostic measured low vocabulary diversity, not generation looping. Replaced with three orthogonal metrics: consecutive-repeat, 4-gram repeat, 8-gram repeat. The five cycle-looped prompts are confirmed by both 4-gram and 8-gram metrics.

All patches were transparently documented in code comments. The patched files in this freeze directory are the exact versions that produced this CSV.

## What the result enables

**Can be claimed externally now:**
- Per-step faithfulness on LLaMA 3.2-3B at α=0.3 with the numbers in the canonical claim.
- Zero confident-position disagreements as a structural property.
- KL monotonicity across the α sweep.
- α=0 identity as a regression-tested invariant.

**Cannot be claimed until M6:**
- Production-realism / coherent free generation.
- Quality-equivalent text output.

**Cannot be claimed until M7:**
- Layer independence.
- Robustness to K and seed variance.

**Cannot be claimed until M8:**
- Cross-model portability / auto-calibration.

## Files in this directory

- `README.md` — this document
- `tasb_m5_faithfulness_20260530_133305.csv` — 3780-row full sweep (9 prompts × 60 tokens × 7 α)
- `tasb_m5_faithfulness.py` — exact code version that produced the CSV
- `tasb_m5_recut.py` — exact recut analysis script
- `m5_console.txt` — captured aggregate console output (manual capture)
- `m5_recut_console.txt` — captured recut console output (manual capture)
- `environment.txt` — pip freeze + CUDA / GPU / transformers version notes

## Reproducibility

```bash
cd ~/.lightning_studio/TASB_Refactor
# With this directory's scripts and the same environment:
python tasb_m5_faithfulness.py  # ~20 minutes
python tasb_m5_recut.py results/tasb_m5_faithfulness_<new_timestamp>.csv
```

Given the same Python / PyTorch / CUDA / transformers / GPU stack, the sampler draws are deterministic via `zlib.crc32` seeding, and results should be numerically identical. Different stack → same statistical pattern, not bit-identical numbers. This is documented in the patch notes inside `tasb_m5_faithfulness.py`.

---

**Frozen 2026-05-30. Do not edit. Reference only.**
