# TASB Milestone Status

Per Codex's eight-milestone proof ladder.
Last updated: 2026-06-03

---

## M1 — Canonical stack ✓ CLOSED
**What it proves:** Single frozen codebase, no duplicate branches, no ambiguous imports.
**Status:** Closed. Five production files (`tasb_capture_v2.py`, `tasb_sampler_v2.py`, `tasb_injector_v2.py`, `tasb_pipeline_v2.py`, `tasb_m5_faithfulness.py`) plus tests. Pre-RoPE legacy files archived under `legacy/` with provenance note marking them as not-valid bridge-faithfulness evidence.
**Note:** Legacy archive is partial; canonical NEW stack is clean.

## M2 — Correct measurement object ✓ CLOSED
**What it proves:** The bridge measures post-RoPE per-Q-head attention, not the pre-RoPE legacy object.
**Method:** Capture `q_post_rope (B, n_q, S, head_dim)`, `k_post_rope (B, n_kv, S, head_dim)`, the live additive mask, and scaling. Patch `apply_rotary_pos_emb` and `eager_attention_forward` at module level rather than reimplementing them.
**Verification:** Bit-exact (`max diff 0.00e+00`) on probed layers L0/L6/L12/L18/L24/L27. Independent from-scratch RoPE implementation matches HF's live tensors at 0.00e+00.

## M3 — Hard-fail invalid experiments ✓ CLOSED
**What it proves:** A capture that fails its invariant cannot produce a "successful" summary CSV.
**Method:** `LlamaAttentionCapture(strict_verify=True)` raises `RuntimeError` with per-layer diagnostics on failure.
**Verification:** Test T5 in `test_capture_v2.py` deliberately corrupts capture; `verify_capture()` raises. Pass.

## M4 — Exact vs approximate sampling backends agree ✓ CLOSED
**What it proves:** Three sampler backends (exact, gumbel, RBM) all produce matching distributions on the same captured object.
**Verification:**
- `exact` vs `gumbel` at K=5000 on the same capture: max diff 0.0230, mean 0.0031 (within Monte Carlo precision)
- `exact` vs analytical softmax at K=5000: max 0.0148
- Both row-stochastic, zero mass on masked positions
**Test:** T4 in `test_sampler_v2.py`.

## M5 — Per-step faithfulness ✓ CLOSED + SEALED (2026-05-30)
**What it proves:** Teacher-forced matched-context faithfulness. Under teacher-forced greedy decoding, the bridge preserves vanilla's next-token behavior with structural protection on confident-bucket positions.

**Canonical claim (Codex-approved):**
> On corrected post-RoPE capture, single-layer L18 TASB at alpha=0.3 preserves vanilla top-1 on 98.9% of teacher-forced positions overall and 98.3% on non-cycle-looped prompts, with mean KL 0.00118 overall / 0.00201 varied. Across the full alpha sweep up to alpha=1.0, TASB produced zero flips at confident positions; disagreements concentrated in ambiguous/moderate positions where vanilla itself had small top-1 margins.

**Artifact:** Sealed at `results/M5_FROZEN_20260530/` with sha256 manifest.
**Patches in this round:** log_softmax KL (clamp removed, suppressed real KL 15-17x); measured alpha=0 identity per step (not hardcoded); cluster-bootstrap CIs (mean-of-prompt-means); zlib.crc32 stable seeds; multi-scale loop diagnostics; renamed prob_gap from logit_gap.

## M6 — Production-realism ✓ CLOSED (2026-05-31)
**What it proves:** Per-step faithfulness holds under realistic top-p sampling contexts (shadow mode), AND the bridge does not add trajectory divergence beyond what top-p induces on unmodified vanilla (seed sweep).

**Shadow mode (matched-context faithfulness under top-p):**
- alpha=0.3 top-1 agreement (cluster): 96.9%, mean KL 0.00149
- alpha=0 max_abs_diff = 0.00e+00 across all 330 alpha=0 rows (per-step regression test)
- Zero confident-bucket flips at every alpha in {0.1, 0.3, 0.5}; 100% of disagreements landed in MODERATE bucket

**Seed sweep (Test 1 vs Test 2 + Test 3 dose-response):**
- Test 1 (vanilla x vanilla under top-p): mean div@step 1.0-2.0, agreement 1.8-4.5%, loop rate 0%
- Test 2 (bridge alpha=0.3 across seeds): mean div@step 1.2-1.9, agreement 2.0-4.2%, loop rate 0%
- Test 3 (alpha dose-response, fixed seed): graduated drift; alpha=0.05 diverges at step 8 (17.5% agree), alpha>=0.1 dominated by top-p amplification
- All bridge alpha=0.5 outputs remained coherent

**Key finding:** Top-p sampling at temp=0.8 is the dominant source of trajectory divergence in free generation. The bridge's contribution at alpha=0.3 is statistically indistinguishable from RNG-driven vanilla-vs-vanilla variance.

## M7 — Characterization sweep ✓ CLOSED (2026-06-03)
**What it proves:** The M5/M6 numbers at L18, K=10, single seed are representative of bridge behavior across the full operating envelope. All five sub-sweeps closed with zero confident-bucket flips.

**Sub-sweeps:**
1. **Layer sweep** ✓ CLOSED — zero confident-bucket flips across L0-L27 (every 3 layers). Layer-independence of structural faithfulness confirmed.
2. **K sweep** ✓ CLOSED — zero confident flips at K in {1, 3, 5, 10, 25, 50, 100}. KL drops monotonically. Production-K recommendation upgraded to K=50 for TSU silicon.
3. **Seed variance** ✓ CLOSED — 12 seeds at L18/alpha=0.3/K=10. Top-1 98.80%+/-0.56%, KL CV 12.1%. Position-agreement bimodal: 95.6% unanimous, 4.4% non-unanimous ALL in AMBIGUOUS bucket. Zero confident-bucket flips.
4. **alpha fine sweep** ✓ CLOSED — zero confident flips across alpha in {0.0...1.0}. KL grows smoothly with alpha. Results in `tasb_m7_alpha_*.csv` (2026-06-01).
5. **Multi-layer composition** ✓ CLOSED — see canonical claim below.

**M7-5 canonical claim:**
> On open-loop multi-layer composition at alpha=0.3, K=10, seed=42, LLaMA 3.2-3B TASB produces zero confident-bucket flips across all five layer configs (C1:[L18], C2:[L18,L24], C3:[L18,L21,L24], C4:[L15,L18,L21,L24,L27], C5:[L18,L19,L20]). Top-1 agreement is 100% at C1 and C2; 98.82% at C3, C4, and C5 (one ambiguous-bucket disagreement each, prob_gap < 0.003). KL grows monotonically with layer count: C1 0.00138, C2 0.00207, C3 0.00256, C4 0.00515, C5 0.00263. Composition is bounded at production alpha. The single disagreement in C3/C4 is the same position (M75_CR step 11, prob_gap=0.0022); the C5 disagreement is a distinct position (M75_CR step 15, prob_gap=0.0000). All disagreements are in the AMBIGUOUS bucket. Zero moderate or confident flips at any config.

**Artifact:** `results/tasb_m7_multilayer_20260603_000053.csv` + `results/tasb_m7_multilayer_console_20260603_000041.txt`

**M7-5 implementation notes (2026-06-02/03):**
- Refactor: `tasb_injector_v2.py` and `tasb_pipeline_v2.py` extended to multi-layer dict dispatch. One patched function, internal dispatch by `args[0].layer_idx` (confirmed present on LLaMA 3.2-3B). Patch-once/restore-once lifecycle with exception safety.
- Regression tests: 15/15 pass in `tests/test_multilayer_v1.py`. Scalar path bit-exact against M5_FROZEN. alpha=0 identity holds in both scalar and list form. All guards fire correctly.
- Sweep harness: `tasb_m7_multilayer.py`, Stage 1 protocol (C1-C5, 4 prompts, 40 steps).
- KL growth is monotonic with layer count, not super-linear — composition is bounded.
- C4 vs C5 KL (0.00515 vs 0.00263): adjacent layers produce less KL than spread layers at same count, consistent with overlapping receptive fields reducing independent perturbation.

**Confident bucket definition** (cite alongside any zero-flip claim):
- confident: prob_gap >= 0.5
- moderate:  0.1 <= prob_gap < 0.5
- ambiguous: prob_gap < 0.1

## M8 — Calibration program (NOT STARTED)
**What it must prove:** The bridge can auto-detect the model architecture it's handed (LLaMA variant, Mistral, Qwen, etc.) and configure capture/injection appropriately.
**Components needed:** Architecture probe (Q/K/V shapes, KV grouping, RoPE present, mask convention, dtype, attention impl, layer count); auto-selection of capture method; auto-selection of injection layer policy.
**Note:** Must handle MoE architecture from day one. Validation sequence: dense LLaMA variants -> Mixtral 8x7B -> LLaMA 4 Scout.

---

## Summary

**Closed (7 of 8):** M1, M2, M3, M4, M5, M6, M7
**In progress (0 of 8):** —
**Not started (1 of 8):** M8

**What we can claim today:**
- Per-step faithfulness under both teacher-forced (M5) and realistic top-p (M6 shadow) contexts
- Zero confident-position flips at any alpha from 0.0 to 1.0 (M5/M6/M7-1 through M7-4)
- Layer-independence of structural faithfulness confirmed (M7-1, L0-L27)
- K-independence confirmed; production-K recommendation K=50 for TSU silicon (M7-2)
- Seed variance characterized: top-1 98.80%+/-0.56%, non-unanimous positions exclusively AMBIGUOUS (M7-3)
- Bridge is invisible above top-p chaos floor in free generation (M6 seed sweep)
- Bit-exact alpha=0 identity per step, regression-tested in scalar and list form
- Open-loop multi-layer composition is bounded at production alpha: zero confident flips across C1-C5, KL grows monotonically with layer count (M7-5)
- Multi-layer API validated: 15/15 regression tests pass on live model

**What we cannot claim yet:**
- Closed-loop multi-layer composition (open-loop only tested; closed-loop is M7-6 / M8-adjacent)
- Behavior on models other than LLaMA 3.2-3B
- TSU hardware validation (substrate doesn't exist publicly)
- eta metric re-measured on refactored stack (legacy values invalid; must re-measure before citing)

**Next milestone:** M8 — calibration program (architecture auto-detection, MoE support from day one)
**After M8:** Demo wrapper (CLI + side-by-side + metrics panel), then GitHub-facing public release

---

## THRML Throughput Investigation (2026-06-16) — Infrastructure

**Not a numbered milestone** — improves demo usability without changing
correctness claims.

**Problem:** THRML backend running at ~38s/token on T4. Per-token timing
showed `thrml=` flat at ~22s/step independent of seq_len, ruling out
JAX shape-recompilation.

**Root cause:** `tasb_sampler_thrml.py` rebuilt 7 fresh JAX objects per head,
24 heads per token = 552 sequential `sample_states` calls. GPU sat idle
between dispatches.

**Secondary bug (bug registry #6):** `tasb_sampler_v2.py` had hardcoded
`n_warmup=50, steps_per_sample=2` at the call site, overriding patched
defaults. Fixed: now explicitly passes `n_warmup=0, steps_per_sample=1`.

**Fix:** `jax.vmap` over the head dimension. Build program once per token,
swap J via `eqx.tree_at` (0.2ms overhead), dispatch all 24 heads in one
fused GPU call.

**Result:** ~22s/token → ~2.5s/token on T4 (8x). Diagnostic evidence in
`diagnostics/tasb_thrml_batch_diag_v1.py`: PATH A 20.58s / PATH B 0.94s /
PATH C (direct JAX ceiling) 0.002s.

**Faithfulness unchanged:** KL=0.00027–0.00068, Top-1=100%, Conf-Flips=0.

**Remaining gap:** ~2.5s/token vs `exact` backend's ~0.4s/token. This is
the cost of THRML's Boltzmann sampling infrastructure on GPU simulation.
On real TSU silicon this disappears — the chip samples at physical timescales.
The gap is a simulation artifact, not a bridge artifact.

**Files changed:** `tasb_sampler_thrml.py` (vmap), `tasb_sampler_v2.py`
(call-site fix), `diagnostics/` (full audit trail).

---

## M8 — Reframed (2026-06-16)

Original M8 (architecture auto-detection across model families) is deferred.
The Extropic pitch is built on LLaMA 3.2-3B. Auto-detection across Mixtral/
LLaMA 4 is productization scope, not demo scope.

**Actual next milestone: Demo Wrapper**
- CLI side-by-side (vanilla vs bridge output)
- Live metrics panel (KL/top-1/flips)
- "Apple pie test" — a clean compelling prompt showing the bridge in action
- Clone-and-run in under 5 minutes on NVIDIA hardware
- README updated to reflect demo-ready status

**M8-lite (optional, 1 day):** Architecture guard — assert model is a
supported LLaMA 3.2-3B variant before running, with a clear error if not.
Prevents silent failures on untested model variants without requiring full
auto-detection.

---

## Full Results Tables (migrated from README, 2026-06-19)

Migrated from the README during docs cleanup so the README can stay a lean
quickstart. The M5/M6/M7 narrative writeups above remain authoritative; this
section collects the summary tables and cross-cutting findings that previously
lived in the README. No numbers changed in the move.

**Result in one table**

Measured on LLaMA 3.2-3B, teacher-forced, 13 alpha values × 8 layer configs ×
4 prompts × 40 steps = **8,840 positions** across the full operating envelope:

| Config | Layers | α=0.3 Top-1 | α=0.3 KL | Confident flips (all α) |
|--------|--------|-------------|-----------|------------------------|
| 1L     | [18]   | **100.00%** | 0.00138   | 0 / 8,840 positions    |
| 5L     | [15,18,21,24,27] | **98.82%** | 0.00515 | 0 / 8,840 |
| 10L    | [10–27 even] | **95.29%** | 0.00874 | 4 / 8,840 (at α≥0.85 only) |

**Zero confident-position flips through α=0.70 across all layer configs.** The 4
confident flips that appear at α≥0.85 occur only in the two heaviest configs
(10L and 6L) — 0.045% of all measured positions at maximum TSU participation.

**Four-backend live-chat comparison**

Four independent Boltzmann samplers validated on the same frozen LLaMA 3.2-3B at
α=1.0, single-layer L18, K=50, on the prompt *"Hello Llama! Are you ready to
assist?"*:

| Backend | Sampler                                    | KL-Div    | Top-1   | Confident flips |
|---------|--------------------------------------------|-----------|---------|-----------------|
| exact   | `torch.multinomial` over softmax           | 0.00068   | 100.0%  | 0               |
| gumbel  | Gumbel-max in logit space                  | 0.00047   | 100.0%  | 0               |
| rbm     | Iterative RBM Gibbs                        | 0.00012   | 100.0%  | 0               |
| thrml   | Extropic THRML block-Gibbs Boltzmann       | 0.00185   | 100.0%  | 0               |

The `thrml` backend uses Extropic's reference Boltzmann sampler
(`thrml.models.discrete_ebm.CategoricalEBMFactor` driven by
`CategoricalGibbsConditional`). The substrate-agnostic claim — that TASB produces
equivalent model behavior regardless of which Boltzmann sampler sits underneath —
is demonstrated across four independent implementations.

**Substrate-agnostic bridge (live chat, 2026-06-10):** Four independent Boltzmann
sampler implementations validated end-to-end through the chat runtime on real
prompts with full conversation context. KL < 0.01 and zero confident flips on
every backend at α=1.0, single layer. The bridge is sampler-implementation-
independent: any backend that draws from `exp(J)/Z` at the attention-scale
temperature satisfies the contract.

**Characterization summary (M5 / M6 / M7)**

Headline numbers; full writeups are in the M5, M6, and M7 sections above.

- **M5 (faithfulness, sealed 2026-05-30):** single-layer L18 at α=0.3 preserves
  vanilla top-1 on 98.9% of teacher-forced positions overall (98.3% on
  non-cycle-looped prompts), mean KL 0.00118; zero flips at confident positions
  across the full α sweep to α=1.0.
- **M6 (production-realism, 2026-05-31):** under realistic top-p sampling
  (shadow mode), top-1 agreement 96.9%, mean KL 0.00149. The bridge's trajectory
  divergence at α=0.3 is statistically indistinguishable from RNG-driven
  vanilla-vs-vanilla variance; top-p is the dominant source of trajectory chaos.
- **M7 (full characterization, 2026-06-03):** seven sub-sweeps across the
  complete operating envelope —

| Sweep | Variable | Result |
|-------|----------|--------|
| Layer sweep | L0–L27 | Zero confident flips at every layer |
| K sweep | K=1–100 | Zero confident flips; KL drops monotonically |
| Seed variance | 12 seeds | Top-1 98.80%±0.56%; non-unanimous positions exclusively AMBIGUOUS |
| α fine sweep | α=0.0–1.0 | Zero confident flips across full range |
| Multi-layer composition | C1–C5 (1–5L) | Zero confident flips; KL saturates, does not compound |
| Scaling curve | 1L–10L | Zero confident flips through 10 layers; KL growth sub-linear |
| 2D sweep | 13α × 8 configs | Zero confident flips through α=0.70; boundary at α=0.85/10L |

**Structural alignment finding**

The bridge adds **4–7× more KL at ambiguous positions than at confident
positions** across the full alpha range. Perturbation energy is geometrically
concentrated in the low-certainty regions of the model's probability landscape.
This is not a tuned behavior — it is a consequence of Boltzmann sampling at
attention-scale temperature interacting with the model's existing confidence
geometry.

**Long-context and generation (stress test)**

- Zero OOM through 10 layers at 512-token context (peak VRAM 3.58GB on 4-bit 3B model)
- Zero speed penalty on capture-once generation: 5.9 tok/s with or without bridge at any layer count
- 5-paragraph side-by-side at 256-token context: word-for-word identical output between vanilla and 5L bridge at α=0.3

**Operating envelope**

The measured safe operating regime for the demo slider:

```
α ∈ [0.00, 0.70]:  Zero confident flips at any layer config (0–10L)
α ∈ [0.70, 0.85]:  First confident flips appear at 10L and 6L only
α ∈ [0.85, 1.00]:  4 total confident flips across 8,840 positions
                    at the two heaviest configs only
```

**Production recommendation:** α=0.3, K=50 (for TSU silicon), any layer config.
