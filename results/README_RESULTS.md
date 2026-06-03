# TASB Results

Summary artifacts from the TASB milestone validation runs. Row-level CSVs (thousands of rows per run) are excluded by `.gitignore`. Only aggregated summary files are committed here.

---

## tasb_m7_multilayer_20260603_000053.csv

**Closes: M7-5 (multi-layer composition)**

Stage 1 protocol run at alpha=0.3, K=10, seed=42 across five layer configurations: C1 [L18], C2 [L18,L24], C3 [L18,L21,L24], C4 [L15,L18,L21,L24,L27], C5 [L18,L19,L20]. Four prompts, 40 steps each (85 evaluable steps total). Proves that open-loop multi-layer composition is bounded at production alpha: zero confident-bucket flips across all configs, top-1 agreement 100% at C1/C2 and 98.82% at C3/C4/C5, KL growing monotonically from 0.00138 (C1) to 0.00515 (C4). All disagreements fall in the AMBIGUOUS bucket (prob_gap < 0.1). To reproduce: `python tasb_m7_multilayer.py` with the default config.

## tasb_m7_scaling_summary_20260603_034851.csv

**Closes: M7-5 extended scaling sweep**

Aggregated scaling sweep over 8 multi-layer configurations (S1–S8), ranging from 1 layer to 10 layers. Tracks top-1 agreement, mean KL, and flip counts by bucket across the full layer-count axis. Confirms the canonical result holds past the C1–C5 protocol: zero confident or moderate flips at any configuration through S6 (10 layers). KL growth slows at higher layer counts, consistent with overlapping receptive fields. To reproduce: `python tasb_m7_scaling.py`.

## tasb_2d_sweep_summary_20260603_113303.csv

**Closes: M7 full 2D characterization**

Comprehensive sweep over the (layer_config × alpha) space. Tests 8+ layer configurations at alpha values from 0.0 through 1.0. Provides the full operating envelope characterization for the bridge: at every (config, alpha) pair, records top-1 agreement, mean KL, JS divergence, mean prob_gap, and flip counts by bucket. Alpha=0 rows validate bit-exact identity (KL=0.0, 0 flips, 100% agreement) across all configs. KL grows smoothly with both alpha and layer count. Zero confident-bucket flips anywhere in the sweep. To reproduce: `python tasb_2d_sweep.py`.

## tasb_m6_seedsweep_summary_20260531_162823.csv

**Closes: M6 production-realism (seed sweep)**

Seed variance sweep under top-p sampling (temp=0.8, top_p=0.9) across 12 random seeds at L18, alpha=0.3, K=10. Establishes that bridge trajectory divergence at alpha=0.3 is statistically indistinguishable from vanilla-vs-vanilla RNG variance under top-p. Mean divergence step 1.2–1.9, agreement 2.0–4.2%, loop rate 0% across all seeds. Key finding: top-p sampling is the dominant source of trajectory divergence; the bridge's contribution is within the noise floor. To reproduce: `python tasb_m6_seedsweep.py`.

## tasb_m7_seed_position_matrix_20260531_224303.csv

**Closes: M7-3 (seed variance)**

Position-level agreement matrix across 12 seeds at L18, alpha=0.3, K=10. Each entry records whether all 12 seeds agreed on top-1 at that (prompt, step) position. Proves bimodal structure: 95.6% of positions are unanimous across all seeds; the 4.4% non-unanimous positions fall exclusively in the AMBIGUOUS bucket (prob_gap < 0.1). Zero non-unanimous positions in the MODERATE or CONFIDENT buckets. To reproduce: `python tasb_m7_seed_variance.py` with the default 12-seed config.
