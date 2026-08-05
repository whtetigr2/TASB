# QUARANTINED — Early Development Results, 2026-07-16

**Status: EARLY TESTING AND DEVELOPMENT. Not a validated result. Not citable.**

**Quarantined 2026-08-05**, moved out of `validation/` entirely — that tree implies
validation, and these do not validate what their filenames suggest.

### The precise nature of the problem

These numbers are **not wrong**. They are a **correct measurement of the wrong thing**.

The sampler genuinely does draw from softmax(logits) — established here cleanly, by two
independent diagnostics. What does not follow is the inference that this validates a
*thermodynamic* bridge. The graph being sampled has no couplings, so the draws are IID
categorical, and IID categorical draws pass a goodness-of-fit test against softmax
trivially.

| | status |
|---|---|
| the measurements | sound |
| the sampler's correctness | genuinely established |
| "therefore the TSU bridge is faithful" | **does not follow** |

Nothing here needs discarding or re-measuring for its own sake. What needs replacing is
the claim attached to it.

---

These three CSVs were recovered from an untracked `results/` directory on a Lightning
studio on 2026-08-05, immediately before that studio was retired. They were never
committed, and they are preserved here for provenance rather than because they support a
conclusion.

Everything below distinguishes what was **measured** from what it **establishes** — which
are not the same thing, and the gap is the whole reason this document exists.

---

## The runs

| file | S | K | positions tested | χ² pass | pass frac | mean p | mean KL | mean TV | elapsed |
|---|---|---|---|---|---|---|---|---|---|
| `…152548.csv` | 256 | 5,000 | 6,048 | 5,703 | 0.9430 | 0.4929 | 0.019879 | 0.053849 | 3.86 s |
| `…152721.csv` | 512 | 1,200 | 12,192 | 11,523 | 0.9451 | 0.4926 | **0.692910** | 0.154631 | 4.82 s |
| `…160835.csv` | 256 | 10,000 | 6,048 | 5,725 | 0.9466 | 0.4962 | 0.007484 | 0.038039 | 3.10 s |
| `…160835.csv` | 512 | 10,000 | 12,192 | 11,596 | 0.9511 | 0.4975 | 0.019365 | 0.053489 | 3.07 s |

`S` = sequence length, `K` = samples drawn per attention row.

---

## What these numbers actually show

**The sampler draws from the distribution it claims to draw from.** Two independent
signals agree on this, and both are the right diagnostics:

1. **Mean p-value ≈ 0.49–0.50 in every run.** Under a correctly specified null, χ²
   p-values are uniform on [0,1], so the mean should sit at 0.5. Observed: 0.4926,
   0.4929, 0.4962, 0.4975. That is the signature of a null that genuinely holds, and it
   is a considerably stronger check than the pass rate alone — a sampler that was subtly
   wrong would skew this mean even while many individual tests still passed.

2. **Pass fraction ≈ 0.943–0.951 against an expected 0.95** at α = 0.05. Four runs
   landing that close to nominal is consistent with correct calibration.

**KL and TV tighten with more samples, as they must.** At S = 256, KL falls 0.0199 →
0.0075 going from K = 5,000 to K = 10,000. At S = 512, KL falls 0.6929 → 0.0194 going
from K = 1,200 to K = 10,000. Monotone improvement with sample count is the expected
behaviour of an empirical estimator and is a (weak) structural check in its own right.

---

## What these numbers do NOT show

This is the important half.

**They do not test detailed balance, despite the filename.** Detailed balance is a
property of a transition kernel — π_i P_ij = π_j P_ji. What was measured is a
goodness-of-fit test of the *stationary/empirical distribution* against softmax. Those are
different claims, and the second does not imply the first. The naming should be corrected
to `goodness_of_fit` before any of this is shown to a reviewer; a physicist will catch it
immediately and it undermines otherwise sound work.

**They do not establish TSU-faithfulness.** A separate finding, recorded 2026-08-05,
determined that the THRML graph constructed in `backends/thrml.py` uses
`CategoricalEBMFactor` with a **single** node group. Per THRML's own base class
(`DiscreteEBMFactor`), the energy term is then `W[c_i]` — unary, with no couplings between
nodes. The joint distribution factorises completely, and consecutive samples are
independent by construction.

That matters here because **an uncoupled categorical sampler passes a χ² test against
softmax trivially.** These runs confirm the sampler is correct; they say nothing about
whether the sampling *mechanism* resembles what a thermodynamic sampling unit does. The
distribution was never the part in doubt.

**They do not cover the production configuration end-to-end.** These are standalone
sampler tests, not full bridge-injection runs against a model.

---

## Anomaly worth chasing

**Run 2 (S = 512, K = 1,200): mean KL = 0.692910.** That is an order of magnitude above
every other run, and it sits within 3×10⁻⁴ of ln 2 ≈ 0.693147.

The benign explanation is severe undersampling: 1,200 draws spread over 512 categories is
roughly 2.3 samples per category, so the empirical distribution is sparse and its KL
against a dense target inflates sharply — while χ² still passes, because χ² accounts for
expected counts. The proximity to ln 2 is then coincidence.

That explanation is plausible and untested. It should be checked rather than assumed,
because a KL pinned at exactly ln 2 is also what you would see from a distribution
collapsed onto half its support. A K-sweep at S = 512 across K ∈ {1200, 2400, 4800, 9600}
would separate the two: genuine undersampling decays smoothly with K, an artefact does not.

---

## Provenance

- **Origin:** untracked `results/` on Lightning studio, commit `9781265` era
- **Recovered:** 2026-08-05, before studio retirement
- **Generating script:** not committed alongside the outputs — the exact code path that
  produced these is not currently reconstructable from the repo, which is itself a reason
  to treat them as development artefacts rather than evidence
- **Hardware:** Lightning studio, GPU configuration unrecorded

## If these are ever to become citable

1. Rename the metric to goodness-of-fit, or implement an actual detailed-balance check
2. Commit the generating script alongside the outputs so runs are reproducible
3. Record the environment — package versions, GPU, seed
4. Resolve the ln 2 anomaly with the K-sweep above
5. Re-run against a coupled graph, once one exists, so the test measures something the
   distribution check cannot already answer for free
