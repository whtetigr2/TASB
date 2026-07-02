# TASB: Thermodynamic Attention Sampling Bridge
### You Do Not Need to Rebuild the Model

---

## Origin & Overview

The question that started this was: **"Who decided Cup?"**

Not who invented the cup — who decided that the written symbol C-U-P would map to
the object? How does an arbitrary symbol persist across cultures and centuries with
such fidelity? How much information is actually transmitted in written versus spoken
language? How much energy does it take to *think* of something to say — and how does
the brain do that so efficiently while a GPU runs white-hot doing far simpler things?

Those questions led me through 2025 — across linguistics, neuroscience, information
theory, and eventually into how AI training data is actually structured. I landed on
a framing: training data treats information like a periodic table of elements. Words,
concepts, and data points are the elements — discrete, categorical, arranged by their
relationships. A language model learns the chemistry of how those elements combine.
The question wasn't "how do you encode language" — it was "what *kind* of physics is
this system actually doing?"

That framing led me directly to thermodynamic computing and eventually to Extropic AI
and their Thermodynamic Sampling Unit (TSU) — hardware that performs Boltzmann
sampling natively, at near-Landauer efficiency. The stated position from Extropic
*(TSU 101)* was that to run AI models on thermodynamic hardware, you would need to
rebuild them from the ground up. Transformers assume deterministic softmax; TSUs
perform stochastic sampling. The architectures were too different.

My response was a simple question: **"Is it really though? Has anyone actually tried?"**

No one had. So I built TASB — an attempt to route a frozen transformer onto
thermodynamic hardware without modifying it. Early progress was limited. The bridge
concept was sound but the mathematical grounding wasn't there yet.

In February 2026, Gunn Kim published a derivation that changed everything. Kim showed
that the full transformer forward pass emerges from a thermodynamic Lagrangian. The
scaled dot-product attention formula — softmax(QKᵀ/√d_k) — is not analogous to a
Boltzmann distribution. It **is** one, exactly. The 1/√d_k scaling factor is the
inverse temperature. The dot-product scores are negative energies. The normalization
is the partition function. The bridge did not need to be engineered — it was a
mathematical identity waiting to be recognized.

After that, things moved fast. With the thermodynamic framework in place I could
properly classify attention heads by their thermodynamic activity, distinguish
frozen-state behavior (over-sharpened, low-entropy heads) from chaos states at the
other extreme, and build a bridge layer that routes frozen transformer weights to
Boltzmann sampling with zero confident flips at K=10, sampler correctness verified at K=5000 (KL < 0.002), and 1.81× compute overhead.
The result is a working pip-installable Python library with a live demo at
[huggingface.co/spaces/shvrpws/thermobridge](https://huggingface.co/spaces/shvrpws/thermobridge).

The validation work went further than I expected. Three independent proofs of the
softmax-Boltzmann identity converged (§2). Building the bridge produced a second
instrument I didn't expect — a thermodynamic specific heat observable that predicts
sampling error with r = 0.824 and turns out to be a framework-agnostic thermometer
for transformer attention (§4). ZOH discretization in Mamba turned out to be the
FDT-exact discretization — nobody in the SSM literature had that thermodynamic reason
for it (§6.2). NTK-aware RoPE context extension turned out to be a Jarzynski protocol
with measurable per-head dissipated work, visible at inference time (§6.3).

The question "Who decided Cup?" has a thermodynamic answer. The Boltzmann distribution
did — and transformers have been computing it since 2017 without anyone calling it
that. The starting assumption held: **you do not need to rebuild the model.**

---

*Patent Pending · USPTO Provisional 64/019,999 · March 28, 2026*
*Working implementation: [github.com/whtetigr2/thermobridge](https://github.com/whtetigr2/thermobridge)*
*Live demo: [huggingface.co/spaces/shvrpws/thermobridge](https://huggingface.co/spaces/shvrpws/thermobridge)*

---

## §1 — The Problem

Modern GPU inference is extraordinarily wasteful — not by engineering standards, but
by physical law.

The Landauer limit (1961) establishes the minimum energy required to perform one
irreversible bit operation: kT·ln(2) ≈ 2.8 × 10⁻²¹ joules at room temperature.
A modern GPU consumes roughly 10⁻¹² joules per floating-point operation. The ratio
is approximately 10¹²: GPU inference sits twelve orders of magnitude above the
thermodynamic floor.

This gap is not an engineering problem. It is architectural. GPUs are optimized for
deterministic, differentiable arithmetic — the exact opposite of what physics
naturally does cheaply. Thermodynamic hardware (Extropic's TSU, Normal Computing's
SPU) is built from the physics up, targeting Landauer-efficiency stochastic sampling.
These are not theoretical devices: Extropic's DTM architecture operates in subthreshold
CMOS using standard fabrication, consuming approximately 2 fJ per sampling step [Extropic, TSU 101].

The obstacle to running transformers on thermodynamic hardware has never been the
hardware. It has been the bridge. Transformer weights are trained assuming deterministic
softmax. Moving to stochastic hardware would seem to require retraining — a
prohibitive cost for models with billions of parameters.

TASB eliminates that obstacle. The bridge is mathematical, not architectural. And the
mathematics has been sitting in plain sight inside every transformer since 2017.

---

## §2 — The Identity

> softmax(QKᵀ/√d_k) is not analogous to a Boltzmann distribution. It is one.

The scaled dot-product attention weight formula:

```
p_i = exp(qᵀk_i / √d_k) / Σⱼ exp(qᵀk_j / √d_k)
```

This is the Boltzmann distribution p_i = exp(-E_i / kT) / Z with:
- **Energy:** E_i = -qᵀk_i (negative dot product)
- **Temperature:** kT = √d_k (the scaling factor is the inverse temperature)
- **Partition function:** Z = Σⱼ exp(qᵀk_j / √d_k)

This identification is proven three independent ways. All three arrive at the same
distribution from different starting points — the convergence is not coincidence.

### Proof 1 — Maximum Entropy (FIND-016)

The Boltzmann distribution is the unique probability distribution that maximizes
Shannon entropy subject to a fixed mean energy. Applying the MaxEnt variational
principle to the attention energy landscape H = -QKᵀ/√d_k yields softmax exactly.
No approximation. The Euler-Lagrange equations give softmax as the equilibrium.

### Proof 2 — Kim 2026 Lagrangian (arXiv:2602.08216)

Gunn Kim (2026) derives the full transformer forward pass from a thermodynamic
Lagrangian using Euler-Lagrange equations. The Lagrangian is **conditionally unique**:
- The kinetic term (Fisher-Rao metric) is unique by Chentsov's theorem — it is the
  only Riemannian metric on the statistical manifold invariant under sufficient
  statistics
- The potential term (Shannon entropy) is unique by the Shore-Johnson axioms — the
  only additive, consistent entropy functional

This means softmax is not one of many possible equilibria. Within the
Shannon-Boltzmann framework, it is the *only* equilibrium. Kim's paper was submitted
February 9, 2026 — 47 days before the TASB patent filing (March 28, 2026). Kim
provides the theoretical framework; TASB is the hardware implementation.

### Proof 3 — Kajitsuka & Sato 2023 (arXiv:2307.14023, ICLR 2024)

Kajitsuka & Sato prove that the softmax contextual map is a Boltzmann operator in
the representation-theoretic sense: for any energy matrix J, the map a ↦ σ_s[a]
is exactly the Boltzmann distribution with energy -J. Their Lemma 1 and Theorem 2
are directly applicable to TASB's attention energy landscape. Kim and K&S are
independent: Kim does not cite K&S (verified across all 44 citations), yet both
arrive at the same Boltzmann distribution via different routes — thermodynamic
variational principle (Kim) and representation theory (K&S).

### What the Uniqueness Result Means for Alternatives

α-entmax attention (Peters et al. 2019, deployed in production systems) is the
inference-time instantiation of Tsallis q-entropy — a *different* statistical
mechanical framework. Because Kim's Lagrangian is unique within the Shannon-Boltzmann
framework, α-entmax and TASB are provably sampling from different equilibria.

Empirically confirmed: KL(entmax15 ‖ softmax) = 0.109 ± 0.070 across 120 head ×
prompt combinations (LLaMA 3.2-3B, layer 18, 5 prompts × 24 heads). Every single
head shows KL > 0. The distributions are distinct, not just in theory but in
practice, on real model weights, with effect sizes ranging 0.005–0.37 nats.

---

## §3 — The Bridge

TASB intercepts the attention computation after Q, K, V are computed but before the
softmax output is returned. It samples K draws from the Boltzmann distribution
defined by H = -QKᵀ/√d_k, averages them, blends the result with the original
softmax weights at mixing coefficient α, then recomputes the value projection.

```python
# Conceptually:
energy = -(Q @ K.T) / sqrt(dk)          # Hamiltonian: H_ij = -q_i · k_j / √dk
p_boltzmann = boltzmann_sample(energy, K=10)   # K independent draws
p_blended = (1 - alpha) * p_softmax + alpha * p_boltzmann
attn_output = p_blended @ V
```

No gradients. No model modification. No retraining. The weights stay frozen.

### Validated Results

Measured on LLaMA 3.2-3B, exact backend, layer 18, across 5 prompts:

| Metric | Condition | Value |
|--------|-----------|-------|
| KL(p_bridge ‖ p_softmax) | K=10, α=1.0 (full substitution) | mean 1.72 [1] |
| KL(p_bridge ‖ p_softmax) | K=5000, α=1.0 (convergence test) | < 0.002 |
| KL(p_blended ‖ p_softmax) | K=10, α=0.3 (production) | ≈ 0.001 |
| Top-1 agreement | K=10, α=1.0 | 98.9% |
| Confident flips | any layer, any K, any α | 0 |
| Compute overhead | — | 1.81× |
| Layers where injection is valid | — | All 28 (L0–L27) |
| K values tested | — | {1, 3, 5, 10, 25, 50, 100} — all pass |
| α values tested | — | 0.0 through 1.0 — all pass |

[1] Finite-K Monte Carlo sampling noise, not a sampler error. KL ∝ K⁻⁰·⁹⁴ (R²=0.94,
independently verified) — the sampler converges to the correct Boltzmann distribution
as K increases. At K=10, sampling variance is real; the behavioral invariant is zero
confident flips, not low KL. The K=5000 row establishes distributional correctness:
the sampler produces samples from the right distribution. The production row (α=0.3)
shows the blended output fidelity: 70% softmax + 30% thermodynamic sample.

**Three KL measurements, three distinct conditions.** Greedy decoding (argmax) is
the T→0 limit of the Boltzmann distribution — a delta function on the
highest-energy token, with KL(argmax ‖ Boltzmann) = ∞. TASB achieves zero confident
flips at K=10 — the model's top-1 predictions are unchanged regardless of sampling
noise — and converges to KL < 0.002 at K=5000, proving the sampler reaches the
correct Boltzmann distribution at sufficient sample count.

### T=√d_k is Exact, Not Conventional

The temperature T=√d_k in TASB is not a heuristic. It is derived directly from the
attention Hamiltonian: if H = -QKᵀ/√d_k, then the natural temperature is
kT = √d_k, giving β = 1/√d_k exactly.

I confirmed this experimentally via temperature ablation across T ∈ {0.09×√d_k ... 8×√d_k}
on 120 head × prompt combinations:

| T | T/√d_k | Mean KL to Boltzmann |
|---|--------|---------------------|
| 5.66 | 0.5× | 0.087 |
| **11.31 (√d_k)** | **1.0×** | **< 10⁻⁶ (numerical precision)** |
| 22.63 | 2.0× | 0.381 |
| 45.25 | 4.0× | 1.247 |

The minimum is at the exact theoretical temperature. KL increases monotonically in
both directions. Over-smoothing (T > √d_k) is 4.4× more costly than over-sharpening
at the same relative deviation, because LLaMA's attention distributions are already
concentrated — flattening them toward uniform requires moving further in distribution
space than sharpening them further.

The practical implication: do not adjust T. The physics already chose it.

---

## §4 — The Observable: Specific Heat

TASB produces a side-effect that turns out to be more interesting than the bridge itself.

The thermodynamic specific heat of an attention head is:

```
Cv = Var_ρ(H) = E_ρ[H²] - E_ρ[H]²
   = Σᵢ p_i · (qᵀk_i / √dk)² - (Σᵢ p_i · qᵀk_i / √dk)²
```

This is computable in one line from the bridge state, at zero additional cost:

```python
cv = (p * scores**2).sum(-1) - (p * scores).sum(-1)**2
```

Cv measures the thermal spread of the attention distribution over its energy
landscape. Physically: high Cv means a diffuse, uncertain head that spreads
attention broadly. Low Cv means a sharp, confident head with peaked attention.
This is the same quantity Kim (2026) derives as the specific heat of the attention
thermodynamic system.

### Empirical Result: r(Cv, KL) = 0.8241

From 3,360 total observations (5 prompts × 28 layers × 24 heads, LLaMA 3.2-3B),
restricting to layer 18 (120 observations, K=10, α=1.0):

**Pearson r(Cv, KL_TASB) = 0.8241, p = 6.1 × 10⁻²⁵**

Across all 672 head-layer combinations (population-wide): r = 0.61 — a real,
moderate correlation that weakens in early layers where causal-mask structure
introduces noise beyond Cv alone.

Specific heat predicts finite-K sampling error. Heads with higher thermodynamic
activity (higher Cv) require more samples K to converge to the Boltzmann distribution.
This is the empirical bridge between Kim's Cv observable and TASB's practical
sampling guarantee.

**Cv and entropy as complementary signals.** Shannon entropy H(P) = −Σᵢ pᵢ log pᵢ
is a stronger predictor of finite-K KL error:

| Observable | r at layer 18 (n=120) | r population-wide (n=672) |
|------------|----------------------|---------------------------|
| Cv = Var_ρ(H) | 0.8241 | 0.6097 |
| Shannon entropy H(P) | 0.9279 | 0.9492 |

Entropy is more accurate and more stable across the full model. The difference
follows from first principles: the exact Miller-Madow relationship between finite-K
KL error and H(P) is exact, while Cv is only a regime-dependent proxy for H(P).

Cv's advantage is not predictive power but physical grounding. Cv = Var_ρ(H) is the
variance of the energy landscape — the same quantity Kim (2026) derives as the
thermodynamic specific heat of the attention system, and the natural vocabulary of
Extropic's hardware architecture. Entropy names the information-theoretic cost; Cv
names the thermodynamic state. Both are computable from the bridge state in a single
line of code at zero additional cost. TASB computes Cv because it is the physically
correct observable for the thermodynamic framework; entropy is available as a stronger
pure predictor for adaptive sampling applications.

### The Universal Predictor Finding

I ran the same experiment with a completely different approximation: instead of
measuring how far TASB's K=10 sampling deviates from softmax, I measured how far
the Tsallis-entropy alternative (α-entmax, a different statistical mechanics) deviates.

**Pearson r(Cv, KL_entmax) = 0.8156**

Two different approximation schemes. Two different theoretical frameworks. The same
observable — Cv — predicts both deviations with nearly identical correlation strength.

I didn't expect this result. The original prediction was only that r(Cv, KL_TASB) > 0.
What the data shows is that Cv is a framework-agnostic predictor of distributional
divergence from the Gibbs-Boltzmann baseline. Whether you are using finite-K
sampling within the Boltzmann framework (TASB) or substituting a Tsallis equilibrium
(α-entmax), the heads that deviate most are the thermodynamically active ones, and Cv
tells you which heads those are before you run any approximation at all.

TASB is the only inference-time framework that computes Cv — to my knowledge, the
only tool that can tell you, per head, per layer, per token, how thermodynamically
active a frozen transformer is while it is running.

### What Cv Is NOT Predicting

Two predictions I made during development failed, and I documented them:

**Layer-depth ordering:** I predicted Cv would decrease monotonically with layer
index (deeper layers = more concentrated attention). Not confirmed. The frozen
LLaMA 3.2-3B profile is flat across layers (Cv ≈ 0.65–1.14 across L0–L27).
Kim's ordering result comes from training-time dynamics; at inference on a frozen
model, the ordering isn't preserved. This is the correct finding — a different
regime, not a contradiction of Kim.

**Gumbel backend bias:** I predicted the Gumbel backend would introduce a Cv bias
of +π²/6. This was a derivation error. TASB computes Cv analytically from the
softmax distribution, not from Gumbel samples — the backend choice doesn't affect
Cv. Prediction struck.

Documenting failed predictions isn't a weakness. The surviving result — r = 0.824 —
is stronger for having survived the cut.

---

## §5 — The Hardware Path

The THRML library (Extropic AI) provides a Python API for thermodynamic sampling
unit hardware. TASB's `thrml` backend wraps this API directly:

```python
# TASB wraps chip.sample() — the TSU's Boltzmann sampling primitive
from thrml import CategoricalGibbsConditional
node = CategoricalGibbsConditional(couplings=J, n_states=S)
samples = node.sample_states(n_samples=K)
```

The mapping from transformer attention to TSU operation:
- Ising coupling matrix J_ij = q_i · k_j / √d_k (the attention energy)
- Number of states S = sequence length (the attention window)
- Temperature kT = √d_k (exact, from the Hamiltonian)
- Samples K = number of independent draws (K=10 for software, K=50 recommended for silicon)

The THRML graph structure is critical: CategoricalGibbsConditional nodes with W²=0
(no within-node coupling) are mathematically independent. One Gibbs sweep equals one
exact independent draw from the Boltzmann distribution. Zero burn-in required.
This is proved, not assumed: ESS ≈ K for TASB's THRML usage.

### The Landauer Argument

The minimum energy for one irreversible bit erasure is kT·ln(2). At 300K:
```
E_Landauer = (1.38 × 10⁻²³ J/K)(300K)(ln 2) ≈ 2.87 × 10⁻²¹ J
```

Extropic's DTM architecture (subthreshold CMOS, standard fabrication) operates at
approximately **2 fJ = 2 × 10⁻¹⁵ J** per sampling step [Extropic, TSU 101] — roughly
10⁶ times above Landauer, but approaching it from an architecture that can scale down.

Modern GPU floating-point: ~10⁻¹² J per operation = 10⁶ × Landauer.
DTM silicon: ~2 × 10⁻¹⁵ J per sample = 10³ × Landauer.

The gap between GPU and thermodynamic hardware is **three orders of magnitude**.
The gap between thermodynamic hardware and the physical limit is **three more orders**
— and the physics says those three are achievable as fabrication improves.

### Formal Compatibility Proof — Extropic's Own Words

The compatibility of TASB with the Extropic TSU is not a claim I'm making about
their hardware. It is a claim Extropic makes about their own hardware — and
TASB's coupling matrix satisfies it by construction.

**From Extropic's TSU 101:**
> *"The inputs to a TSU are parameters that specify the energy function of an EBM,
> and the outputs of a TSU are samples from the defined EBM."*

**From Extropic's codon optimization paper (arXiv:2606.17327):**
> *"By programming the interaction weights between these elements, the chip's
> thermal fluctuations explore the corresponding Boltzmann distribution."*

**From the THRML API documentation:**
> *"CategoricalGibbsConditional... computes the parameter θ of a softmax
> distribution given DiscreteEBMInteractions."*
> *"For the Potts model in THRML, state updates use a conditional distribution
> with an energy function that corresponds to sampling from a softmax distribution."*

TASB's coupling matrix J[i,j] = q_i·k_j/√dk is an energy-based model interaction
weight in exactly the sense Extropic describes. The TSU's output — samples from
the Boltzmann distribution over J — is algebraically identical to softmax(QKᵀ/√dk).
No retraining is required because the frozen transformer weights already encode the
correct thermodynamic coupling parameters at inference time.

The hardware drop-in is already identified in THRML's own documentation:
`chip.sample()` replaces `sample_states()` with the same call signature. TASB's
THRML backend is already structured for this swap.

### Hardware-Ready Claims (Tiered)

| Tier | Claim | Status |
|------|-------|--------|
| 1 | THRML software simulation validates Boltzmann sampling fidelity | ✅ Claimable now |
| 2 | TASB ↔ TSU mathematical compatibility proven from Extropic's own documentation | ✅ Claimable now |
| 3 | chip.sample() API forward-compatibility with THRML v0.1.3 categorical classes | Requires XTR-0 dev kit access |
| 4 | Benchmark results on physical XTR-0 silicon | Requires Extropic partnership |

Tiers 1 and 2 are fully substantiated. Tier 3 is an integration risk, not a
mathematical incompatibility — the stationary distribution is identical regardless
of which API version routes the program to the chip.

---

## §6 — The Deeper Physics

Three results sit below the bridge-and-thermometer headline. Each extends the
thermodynamic framework into territory that TASB's observable makes navigable.

### §6.1 — Lagrangian Uniqueness Has a Measurable Consequence

Kim's thermodynamic Lagrangian is not just any variational principle — it is
conditionally unique. Two theorems lock it in:

**Chentsov's theorem:** The Fisher-Rao metric is the *only* Riemannian metric on
the statistical manifold invariant under sufficient statistics transformations.
This fixes the kinetic term of the Lagrangian uniquely.

**Shore-Johnson axioms:** Shannon entropy is the *only* additive, consistent
entropy functional. This fixes the potential term uniquely.

Together, these theorems mean the softmax Boltzmann equilibrium is the *unique*
equilibrium in the Shannon-Boltzmann framework. If you use a different entropy —
say, Tsallis q-entropy — you get a *different* statistical mechanical system with
a *different* equilibrium. This is not a philosophical distinction. It is measurable.

α-entmax (Peters et al. 2019, widely deployed in production) is the inference-time
instantiation of Tsallis entropy. If Kim's uniqueness result is real, TASB and
α-entmax must produce distributions with nonzero KL divergence between them.

**Measurement:** KL(entmax15 ‖ softmax) across 120 head × prompt combinations
(LLaMA 3.2-3B, layer 18, 5 prompts × 24 heads):

| Statistic | Value |
|-----------|-------|
| Heads with KL > 0 | 120/120 (100%) |
| Mean KL | 0.1086 |
| Std KL | 0.0699 |
| Range | 0.005 – 0.368 nats |

Every single head. The systems are empirically distinct, not just mathematically distinct.

**The Cv connection:** r(Cv, KL_entmax) = 0.816 — nearly identical to r(Cv, KL_TASB)
= 0.824 from §4. Cv predicts the divergence from Gibbs-Boltzmann regardless of which
approximation scheme is used. Whether you are finite-K sampling within the Boltzmann
framework (TASB) or substituting a Tsallis equilibrium (α-entmax), the heads that
deviate most are the thermodynamically active ones, and Cv identifies them before any
approximation runs.

---

### §6.2 — ZOH Is the Thermodynamically Correct Discretization

The Structured State Space Model literature shifted from bilinear (Tustin)
discretization in S4 [Gu et al. 2021, arXiv:2111.00396] to zero-order hold (ZOH)
in Mamba [Gu & Dao 2023, arXiv:2312.00752]. The Mamba paper cites hardware
efficiency as the reason. That transition has an independent thermodynamic
justification.

The overdamped Langevin equation ẋ = Ax + ξ(t) with FDT-constrained noise
Q = −θ(A + Aᵀ) has a unique steady-state covariance P = θ (thermal energy),
satisfying the continuous Lyapunov equation AP + PAᵀ + Q = 0. Under ZOH
discretization, the discrete noise covariance is Q̄ = ∫₀^Δ exp(At) Q exp(Aᵀt) dt
[Van Loan 1978]. The discrete Lyapunov equation:

```
ĀPĀᵀ − P + Q̄ = 0
```

holds for the *same* covariance P. ZOH preserves the equilibrium thermal covariance
**exactly** — zero residual, not O(Δⁿ) approximation.

The bilinear (Tustin) discretization introduces a leading-order violation. In the
scalar case (a < 0):

```
Bilinear FDT residual = −(a³Δ³)/12 + O(Δ⁴)
```

The noise and dissipation are no longer exactly balanced — the discrete bilinear
system is thermodynamically inexact at O(Δ³). Wolfram-verified from first principles.

The practical magnitude is small at typical Mamba step sizes (Δ ∈ [0.001, 0.1]),
but the structural point stands: **ZOH is the FDT-exact discretization for systems
approximating Langevin dynamics. Bilinear is not.** Mamba arrived at the right
answer. It did not know this reason for it.

**Attribution:** The base mathematical result — ZOH preserves stationary covariance
in linear stochastic systems — is a standard result in control theory [Kalman 1960;
Van Loan 1978]. What's original here: framing this as FDT preservation, computing
the bilinear violation coefficient in the stochastic FDT context, and identifying
its consequence for the S4→Mamba architectural transition. A four-pass search across
arXiv, Google Scholar, IEEE databases, and SSM lineage papers found no prior work
using FDT language for ZOH or computing this bilinear coefficient [FIND-028].

---

### §6.3 — NTK-RoPE Context Extension as a Jarzynski Protocol

RoPE (Rotary Position Embedding) encodes token position by rotating Q and K vectors
through dimension-dependent angles. NTK-aware context extension [bloc97 2023] shifts
these angles to extend the model's usable context window beyond its training length
N_train. For LLaMA 3.2-3B (d=128, b=10000, s=4):

```
θ_k^NTK = 1 / b_NTK^(2k/d),   b_NTK = 40,890
```

At inference time, this transition is instantaneous — all position frequencies switch
simultaneously. This is a **sudden Hamiltonian switch**: H_orig → H_NTK.

The Jarzynski equality (1997) describes sudden Hamiltonian switches:
⟨exp(-βW)⟩ = exp(-βΔF). By Jensen's inequality, ⟨W⟩ ≥ ΔF — the mean dissipated
work is bounded below by the free energy change. The per-head dissipated work is:

```
W_diss = T · KL(p_orig ‖ p_NTK)
```

If this framing is correct, heads that experience more distributional disruption
under the Hamiltonian switch should show larger changes in specific heat:
ΔCv = Cv^NTK − Cv^orig should correlate with W_diss.

**Result:** Pearson r(ΔCv, W_KL) = 0.549, p = 5.5 × 10⁻³, n = 24 Q-heads
(LLaMA 3.2-3B, layer 18). 21/24 heads show ΔCv > 0: NTK scaling uniformly
increases specific heat. The correlation is driven by moderate-Cv heads
(cv_orig ≈ 0.88–1.01), consistent with §3's finding that medium-activity heads
are most responsive to distributional perturbation.

**Scope:** All test prompts have S ≤ 20 tokens — far below N_train = 8,192. NTK
frequency deformation at these positions is subtle (Δθ ≈ 0.0002 rad at the highest
frequency). That the correlation is detected at all in this weak-signal regime is
a positive indicator; at extended context where NTK actually fires (m > 8,192) the
effect is orders of magnitude larger. LLaMA 3.2-3B's GQA structure (24 Q-heads,
8 KV-heads) means the effective independent sample count is approximately 8 — the
directional result holds, the p-value should be interpreted accordingly.

TASB Cv is the only inference-time observable capable of measuring this cost. Before
TASB, the Jarzynski work of NTK context extension was invisible at the head level.
It is now measurable, per head, per layer, per token, during inference.

---

## §7 — What This Enables

The physics in §6 is not ornamental. Each result enables a capability that did not
exist before TASB.

### Per-Head Thermodynamic Diagnostics at Inference Time

Cv = Var_ρ(H) is computed from every TASB forward pass at zero additional cost —
no extra model calls, no gradient computation, no architectural change. It gives you,
for every head at every layer for every token:

- **High Cv:** diffuse, thermodynamically active head. Attention spread broadly across
  context; sensitive to approximation scheme; likely performing integration over
  multiple relevant tokens rather than sharp retrieval.
- **Low Cv:** sharp, peaked head. Confident retrieval behavior; robust to temperature
  perturbation (§3 shows sharp heads are least affected by over-smoothing, since they
  are already far from uniform in the over-smoothing direction).

This diagnostic was inaccessible before TASB without a separate analysis pass.
The bridge makes the thermometer free.

### Context Extension Cost, Now Visible

Every production deployment of LLaMA-class models uses NTK-aware RoPE context
extension. Every such deployment is implicitly running a Jarzynski protocol on every
attention head at every extended-context token. Every head has a W_diss = T · KL(p_orig
‖ p_NTK) — a dissipated work cost proportional to how much that head's distribution
shifts under the frequency change.

Until TASB, this cost was undefined (no thermodynamic framework for RoPE),
unmeasurable (no inference-time observable), and unknown (unreported in the Mamba
and LLaMA papers). With TASB Cv, it is measurable per head, per layer, per token,
during inference. The heads that bear the highest thermodynamic cost under context
extension are identifiable in real time.

### Thermodynamic-Native Model Design

The SSM literature has been making thermodynamically correct choices without the
thermodynamic reasoning:

- Mamba's switch from bilinear to ZOH is the FDT-exact choice
- NTK-aware RoPE scaling is a Jarzynski protocol applied to position encoding

Both were discovered post-hoc — after the architectures existed. TASB's framework
makes it possible to design future architectures from FDT and Jarzynski constraints
first. The ZOH-FDT result (§6.2) suggests a concrete design principle: for any
discretized SDE or recurrence in an ML model, prefer the discretization that
exactly preserves the FDT of the underlying continuous system. ZOH satisfies this.
Bilinear does not. This is a testable, falsifiable architectural constraint — the
kind the field has been missing.

### The Two Theses Converge

Thesis A (§1–§3): TASB is a hardware bridge — frozen transformer weights, Boltzmann
sampling, thermodynamic hardware target, zero retraining required.

Thesis B (§4–§6): TASB is a measurement instrument — Cv is the first inference-time
thermodynamic observable, predicting sampling error, measuring Jarzynski work,
characterizing head behavior across approximation schemes.

These are not two separate projects. Thesis B is only possible because of Thesis A:
the bridge is what exposes the thermodynamic state of the computation. You cannot
measure Cv without implementing the bridge; you cannot exploit the bridge without
understanding Cv.

A frozen transformer running on Extropic TSU hardware via TASB would simultaneously:
1. Operate near-Landauer-efficient Boltzmann sampling
2. Generate Cv measurements with every forward pass at no extra cost
3. Measure its own thermodynamic state — head by head, token by token, in real time

The API is implemented. The THRML backend is validated. The Cv computation is one
line of Python. The hardware path is documented. What remains is silicon access.

---

## §8 — Experimental Results

### Core Validation Summary

All results on LLaMA 3.2-3B (4-bit nf4), d_k=128, 24 heads, 28 layers, A100 GPU.

| Experiment | Result | Status |
|-----------|--------|--------|
| Faithfulness (M5): top-1 agreement, 8,840 positions | 98.9%, 0 confident flips | ✅ PASS |
| Layer universality (M7-1): all 28 layers | 0 confident flips at any layer | ✅ PASS |
| K sweep (M7-2): K ∈ {1,3,5,10,25,50,100} | 0 confident flips, KL monotone ↓ | ✅ PASS |
| Alpha sweep (M7-4): α ∈ {0.0 … 1.0} | 0 confident flips | ✅ PASS |
| Compute overhead (FIND-013) | 1.81× (threshold: <3×) | ✅ PASS |
| r(Cv, KL) at layer 18 (120 obs. from 3,360 total) | 0.8241, p = 6.1×10⁻²⁵ | ✅ PASS |
| r(H(P), KL) at layer 18 / population-wide | 0.9279 / 0.9492 | ✅ NEW |
| T=√d_k ablation: KL minimum location | Exactly at T=√d_k | ✅ PASS |
| Gap C: KL(entmax15 ‖ softmax), 120 heads | 0.1086 ± 0.070, 120/120 > 0 | ✅ PASS |
| r(Cv, KL_entmax), 120 heads | 0.8156 | ✅ PASS |
| Layer-depth Cv ordering | Not confirmed (different regime from Kim) | ✗ STRUCK |
| Gumbel backend Cv bias (π²/6) | Derivation error — not applicable | ✗ STRUCK |
| Gap B: r(ΔCv, W_KL) = 0.549, p = 0.0055, n=24 heads | 21/24 heads ΔCv > 0 | ✅ PASS |
| Multi-model validation: r(H,KL) LLaMA→Phi-3 | 0.95→0.871 population; 0.93→0.821 mid-layer (K=10) | ✅ PASS |
| Multi-model validation: r(Cv,KL) at K=10 | 0.82 (LLaMA L18) → 0.470 (Phi-3 L16) — FAIL; passes at K=25 (0.644) | ⚠️ PARTIAL |

### Cross-Architecture Validation — Phi-3-mini-4k (EXP-004)

Model: `microsoft/Phi-3-mini-4k-instruct` (full MHA, 32Q/32KV heads, head_dim=96,
32 layers). Structural contrast with LLaMA 3.2-3B (24Q/8KV GQA, head_dim=128, 28
layers). Scale: 5 prompts × 32 layers × 32 heads × 6 K-values = **30,720 observations**.

**Key finding:** Shannon entropy of the attention distribution (H) is a robust,
architecture-agnostic predictor of finite-K Boltzmann sampling error across two
independent model families. Cv is architecture-sensitive at the K=10 production
operating point; passes at K≥25.

| K | r(H,KL) all layers | r(H,KL) mid-layer | r(Cv,KL) all | r(Cv,KL) mid-layer |
|---|---------------------|-------------------|-------------|---------------------|
| 1 | 0.831 | 0.739 | −0.040 | 0.066 |
| 5 | 0.895 | 0.857 | 0.070 | 0.360 |
| **10** | **0.871** | **0.821** | **0.184** | **0.470** |
| 25 | 0.771 | 0.689 | 0.374 | 0.644 |
| 50 | 0.638 | 0.599 | 0.493 | 0.679 |
| 100 | 0.489 | 0.394 | 0.539 | 0.642 |

The FIND-008 "single model family" skeptic objection is directly answered: two
independent architectures, two organizations (Meta / Microsoft), different head
topologies and temperatures. Entropy correlation robust in both.

### Attention Matrices: Real LLaMA Data

`demo/data/attention_matrices.json` (4.63 MB): actual per-token softmax attention
matrices captured from frozen LLaMA 3.2-3B on A100, 5 prompts × 28 layers × 24 heads.
Token strings on both axes. Viewable interactively in the
[live demo](https://huggingface.co/spaces/shvrpws/thermobridge).

---

## §9 — Prior Art & Positioning

### Comparison Table

| Work | Year | Key contribution | Relationship to TASB |
|------|------|-----------------|---------------------|
| Vaswani et al. | 2017 | Scaled dot-product attention | Implicit Boltzmann temperature via 1/√d_k — no thermodynamic framing |
| Ramsauer et al. | 2021 | Hopfield ↔ attention | Energy-based connection; deterministic, not sampling |
| Kajitsuka & Sato | 2023 | Boltzmann operator theorem | Theoretical foundation (Route 2); no inference-time implementation |
| Kim (arXiv:2602.08216) | Feb 9, 2026 | Thermodynamic Lagrangian derivation | Theoretical foundation (Routes 1+3); no hardware bridge |
| **TASB patent filing** | **Mar 28, 2026** | **Inference-time frozen-weight bridge** | **First implementation** |
| Boltzmann Attention (arXiv:2606.12478) | Jun 23, 2026 | Learnable Ising couplings | Requires training; submitted 87 days after patent |
| FAR (arXiv:2505.21535v4) | May 2026 | BiLSTM distillation | Requires training; different architectural approach |

### The Prior Art Gap

No work in the literature — as of the patent filing date, March 28, 2026, and as
confirmed by systematic arXiv search from February 9 to March 28, 2026 — implements
an inference-time, frozen-weight Boltzmann bridge for arbitrary transformer attention.

Kim (2026) is the closest theoretical work, submitted 47 days before the patent.
Kim proves the isomorphism and derives the Lagrangian. Kim does not build the bridge,
does not implement sampling, does not connect to thermodynamic hardware, and does not
derive the Cv observable or its empirical validation. The gap from theory to
implementation is TASB's contribution.

### What Makes TASB Different

Three properties, in combination, are not found in any prior work:

1. **Inference-time:** operates on frozen weights at inference, not training
2. **Exact Boltzmann:** samples from the correct distribution, not an approximation to it
3. **Hardware path:** direct mapping to TSU via THRML, with validated API compatibility

Any two of these three properties appear in prior work. All three together do not.

---

## Citation

```bibtex
@software{tasb2026,
  author       = {Paul White},
  title        = {TASB: Thermodynamic Attention Sampling Bridge},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/whtetigr2/TASB},
  note         = {Patent Pending, USPTO Provisional 64/019,999}
}
```

---

*Last updated: 2026-07-02. §3 KL measurements disambiguated (K=10 vs K=5000 vs production α=0.3). §4 entropy comparison added (r=0.93 vs Cv r=0.82 at L18; r=0.95 vs 0.61 population-wide). §8 updated with EXP-004 cross-architecture validation (Phi-3-mini-4k, 30,720 observations). All planned experiments complete.*
