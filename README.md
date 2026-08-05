# thermobridge

**Thermodynamic attention sampling for frozen transformers.**

> ## Status: the original premise is closed — negative result, 2026-08-05
>
> thermobridge began as an attempt to run a **frozen** transformer's attention on
> Extropic's thermodynamic sampling hardware as a drop-in backend. Deep verification
> on 2026-08-05 established that this **cannot pay off energetically**, and the
> reason is structural rather than an implementation defect.
>
> **A frozen autoregressive transformer has no intractable normalisation anywhere in
> its forward pass** — its conditionals are tractable by construction. Its energy is
> spent on dense linear algebra, and a TSU's primitive is sampling. The one genuinely
> Boltzmann-shaped operation is the attention softmax, which is **1.16% of an
> attention row**; logits and the weighted sum are 49.4% each and are irreducible,
> because you cannot sample from `softmax(q·Kᵀ)` without first computing `q·Kᵀ` to
> program the couplings.
>
> Whole-model energy ceiling: **~0.39%, even at zero TSU energy.**
>
> This repository is preserved as the evidence trail. See
> [`validation/results/tsu_attention_20260805.md`](validation/results/tsu_attention_20260805.md)
> for the full measurement record, controls, and refuted attempts.

---

## Corrections to earlier claims in this repository

Two classes of earlier claim did not survive verification. They are corrected here
rather than deleted, because the measurements were sound — what was wrong was the
inference drawn from them.

**1. The sampler was uncoupled (IID), so "block Gibbs" was doing nothing.**
The THRML graph was built with `CategoricalEBMFactor` over a **single node group**.
Per `DiscreteEBMFactor` the energy term is then unary — no couplings — so the joint
factorises completely and consecutive samples are independent by construction.
Measured: lag-1 autocorrelation **0.013**, against **0.529** for a genuinely coupled
graph.

Consequences for the old validation table:

| earlier claim | status |
|---|---|
| "T2.C Detailed balance (χ², p=0.43)" | **Mislabelled and uninformative.** It was a goodness-of-fit test of the empirical distribution, not a detailed-balance test of a transition kernel. An uncoupled categorical sampler passes it trivially. |
| "T2.B Gibbs chain mixing, R-hat = 1.0003" | **Vacuous on an uncoupled graph.** A separate R-hat of 14.56 chased earlier was a units artifact; computed correctly it is 0.9996. |
| `thrml` backend "Hardware-ready" | **Withdrawn.** The graph it built had no couplings and was not a TSU path. |
| T3/T5 perplexity (LLaMA 3.2-3B, OLMoE-1B-7B) | **Measurements stand, framing does not.** They show that K-sample attention approximates softmax attention. They do not show anything thermodynamic. |

**2. The "1.09× architectural floor" was sequence-length dependent.**
It was measured at short S and does not transfer. Same configuration
(4 sink + 16 local window), exact, no sampling:

| S | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| × baseline | 1.51 | 2.23 | 4.72 | **9.34** |
| attention mass captured | 0.85 | 0.79 | 0.73 | 0.68 |

---

## What was established, and stands

Measured on GPT-2 small, wikitext-2, S=512, with controls throughout. Environment
pinned in [`requirements.lock`](requirements.lock).

**A closed-form, training-free embedding of frozen attention onto Extropic's
published substrate.** Binary Bernoulli p-bits, strictly pairwise energy, bipartite,
degree 4 — all matching arXiv:2510.23972. Nothing is trained; every parameter is
written directly from the frozen model's logits.

```
u_j(s) = alpha * ( sum_i sigma_i(j) * s_i  -  (b-1) )  +  l_j
E(s,h) = - sum_j h_j * u_j(s)
```

In spin form this is exactly an Ising model — edge weight `W_ij = (alpha/2)*sigma_i(j)`,
hidden bias `b_j = (l_j - alpha*(b-1))/2` — and it is built in THRML with `IsingEBM`
and sampled by two-colour block Gibbs, the DTCA's native operation.

| result | value |
|---|---|
| embedding exactness (enumerated, α=40) | 5.6e-16 |
| THRML TV vs frozen-model softmax | 0.056 |
| THRML lag-1 autocorrelation | **0.913** (genuinely coupled) |
| same target, true IID draws | 0.0002 |
| α=0 control (couplings zeroed) | TV 0.590 — couplings are load-bearing |
| tree-factored max degree | **4** (paper states ~12) |
| p-bits per attention row | 126 |
| sampling convergence | **1.032×** of its own exact floor |

**Independent reproduction of the paper's Mixing–Expressivity Tradeoff.** Pushing α
for exactness freezes the chain: at α≥15, TV 0.9997 with **one state visited out of
16**. Two failure modes cross — below α≈5 the equilibrium is wrong, above it the
chain never reaches equilibrium. Reached from frozen-transformer attention, a
different problem than the paper's.

**The attention sink is real and strengthens with sequence length.** Removing the
absolute sink wires (`n_sink=0`, local window only) costs 339× at S=128 and
**6576×** at S=1024.

**Three refuted attempts to let the device compute the field itself**, all with
controls (random-gate control: 620× baseline):

| approach | result |
|---|---|
| raw-QK gate, `sigmoid(a·ℓ+b)` | 14.9× — bias-limited, K=8→512 barely moves it |
| z-score gate | 16–53× |
| global lateral inhibition | 65–102×, monotonically worse in λ |

Only the **top-k** gate worked (1.21×), and only because `τ` is a per-row statistic —
the partition function in disguise, which requires the host.

---

## The surviving route

Extropic's own Appendix J (HTDML): a small trained adapter into a binary latent
space where the DTM **is** the generator, rather than a sampler bolted onto
irreducible linear algebra. That requires training an adapter — not retraining the
transformer, but not "strictly frozen" either.

---

## Repository layout

```
validation/experiments/   every script behind the results below
validation/results/       measured outputs, dated
  tsu_attention_20260805.md    <- the full record
archive/                  quarantined early development (see its README)
```

## Limits of this evidence

GPT-2 small only; one 512-token text; perplexity only. The per-p-bit-update energy
figure (~0.49 fJ) is **backed out** of the paper's Appendix E.4, not measured.
The THRML validation covers the flat m=16 RBM, not the full tree. Not ported to a
torx `DFG`. No TSU silicon exists to test on — everything here is simulation, as in
the paper.

## Citation

```bibtex
@misc{shaver2026thermobridge,
  author    = {Shaver, Paul W.},
  title     = {thermobridge: Thermodynamic Attention Sampling for Frozen Transformers},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/whtetigr2/TASB}
}
```

Primary source for all hardware constraints and energy figures used above:
A. Jelinčič et al., *An efficient probabilistic hardware architecture for
diffusion-like models*, [arXiv:2510.23972](https://arxiv.org/abs/2510.23972).

## License

MIT — see [LICENSE](LICENSE)

Patent Pending: USPTO Provisional 64/019,999 (filed 2026-03-28). Retained as a record
of filing; note that the energy conclusion above bears on the scope of what the
disclosed approach can deliver.
