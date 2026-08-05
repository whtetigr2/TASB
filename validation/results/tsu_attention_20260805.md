# TSU-native attention for a frozen transformer — results, 2026-08-05

Environment: Lightning studio, Tesla T4. `thrml 0.1.4`, `extro-torx 0.0.1`, `jax 0.11.0`,
`torch 2.8.0+cu128`, `transformers 4.57.6`, `numpy 2.5.1`. Eval text: wikitext-2 test,
first 20k chars, GPT-2 small, `attn_implementation='eager'`.

Primary source for all hardware constraints and energy figures:
**arXiv:2510.23972**, *An efficient probabilistic hardware architecture for diffusion-like
models* (Extropic — the DTM / "Thermalizers" paper).

---

## 0. Toolchain

| package | status |
|---|---|
| `thrml` 0.1.4 | installed |
| `extro-torx` 0.0.1 | **installed this session** (`pip install extro-torx`) |
| thermalizers | **no public code exists** — not on PyPI, not in `github.com/extropic-ai`. The org has exactly `torx`, `thrml`, `codon_opt`, `thrml-skill`. "Thermalizers" is the paper + a closed internal compiler. |

## 1. Hardware constraints, per the paper (not inferred)

- **Binary Bernoulli p-bits only** — "Each node represents a single Bernoulli random variable."
- **Quadratic (pairwise) energy only** — App C.1 "Quadratic EBMs".
- **Bipartite / two-colourable** — "nodes can be separated into two blocks... each color block can be sampled in parallel."
- **Degree ~12** — "Each variable was connected to several (in most cases, 12) of its neighbors."
- One full Gibbs iteration = `2·τ_RNG`; layers "mix in tens of iterations"; `n_Gibbs≈100` used "to be conservative".
- App I: their own integer embedding is a **thermometer sum-code** `x = Σₖ sₖ`, which spans only a 2-parameter family per variable.

**Consequence:** no single quadratic EBM on binary p-bits can represent an arbitrary
m-way categorical. Confirmed both by their App I construction and by measurement below.

## 2. Closed-form bipartite RBM embedding of an attention softmax

For support size m with frozen-model logits ℓ, visibles `s∈{−1,+1}^b` (b=log₂m),
hiddens `h∈{0,1}^m`:

```
u_j(s) = α·( Σᵢ σᵢ(j)·sᵢ − (b−1) ) + ℓ_j
E(s,h) = − Σ_j h_j · u_j(s)
```

Marginalising h gives `p(s) ∝ exp Σ_j softplus(u_j(s)) → softmax(ℓ)`.
Strictly pairwise, bipartite, visible-degree m, hidden-degree b. **Nothing is trained**;
every parameter is closed-form in the frozen model's logits.

Exactness of the embedding (enumerated, m=16):

| α | max abs err | TV | KL |
|---|---|---|---|
| 10 | 2.9e-3 | 3.0e-3 | 5.5e-5 |
| 20 | 1.3e-7 | 1.4e-7 | 1.2e-13 |
| 40 | 5.6e-16 | 5.8e-16 | ~0 |

## 3. The mixing wall — independently reproducing the paper's MET

α is in units of kT, so exactness demands large barriers. 2-block Gibbs, m=16, 200k steps:

| α | TV | autocorr | states visited | ESS/N |
|---|---|---|---|---|
| 2 | 0.174 | 0.636 | 16 | 0.222 |
| 5 | 0.038 | 0.908 | 16 | 0.048 |
| 10 | 0.066 | 0.9993 | 7 | 0.0004 |
| 15+ | **0.9997** | 0.0 | **1** | frozen |

Two failure modes cross: below α≈5 the *equilibrium* is wrong; above it the chain
*never reaches* equilibrium. This is the paper's **Mixing–Expressivity Tradeoff**
("barriers lead to exponentially large expected transition times... reflected in a
rapidly growing mixing time"), reproduced here on a new problem.

**Precompensation** — solve for ℓ′ with `RBM_α(ℓ′) = softmax(ℓ)`, cheap because the
visible space is only 2^b states:

| α | TV (exact equilibrium, precompensated) | autocorr | τ |
|---|---|---|---|
| 1–3 | 0.58 / 0.41 / 0.20 — fixed point cannot converge | 0.01–0.15 | ~1 |
| 5 | 2.0e-2 | 0.663 | 4.9 |
| 8 | **1.9e-4** | 0.977 | 87 |

Below α≈5 the RBM *cannot represent* an arbitrary 16-way categorical at any ℓ′.

## 4. Support requirement — correction to an earlier claim

An earlier session reported a **1.09× architectural floor**. That was measured at short
sequence length and **does not transfer**. Exact (no sampling), sink + local window:

| n_sink=4, w=16 (m=20) | S=128 | S=256 | S=512 | S=1024 |
|---|---|---|---|---|
| × baseline | 1.51 | 2.23 | 4.72 | **9.34** |
| mass captured | 0.85 | 0.79 | 0.73 | 0.68 |

The **sink finding is robust and strengthens with S**: `n_sink=0, w=16` gives 339× at
S=128 and 6576× at S=1024. Extra sink wires beyond 4 add little; the *window* carries
the residual. At S=512, m=128 is needed for ~1.63×.

## 5. Tree factorisation — degree-bounded by construction

Flat embedding needs visible-degree = m, which blows the degree budget. Exact chain rule
`p(j) = Πₖ p(gₖ | g₁..gₖ₋₁)` makes max degree the **branching factor**, not m.
S=512, m=64 (4 sink + 60 local), exact floor 42.87 (2.042× baseline), control 20986:

| tree | max degree | p-bits/row | chains | ppl | × floor | × baseline |
|---|---|---|---|---|---|---|
| 4×16 | 16 | 86 | 32 | 74.55 | 1.74 | 3.55 |
| 8×8 | 8 | 99 | 32 | 53.86 | 1.26 | 2.57 |
| 4×4×4 | 4 | 126 | 32 | 58.57 | 1.37 | 2.79 |
| 2⁶ | 2 | 189 | 32 | 56.85 | 1.33 | 2.71 |
| 4×4×4 | 4 | 126 | 64 | 49.37 | 1.15 | 2.35 |
| **4×4×4** | **4** | **126** | **128** | **44.26** | **1.032** | **2.11** |

Degree 4 — inside the paper's stated 12 — and the sampler converges to **1.03× of the
exact support floor**. The sampler is no longer the bottleneck; support restriction is.
α=8 gives 9.28× floor (MET again).

## 6. THRML-native validation — the IID problem is dead

Spin form (`h=(1+t)/2`) is exactly an Ising model:
edge weight `W_ij = (α/2)·σᵢ(j)`, hidden bias `b_j = (ℓ_j − α(b−1))/2`, visible bias 0.
Built with `IsingEBM` + `IsingSamplingProgram`, free blocks `[Block(visibles), Block(hiddens)]`.
1280 nodes / 4096 edges / 2 colour blocks, 64 rows packed SIMD, 20k samples:

| metric | value |
|---|---|
| TV vs frozen-model softmax | **0.0564** (median 0.0522) |
| lag-1 autocorrelation | **0.9129** — genuinely coupled |
| same target, true IID draws | 0.0002 — what the old uncoupled build gave |
| α=0 control (couplings zeroed) | TV **0.5901** — couplings are load-bearing |

THRML convention verified from source, not assumed: `True`=+1, `p(+1)=sigmoid(2γ)`,
`γ = Σ W·(spin product)` ⟹ `E = −Σ W·(spin product)`.

**Bug found and fixed here:** the `{0,1}→{−1,+1}` change of variables carries an offset
`−α(b−1)` in the hidden bias. Dropping it gave TV 0.81 (worse than no couplings at all).

## 7. Energy accounting — the verdict

Grounded in the paper's own method: GPU = A100, "19.5 TFLOPS for Float32 and 400W"
⟹ **20.51 pJ/FLOP**. TSU energy per p-bit update backed out of App E.4
(`E_DTM ≈ 1.6 nJ`, dominated by `E_samp`; 4096 cells × 100 Gibbs × 8 layers)
⟹ **≈0.49 fJ/update**.

Per attention row, m=64, d=64:

| step | FLOP | nJ | share |
|---|---|---|---|
| logits `q·k_j` | 8192 | 168.04 | 49.4% |
| **softmax** | **192** | **3.94** | **1.16%** |
| weighted sum | 8192 | 168.04 | 49.4% |
| total | 16576 | 340.02 | 100% |

TSU cost of the step it replaces: 126 p-bits × 90 iters × 128 chains = 1,451,520 updates
= **0.709 nJ vs 3.938 nJ digital → 5.56× cheaper**, robust down to ~3 fJ/update.

**But Amdahl:**

| | |
|---|---|
| whole attention row, all-digital | 340.02 nJ |
| with TSU softmax | 336.79 nJ |
| net saving | **0.95% of the row** |
| rolled up to whole model (attention ≈⅓ of FLOPs) | **0.317%** |
| ceiling at *zero* TSU energy | **0.386%** |

**The matmuls are irreducible: you cannot sample from `softmax(q·Kᵀ)` without first
computing `q·Kᵀ` to program the couplings.** No p-bit energy fixes this.

---

## Verdict

The architecture is real, faithful to Extropic's published substrate, needs no
retraining, and works in simulation at 1.03× of its own exact floor. It does **not**
meet Extropic's stated energy standards — not because the TSU is bad, but because a
frozen transformer's energy lives in dense linear algebra, and the TSU's primitive is
sampling. **~0.39% is the hard ceiling.**

The route that survives is Extropic's own App J (HTDML): a small trained adapter into a
binary latent space with the DTM operating there. That requires training an adapter —
not retraining the transformer, but not "strictly frozen" either.

## Limits of this evidence

- GPT-2 small only; one 512-token text; perplexity only. No other model, no RoPE model.
- The 0.49 fJ/update figure is a **back-out**, not a measurement — grid size and depth
  were read from prose, and the paper's own numbers came through a partially
  math-stripped HTML render. Sensitivity swept over 4 orders of magnitude (§7).
- The tree sampler is validated in PyTorch; the THRML validation (§6) is the flat
  m=16 RBM, not yet the full tree.
- Not ported to a torx `DFG` (`ChainFactor`/`TiledFactor`) — the semantics map cleanly
  but that is unbuilt.
- No TSU silicon exists to test on; everything here is simulation, as in the paper.
