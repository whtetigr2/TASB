"""
R-HAT RESOLUTION — is max_rhat=14.56 a real convergence failure, or an artifact?

Hypothesis: artifact. The committed implementation uses a THEORETICAL binomial
variance p(1-p)/K as the within-chain term instead of the observed within-chain
variance. For near-zero p that denominator collapses and the ratio explodes.

Three computations on the SAME data:
  1. committed formula          -> expect large, matching the June result
  2. correct Gelman-Rubin       -> expect ~1.0 for IID draws
  3. rhat vs magnitude of p     -> if artifact, large rhat concentrates at small p
"""
import numpy as np, jax, jax.numpy as jnp
from thrml import (CategoricalNode, Block, BlockGibbsSpec,
                   FactorSamplingProgram, SamplingSchedule, sample_states)
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional

S, K, M, SEED = 16, 100, 4, 0
rng = np.random.default_rng(SEED)
W = jnp.asarray(rng.normal(0, 2.0, size=(S, S)))

def draw_chain(seed):
    """One 'chain' = K draws, exactly as thrml_sample does it."""
    nodes = [CategoricalNode() for _ in range(S)]
    blk = Block(nodes)
    prog = FactorSamplingProgram(
        gibbs_spec=BlockGibbsSpec(free_super_blocks=[blk], clamped_blocks=[]),
        samplers=[CategoricalGibbsConditional(S)],
        factors=[CategoricalEBMFactor([blk], W)],
        other_interaction_groups=[])
    k = jax.random.key(seed); k1, k2 = jax.random.split(k)
    init = [jax.random.randint(k1, (S,), 0, S, dtype=jnp.uint8)]
    sm = sample_states(k2, prog, SamplingSchedule(0, K, 1), init, [], [blk])
    return np.asarray(sm[0])                        # (K, S) raw draws

raw = np.stack([draw_chain(SEED + m * 1000) for m in range(M)])   # (M, K, S)
# one-hot -> per-chain empirical probability p[m, i, j]
onehot = (raw[..., None] == np.arange(S)).astype(float)           # (M,K,S,S)
p_chain = onehot.mean(axis=1)                                     # (M,S,S)

# ---------- 1. committed formula ----------
committed = []
for i in range(S):
    for j in range(S):
        s = p_chain[:, i, j]
        var_between = K * np.var(s, ddof=1)
        var_within = np.mean(s * (1 - s) / K)          # theoretical, not observed
        r = float(np.sqrt((((M-1)/M)*var_within + (1/M)*var_between) / var_within)) \
            if var_within > 1e-10 else 1.0
        committed.append((r, float(s.mean())))
committed = np.array(committed)

# ---------- 2. correct Gelman-Rubin on the raw draws ----------
correct = []
for i in range(S):
    for j in range(S):
        x = onehot[:, :, i, j]                          # (M, K) indicator series
        chain_means = x.mean(axis=1)
        Wv = x.var(axis=1, ddof=1).mean()               # OBSERVED within-chain variance
        Bv = K * np.var(chain_means, ddof=1)
        if Wv <= 1e-12:
            continue
        Vhat = ((K - 1) / K) * Wv + (1 / K) * Bv        # n = K, not M
        correct.append(float(np.sqrt(Vhat / Wv)))
correct = np.array(correct)

print("=" * 70)
print(f"data: M={M} chains x K={K} draws, S={S}, IID by construction")
print("=" * 70)
print(f"1. COMMITTED formula   max={committed[:,0].max():8.4f}  mean={committed[:,0].mean():7.4f}")
print(f"2. CORRECT Gelman-Rubin max={correct.max():8.4f}  mean={correct.mean():7.4f}   (IID -> ~1.0)")
print()
big = committed[committed[:, 0] > 2.0]
small = committed[committed[:, 0] <= 2.0]
print("3. WHERE the large values live (artifact test):")
print(f"   entries with rhat > 2 : n={len(big):4d}   mean p = {big[:,1].mean():.6f}" if len(big) else "   none")
print(f"   entries with rhat <=2 : n={len(small):4d}   mean p = {small[:,1].mean():.6f}")
if len(big):
    print(f"   -> large rhat sits at p {small[:,1].mean()/max(big[:,1].mean(),1e-12):.0f}x SMALLER than the rest")
print()
verdict = "ARTIFACT" if correct.max() < 1.15 and committed[:,0].max() > 2 else "REAL SIGNAL"
print(f"VERDICT: {verdict}")
