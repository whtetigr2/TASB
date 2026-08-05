"""
STEP 1 — the gate as a thrml spin program, and the honest question it raises.

The gate is h_j ~ Bernoulli(sigmoid(l_j - l_tau)), independent across j. As an
EBM that is S spins with local fields and NO couplings: a degree-0 graph.

QUESTION I HAVE TO ASK: if it is uncoupled, is this not exactly the criticism I
levelled at thermobridge? An uncoupled graph needs no TSU -- a GPU RNG does it.

Two things this checks:
  1. does thrml reproduce the intended Bernoulli probabilities from fields alone?
     (i.e. is the gate really expressible as a thrml spin program at all)
  2. what is the COMPILATION ERROR in Thermalizers' sense -- KL between the true
     target factor and the hardware-native EBM -- for the gate vs for a
     categorical? A categorical needs mutual exclusion, which is NOT native.
"""
import numpy as np, jax, jax.numpy as jnp
from thrml import (SpinNode, Block, BlockGibbsSpec, FactorSamplingProgram,
                   SamplingSchedule, sample_states)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional

S, K, SEED = 32, 20000, 0
rng = np.random.default_rng(SEED)

def gate_program(fields):
    """S independent spins with local fields. Degree-0 -> ONE block, valid."""
    nodes = [SpinNode() for _ in range(S)]
    blk = Block(nodes)
    prog = FactorSamplingProgram(
        gibbs_spec=BlockGibbsSpec(free_super_blocks=[blk], clamped_blocks=[]),
        samplers=[SpinGibbsConditional()],
        factors=[SpinEBMFactor([blk], jnp.asarray(fields))],
        other_interaction_groups=[])
    return prog, blk

def sample_gate(fields, key, n=K):
    prog, blk = gate_program(fields)
    k1, k2 = jax.random.split(key)
    init = [jax.random.bernoulli(k1, 0.5, (S,))]
    sm = sample_states(k2, prog, SamplingSchedule(0, n, 1), init, [], [blk])
    return np.asarray(sm[0]).astype(int)          # (n, S) in {0,1}

# ---- 1. does thrml reproduce the intended Bernoulli probabilities? ----
logits = rng.normal(0, 1.5, S)
tau = np.sort(logits)[-max(1, int(0.1 * S))]
want = 1.0 / (1.0 + np.exp(-(logits - tau)))       # target gate probabilities
# THRML convention E = -sum W*s with s in {-1,+1}; P(s=+1)=sigmoid(2W).
# For P(+1)=want we need W = 0.5*logit(want) = 0.5*(l_j - tau).
fields = 0.5 * (logits - tau)

x = sample_gate(fields, jax.random.key(SEED))
got = x.mean(axis=0)
print("=" * 74)
print("STEP 1 — gate as a thrml spin program (S=%d, %d draws)" % (S, K))
print("=" * 74)
print(f"  max |empirical - intended| : {np.abs(got - want).max():.4f}")
print(f"  mean|empirical - intended| : {np.abs(got - want).mean():.4f}")
print(f"  -> thrml {'REPRODUCES' if np.abs(got-want).max() < 0.02 else 'DOES NOT reproduce'} the gate")

# lag-1 autocorrelation: uncoupled => ~0, and that is the honest problem
ac = [abs(np.corrcoef(x[:-1, j], x[1:, j])[0, 1]) for j in range(S) if x[:, j].std() > 1e-9]
print(f"  lag-1 autocorrelation      : {np.mean(ac):.4f}  (uncoupled -> ~0)")

# ---- 2. compilation error in Thermalizers' sense ----
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()
p_true = softmax(logits)

# gate's induced distribution over WHICH position contributes, per draw,
# normalised the way the estimator uses it
w = x / np.maximum(x.sum(1, keepdims=True), 1.0)
p_gate = w.mean(axis=0); p_gate = p_gate / p_gate.sum()

def kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())
def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())

print()
print("STEP 2 — compilation error vs the target attention factor")
print(f"  KL(softmax || gate)  : {kl(p_true, p_gate):.4f}")
print(f"  TV(softmax , gate)   : {tv(p_true, p_gate):.4f}")
print()
print("  NATIVITY (the part that matters for Thermalizers):")
print("   gate        : S spins, degree 0, 1 colour  -> hardware-native EXACTLY,")
print("                 zero structural compilation error, fully parallel")
print("   categorical : needs mutual exclusion = K_S, degree S-1, S colours")
print("                 -> NOT native on a degree-16 2-coloured device")
