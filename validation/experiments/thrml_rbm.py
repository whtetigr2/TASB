# ==========================================================================
#  THRML-NATIVE bipartite RBM for attention softmax  -- COUPLED, not IID.
#
#  This is the construction expressed in THRML's own primitives (IsingEBM +
#  IsingSamplingProgram + block Gibbs), so the claim rests on THRML's
#  semantics rather than on my own PyTorch algebra.
#
#  Spin form.  Hiddens h_j in {0,1} -> t_j in {-1,+1} via h=(1+t)/2:
#      E(s,t) = -(alpha/2) sum_{i,j} sigma_i(j) s_i t_j  -  (1/2) sum_j l_j t_j
#  which is EXACTLY an Ising model:
#      edge (visible i, hidden j) weight  W_ij = alpha/2 * sigma_i(j)
#      bias on hidden j                   b_j  = l_j / 2
#      bias on visible i                  0
#  THRML's convention E = -sum W*(spin product) is what makes the signs line
#  up; that convention was read out of thrml source, not assumed.
#
#  Two blocks (visibles, hiddens) with NO intra-block edges => the graph is
#  bipartite and 2-colourable, so THRML's block Gibbs over
#  [Block(visibles), Block(hiddens)] is EXACT alternating Gibbs -- the same
#  operation the DTCA performs in hardware.
#
#  R independent rows are packed into the same two blocks (SIMD), which is
#  how a real chip would host many attention rows at once.
# ==========================================================================
import math
import jax, jax.numpy as jnp, numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

np.random.seed(0)
M      = 16                      # support size
B_BITS = int(math.log2(M))
ALPHA  = 5.0
R      = 64                      # independent attention rows packed together

SIG = np.array([[1.0 if (j >> i) & 1 else -1.0 for i in range(B_BITS)]
                for j in range(M)])                          # (M, b) codes


# ---- reference: exact RBM visible marginal + precompensation (numpy) -----
def rbm_marginal(lp, alpha):
    u = alpha * (SIG @ SIG.T - (B_BITS - 1)) + lp[None, :]
    s = np.logaddexp(0.0, u).sum(1)          # -F(state), one entry per visible state
    e = np.exp(s - s.max())
    return e / e.sum()


def precompensate(l, alpha, iters=200):
    tgt = np.exp(l - l.max()); tgt /= tgt.sum()
    lp = l.copy()
    for _ in range(iters):
        p = np.clip(rbm_marginal(lp, alpha), 1e-300, None)
        lp = lp + (np.log(tgt) - np.log(p)); lp -= lp.max()
    return lp, tgt


# ---- build the THRML graph ----------------------------------------------
logits = np.random.randn(R, M) * 2.0
LP = np.stack([precompensate(logits[r], ALPHA)[0] for r in range(R)])
TGT = np.stack([np.exp(l - l.max()) / np.exp(l - l.max()).sum() for l in logits])

vis = [[SpinNode() for _ in range(B_BITS)] for _ in range(R)]
hid = [[SpinNode() for _ in range(M)] for _ in range(R)]
vis_flat = [n for row in vis for n in row]
hid_flat = [n for row in hid for n in row]

edges, weights = [], []
for r in range(R):
    for i in range(B_BITS):
        for j in range(M):
            edges.append((vis[r][i], hid[r][j]))
            weights.append(ALPHA / 2.0 * SIG[j, i])

nodes = vis_flat + hid_flat
# h_j in {0,1} -> t_j in {-1,+1} via h=(1+t)/2 carries an OFFSET:
#   u_j(s) = alpha*(sum_i sigma_i(j) s_i) - alpha*(b-1) + l_j
#   => hidden bias  b_j = (l_j - alpha*(b-1)) / 2      <- the -alpha*(b-1) is
#   what holds a hidden OFF unless its code matches; dropping it flattens
#   the construction (measured: TV 0.81 instead of ~0.02).
# Visible bias is 0 because sum_j sigma_i(j) = 0 over a full binary code set.
hid_bias = (LP - ALPHA * (B_BITS - 1)) / 2.0
biases = np.concatenate([np.zeros(R * B_BITS), hid_bias.reshape(-1)])

ebm = IsingEBM(nodes=nodes, edges=edges,
               biases=jnp.array(biases, dtype=jnp.float32),
               weights=jnp.array(weights, dtype=jnp.float32),
               beta=jnp.array(1.0, dtype=jnp.float32))

BV, BH = Block(vis_flat), Block(hid_flat)
prog = IsingSamplingProgram(ebm, free_blocks=[BV, BH], clamped_blocks=[])
print('THRML graph: %d nodes (%d visible + %d hidden), %d edges, 2 colour blocks'
      % (len(nodes), R * B_BITS, R * M, len(edges)))
print('degree: visible=%d, hidden=%d   bipartite=yes  chromatic=2' % (M, B_BITS))

N_SAMP = 20000
sched = SamplingSchedule(n_warmup=500, n_samples=N_SAMP, steps_per_sample=1)
key = jax.random.PRNGKey(0)
init = [jnp.asarray(np.random.rand(R * B_BITS) < 0.5),
        jnp.asarray(np.random.rand(R * M) < 0.5)]
out = sample_states(key, prog, sched, init, [], [BV])
sv = np.asarray(out[0])                       # (n_samples, R*b) bool
print('sampled states:', sv.shape)

sv = sv.reshape(N_SAMP, R, B_BITS)
pw = 2 ** np.arange(B_BITS)
traj = (sv * pw).sum(-1)                      # (n_samples, R) decoded position

emp = np.stack([np.bincount(traj[:, r], minlength=M) for r in range(R)]).astype(float)
emp /= emp.sum(1, keepdims=True)
tv = 0.5 * np.abs(emp - TGT).sum(1)


def ac1(x):
    x = x.astype(float) - x.astype(float).mean(); d = (x * x).mean()
    return float((x[:-1] * x[1:]).mean() / d) if d > 0 else 0.0


acs = np.array([ac1(traj[:, r]) for r in range(R)])
iid = np.stack([np.random.choice(M, size=N_SAMP, p=TGT[r]) for r in range(R)])
acs_iid = np.array([ac1(iid[r]) for r in range(R)])

print('\n--- THRML block-Gibbs result over %d rows, %d samples ---' % (R, N_SAMP))
print('TV(empirical, softmax target) : mean %.4f   median %.4f   max %.4f'
      % (tv.mean(), np.median(tv), tv.max()))
print('lag-1 autocorrelation         : mean %.4f  <- COUPLED chain' % acs.mean())
print('same target, true IID draws   : mean %.4f  <- what the old uncoupled build gave'
      % acs_iid.mean())
print('\nCONTROL -- alpha=0 (couplings removed) must destroy the target:')
ebm0 = IsingEBM(nodes=nodes, edges=edges,
                biases=jnp.array(biases, dtype=jnp.float32),
                weights=jnp.zeros(len(weights), dtype=jnp.float32),
                beta=jnp.array(1.0, dtype=jnp.float32))
prog0 = IsingSamplingProgram(ebm0, free_blocks=[BV, BH], clamped_blocks=[])
out0 = sample_states(jax.random.PRNGKey(1), prog0, sched, init, [], [BV])
t0 = (np.asarray(out0[0]).reshape(N_SAMP, R, B_BITS) * pw).sum(-1)
e0 = np.stack([np.bincount(t0[:, r], minlength=M) for r in range(R)]).astype(float)
e0 /= e0.sum(1, keepdims=True)
print('  TV with couplings zeroed    : mean %.4f  (vs %.4f coupled)'
      % ((0.5 * np.abs(e0 - TGT).sum(1)).mean(), tv.mean()))
