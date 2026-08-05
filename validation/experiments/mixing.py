# ==========================================================================
# THE MAKE-OR-BREAK: does the bipartite RBM actually MIX at the alpha needed
# for accuracy?  alpha is in units of kT, so alpha=20 => 20 kT barriers.
# Adversarial expectation: it locks into a mode and never leaves.
# Measured by 2-block Gibbs (visibles | hiddens, hiddens | visibles).
# ==========================================================================
import numpy as np
rng = np.random.default_rng(0)
sig = lambda x: 1.0/(1.0+np.exp(-x))

def build(m):
    b = int(np.log2(m))
    C = np.array([[1.0 if (j>>i)&1 else -1.0 for i in range(b)] for j in range(m)])
    return b, C

def gibbs(l, alpha, n_steps, burn, rng, s0=None):
    m = len(l); b, C = build(m)
    s = rng.choice([-1.0,1.0], size=b) if s0 is None else s0.copy()
    traj = np.empty(n_steps, dtype=np.int32)
    for t in range(n_steps):
        u = alpha*(C @ s - (b-1)) + l                      # (m,)
        h = (rng.random(m) < sig(u)).astype(np.float64)    # hiddens | visibles
        f = alpha * (h @ C)                                # (b,) field on visibles
        s = np.where(rng.random(b) < sig(2*f), 1.0, -1.0)  # visibles | hiddens
        traj[t] = int(sum((1<<i) for i in range(b) if s[i] > 0))
    return traj[burn:]

def autocorr1(x):
    x = x.astype(float); x = x - x.mean()
    d = (x*x).mean()
    return float((x[:-1]*x[1:]).mean()/d) if d > 0 else 0.0

m = 16; l = rng.normal(0, 3.0, m)
tgt = np.exp(l-l.max()); tgt /= tgt.sum()
N, BURN = 200_000, 20_000
print(f'm={m}  N={N}  target entropy={-(tgt*np.log(tgt)).sum():.3f} nats  max p={tgt.max():.3f}')
print()
print('alpha   TV(empirical,softmax)   autocorr(lag1)   distinct states visited   ESS/N')
for alpha in (2,5,8,10,15,20,30):
    tr = gibbs(l, alpha, N, BURN, np.random.default_rng(1))
    emp = np.bincount(tr, minlength=m).astype(float); emp /= emp.sum()
    tv = 0.5*np.abs(emp-tgt).sum()
    ac = autocorr1(tr)
    ess = (1-ac)/(1+ac) if ac > -1 else 1.0
    print(f'{alpha:<8}{tv:<24.4f}{ac:<17.4f}{len(np.unique(tr)):<26}{ess:.4f}')
