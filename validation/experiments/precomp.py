# ==========================================================================
# FIX 1 -- PRECOMPENSATION.  The alpha->inf route is dead (mode collapse).
# Instead run at LOW alpha (mixes fine) and remove the equilibrium bias
# exactly: solve for l' such that RBM_alpha(l') == softmax(l).
# Legitimate because at m=16 the visible space is only 2^4=16 states, so the
# RBM marginal is computable in closed form and the fixed point is cheap.
# This is a HOST-side O(m^2) precompute, done once per attention row.
# ==========================================================================
import numpy as np
rng = np.random.default_rng(0)
sig = lambda x: 1.0/(1.0+np.exp(-x))

def build(m):
    b = int(np.log2(m))
    C = np.array([[1.0 if (j>>i)&1 else -1.0 for i in range(b)] for j in range(m)])
    return b, C

def rbm_marginal(lp, alpha):
    m = len(lp); b, C = build(m)
    u = alpha*(C @ C.T - (b-1)) + lp[None,:]     # S=C: state s==code(s)
    F = -np.logaddexp(0.0, u).sum(axis=1)
    p = np.exp(-(F-F.min())); return p/p.sum()

def precompensate(l, alpha, iters=200):
    tgt = np.exp(l-l.max()); tgt /= tgt.sum()
    lp = l.copy()
    for _ in range(iters):
        p = rbm_marginal(lp, alpha)
        lp = lp + (np.log(tgt) - np.log(np.clip(p,1e-300,None)))   # multiplicative fixed point
        lp -= lp.max()
    return lp, tgt

def gibbs(lp, alpha, n_steps, burn, seed):
    m = len(lp); b, C = build(m); r = np.random.default_rng(seed)
    s = r.choice([-1.0,1.0], size=b); traj = np.empty(n_steps, np.int32)
    for t in range(n_steps):
        u = alpha*(C @ s - (b-1)) + lp
        h = (r.random(m) < sig(u)).astype(np.float64)
        f = alpha*(h @ C)
        s = np.where(r.random(b) < sig(2*f), 1.0, -1.0)
        traj[t] = int(sum((1<<i) for i in range(b) if s[i] > 0))
    return traj[burn:]

def ac1(x):
    x = x.astype(float)-x.astype(float).mean(); d=(x*x).mean()
    return float((x[:-1]*x[1:]).mean()/d) if d>0 else 0.0

m=16; l = rng.normal(0,3.0,m)
print('PRECOMPENSATED: equilibrium bias removed analytically, then sampled.')
print('alpha   TV(exact eq.)   TV(sampled,200k)   autocorr   ESS/N   tau(steps)')
for alpha in (1,2,3,5,8):
    lp,tgt = precompensate(l, alpha)
    p_eq = rbm_marginal(lp, alpha)
    tv_eq = 0.5*np.abs(p_eq-tgt).sum()
    tr = gibbs(lp, alpha, 200_000, 20_000, 1)
    emp = np.bincount(tr, minlength=m).astype(float); emp/=emp.sum()
    tv_s = 0.5*np.abs(emp-tgt).sum(); a = ac1(tr)
    ess = (1-a)/(1+a); tau = (1+a)/(1-a)
    print(f'{alpha:<8}{tv_eq:<16.2e}{tv_s:<19.4f}{a:<11.4f}{ess:<8.4f}{tau:.1f}')
print()
print('CONTROL -- is this chain actually COUPLED (not the old IID bug)?')
lp,tgt = precompensate(l, 3.0)
tr = gibbs(lp, 3.0, 200_000, 20_000, 1)
print(f'  autocorr of coupled bipartite RBM chain : {ac1(tr):.4f}')
iid = np.random.default_rng(7).choice(m, size=180_000, p=tgt)
print(f'  autocorr of a genuine IID categorical   : {ac1(iid):.4f}   <-- old construction looked like this')
