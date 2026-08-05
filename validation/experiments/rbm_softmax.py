# ==========================================================================
# CLOSED-FORM RBM CONSTRUCTION: sparse-support softmax as a BIPARTITE EBM.
#
# Claim to test: for a support of m positions with frozen-model logits l,
# there is a BIPARTITE, degree-bounded, PAIRWISE Boltzmann machine whose
# visible marginal is exactly softmax(l) -- with parameters written down in
# CLOSED FORM from l (no training, no gradient, no retraining of the model).
#
#   visibles s in {-1,+1}^b, b = log2(m)   -- binary code of the position
#   hiddens  h_j in {0,1}, one per support position j
#   u_j(s) = alpha*( sum_i sigma_i(j) s_i - (b-1) ) + l_j
#   F(s)   = -sum_j softplus(u_j(s))        (free energy after marginalising h)
#
# If exactly one j matches s, softplus -> alpha + l_j and the rest -> 0,
# so p(s) prop exp(alpha + l_j) prop softmax(l).  Test that numerically.
# ==========================================================================
import numpy as np
np.random.seed(0)

def codes(m, b):
    # sigma[j,i] = +-1 : binary code of j
    return np.array([[1.0 if (j >> i) & 1 else -1.0 for i in range(b)] for j in range(m)])

def rbm_visible_dist(l, alpha):
    m = len(l); b = int(np.log2(m)); assert 2**b == m
    sig = codes(m, b)                      # (m, b) couplings W[j,i] = alpha*sig[j,i]
    S   = codes(m, b)                      # enumerate all 2^b visible states
    # u[s, j] = alpha*(sig[j].S[s] - (b-1)) + l[j]
    u = alpha * (S @ sig.T - (b - 1)) + l[None, :]
    F = -np.logaddexp(0.0, u).sum(axis=1)  # softplus = logaddexp(0,u)
    p = np.exp(-(F - F.min())); p /= p.sum()
    return p                               # index s corresponds to position s

print('m   alpha   max|p_rbm - softmax|      TV        KL(rbm||softmax)')
for m in (4, 8, 16):
    l = np.random.randn(m) * 3.0
    tgt = np.exp(l - l.max()); tgt /= tgt.sum()
    for alpha in (2, 5, 10, 20, 40):
        p = rbm_visible_dist(l, alpha)
        tv = 0.5*np.abs(p-tgt).sum()
        kl = np.sum(p*np.log(np.clip(p,1e-300,None)/tgt))
        print(f'{m:<4}{alpha:<8}{np.abs(p-tgt).max():<24.3e}{tv:<10.3e}{kl:.3e}')
    print()

# degree / topology audit against Z1 (bipartite, degree<=16)
for m in (4,8,16,32):
    b = int(np.log2(m))
    print(f'm={m:<3} b={b}  p-bits={b+m:<4} visible-degree={m:<3} hidden-degree={b:<3} bipartite=YES chromatic=2')
