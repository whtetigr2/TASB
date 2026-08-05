# ==========================================================================
#  TSU-NATIVE ATTENTION for a FROZEN model.
#
#  Offloads the attention expectation  out_i = E_{j~softmax(l_i)}[v_j]
#  onto a Z1/DTCA-native substrate, per arXiv:2510.23972:
#    * binary Bernoulli p-bits            ("only binary random variables")
#    * QUADRATIC (pairwise) energy only   (App C.1 "Quadratic EBMs")
#    * BIPARTITE / two-colorable          ("nodes separated into two blocks")
#    * 2-block Gibbs, one iteration = 2*tau_RNG
#    * degree <= 16
#
#  Per attention row, over a sparse support of m=16 positions
#  (n_sink absolute "sink" wires + w local window):
#     visibles s in {-1,+1}^4   (b = log2 m)          <- 4 p-bits
#     hiddens  h in {0,1}^16    (one per support pos) <- 16 p-bits
#     E(s,h) = -sum_j h_j [ alpha*( sum_i sigma_i(j) s_i - (b-1) ) + l_j ]
#  => strictly pairwise, bipartite, visible-degree 16, hidden-degree 4.
#  Marginalising h gives p(s) -> softmax(l) as alpha grows.
#  NOTHING IS TRAINED. All params are closed-form in the frozen logits l.
# ==========================================================================
import math, json, time
import torch, torch.nn.functional as Fn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers.models.gpt2.modeling_gpt2 as G

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
DT = torch.float32

S       = 512
N_SINK  = 4
W       = 12
M       = N_SINK + W          # 16 support slots -> b = 4
B_BITS  = int(math.log2(M))
NEG     = -30.0               # finite stand-in for -inf, keeps precompensation stable

CODES = torch.tensor([[1.0 if (j >> i) & 1 else -1.0 for i in range(B_BITS)]
                      for j in range(M)], device=dev, dtype=DT)      # (M, b)
GRAM  = CODES @ CODES.T                                              # (M, M)


def support_index(S, n_sink, w, device):
    """(S, M) gather indices + (S, M) validity mask. Causal, duplicates removed."""
    i   = torch.arange(S, device=device)
    sk  = torch.arange(n_sink, device=device).expand(S, n_sink)
    loc = i[:, None] - torch.arange(w - 1, -1, -1, device=device)[None, :]
    idx = torch.cat([sk, loc], 1)                                    # (S, M)
    ok  = (idx >= 0) & (idx <= i[:, None])
    dup = (idx[:, n_sink:, None] == idx[:, None, :n_sink]).any(-1)
    ok = ok.clone()
    ok[:, n_sink:] &= ~dup
    return idx.clamp(min=0), ok


def rbm_marginal(lp, alpha):
    """Exact visible marginal of the RBM by enumerating all 2^b = M states."""
    u = alpha * (GRAM - (B_BITS - 1)) + lp[:, None, :]               # (R, M, M)
    F = -Fn.softplus(u).sum(-1)                                      # (R, M)
    return torch.softmax(-F, -1)


def precompensate(l, alpha, iters):
    """Solve for l' with RBM_alpha(l') == softmax(l). Host-side O(m^2) per row."""
    tgt = torch.softmax(l, -1)
    lt  = torch.log(tgt.clamp_min(1e-30))
    lp  = l.clone()
    for _ in range(iters):
        p  = rbm_marginal(lp, alpha).clamp_min(1e-30)
        lp = lp + (lt - torch.log(p))
        lp = lp - lp.max(-1, keepdim=True).values
    return lp


def gibbs_probs(lp, alpha, n_burn, n_keep, gen):
    """2-block Gibbs on the bipartite RBM -> empirical visible distribution."""
    R = lp.shape[0]
    s = torch.where(torch.rand(R, B_BITS, device=dev, generator=gen) < 0.5, -1.0, 1.0).to(DT)
    pw = (2 ** torch.arange(B_BITS, device=dev)).to(DT)
    cnt = torch.zeros(R, M, device=dev, dtype=DT)
    one = torch.ones(R, 1, device=dev, dtype=DT)
    for t in range(n_burn + n_keep):
        u = alpha * (s @ CODES.T - (B_BITS - 1)) + lp                # (R, M)  hiddens | visibles
        h = (torch.rand(R, M, device=dev, generator=gen) < torch.sigmoid(u)).to(DT)
        f = alpha * (h @ CODES)                                      # (R, b)  visibles | hiddens
        s = torch.where(torch.rand(R, B_BITS, device=dev, generator=gen)
                        < torch.sigmoid(2 * f), 1.0, -1.0)
        if t >= n_burn:
            j = (((s > 0).to(DT)) * pw).sum(-1).long()
            cnt.scatter_add_(1, j[:, None], one)
    return cnt / n_keep


MODE = dict(kind='baseline')


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    Bs, H, Sq, D = query.shape
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    logits = (query @ key.transpose(-1, -2)) * sc                    # (B,H,S,S)
    causal = torch.ones(Sq, Sq, device=query.device, dtype=torch.bool).tril()
    logits = logits.masked_fill(~causal, float('-inf'))
    k = MODE['kind']
    if k == 'baseline':
        w = torch.softmax(logits, -1)
        return (w @ value).transpose(1, 2).contiguous(), w

    idx, ok = support_index(Sq, N_SINK, W, query.device)              # (S,M)
    okR = ok.expand(Bs, H, Sq, M).reshape(-1, M)
    li = logits.gather(-1, idx.expand(Bs, H, Sq, M))                  # (B,H,S,M)
    li = li.masked_fill(~ok, NEG).clamp_min(NEG)
    R  = Bs * H * Sq
    lf = li.reshape(R, M)

    if k == 'exact':
        p = torch.softmax(lf, -1)
    elif k == 'random':
        p = torch.softmax(torch.randn_like(lf).masked_fill(~okR, NEG), -1)
    elif k == 'rbm':
        lp = precompensate(lf, MODE['alpha'], MODE['precomp']) if MODE['precomp'] else lf
        p  = gibbs_probs(lp, MODE['alpha'], MODE['burn'], MODE['keep'], MODE['gen'])
    else:
        raise ValueError(k)

    p = (p * okR).reshape(Bs, H, Sq, M)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
    vi = value[:, :, idx, :]                                          # (B,H,S,M,D)
    out = (p[..., None] * vi).sum(-2)
    return out.transpose(1, 2).contiguous(), p


G.eager_attention_forward = attn

tok = GPT2TokenizerFast.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(dev).eval()

TEXT = open('/tmp/eval.txt', encoding='utf-8').read()
ids  = tok(TEXT, return_tensors='pt').input_ids[:, :S].to(dev)
print('eval tokens:', tuple(ids.shape), 'device:', dev, flush=True)


@torch.no_grad()
def ppl():
    return float(torch.exp(mdl(ids, labels=ids).loss))


res = {}
MODE = dict(kind='baseline'); res['baseline_full_attention'] = ppl()
print('baseline (full exact attention)   ppl = %.3f' % res['baseline_full_attention'], flush=True)
MODE = dict(kind='exact');    res['support_exact_floor'] = ppl()
print('support-restricted EXACT (floor)  ppl = %.3f' % res['support_exact_floor'], flush=True)
MODE = dict(kind='random');   res['CONTROL_random'] = ppl()
print('CONTROL random weights on support ppl = %.3f   <- must be huge' % res['CONTROL_random'], flush=True)

print('\nTSU-sampled (bipartite RBM, 2-block Gibbs, closed-form params):', flush=True)
hdr = '%-8s%-10s%-7s%-7s%-10s%-10s%s' % ('alpha', 'precomp', 'burn', 'keep', 'ppl', 'x floor', 'x baseline')
print(hdr, flush=True)
for alpha, pc, burn, keep in [(3.0, 60, 50, 200), (5.0, 60, 50, 200), (5.0, 60, 100, 800),
                              (8.0, 60, 200, 800), (5.0, 0, 50, 200), (8.0, 60, 500, 2000)]:
    MODE = dict(kind='rbm', alpha=alpha, precomp=pc, burn=burn, keep=keep,
                gen=torch.Generator(device=dev).manual_seed(1234))
    t0 = time.time(); v = ppl(); dt = time.time() - t0
    res['rbm_a%s_pc%d_b%d_k%d' % (alpha, pc, burn, keep)] = v
    print('%-8s%-10s%-7s%-7s%-10.3f%-10.4f%.4f   (%.0fs)'
          % (alpha, pc, burn, keep, v, v / res['support_exact_floor'],
             v / res['baseline_full_attention'], dt), flush=True)

json.dump(res, open('/tmp/tsu_attn.json', 'w'), indent=2)
print('\nwrote /tmp/tsu_attn.json', flush=True)
