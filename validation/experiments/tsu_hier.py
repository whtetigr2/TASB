# ==========================================================================
#  HIERARCHICAL TSU ATTENTION  -- degree-bounded by construction.
#
#  Problem found by measurement: a flat RBM embedding of an m-way attention
#  softmax needs visible-degree = m.  Fidelity at S=512 needs m ~ 64-128,
#  which blows through the DTCA degree budget (paper: "in most cases, 12").
#
#  Fix, following the paper's own thesis (App C.1 quadratic EBMs + the DTM
#  chain): do not ask ONE monolithic EBM to represent the whole categorical.
#  Factor it exactly:
#        p(j) = p(g) * p(j | g),      g = group(j)
#        p(g) prop exp( LSE_{j in g} l_j )
#  Stage 1: choose group g   -> b1=log2(G) visibles + G hiddens
#  Stage 2: choose j within g -> b2=log2(m/G) visibles + m/G hiddens
#  Both stages are bipartite, pairwise, binary, and degree <= 16.
#  The factorisation is EXACT -- no approximation, nothing trained.
#  Stage 2 is conditioned on stage 1 exactly as the DTM conditions each
#  denoising step on the previous ("blue nodes ... stay fixed throughout
#  the Gibbs sampling").  This is a 2-step compiled Gibbs chain.
#
#  Also switches from ONE long chain to n_chains PARALLEL replicas, which is
#  how the hardware would actually be used (>=500k p-bits on a Z1 stick) and
#  which converts autocorrelation into raw parallelism.
# ==========================================================================
import math, json, time
import torch, torch.nn.functional as Fn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers.models.gpt2.modeling_gpt2 as G

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
DT = torch.float32
NEG = -30.0

S      = 512
N_SINK = 4
W      = 60
M      = N_SINK + W           # 64 support slots
GRP    = 4                    # 4 groups of 16
GSZ    = M // GRP


def codes(n):
    b = int(math.log2(n))
    return torch.tensor([[1.0 if (j >> i) & 1 else -1.0 for i in range(b)]
                         for j in range(n)], device=dev, dtype=DT), b


C1, B1 = codes(GRP)           # stage 1: (4,2)
C2, B2 = codes(GSZ)           # stage 2: (16,4)
G1, G2 = C1 @ C1.T, C2 @ C2.T


def rbm_marginal(lp, alpha, gram, b):
    u = alpha * (gram - (b - 1)) + lp[:, None, :]
    return torch.softmax(Fn.softplus(u).sum(-1), -1)


def precompensate(l, alpha, gram, b, iters=60):
    tgt = torch.softmax(l, -1)
    lt  = torch.log(tgt.clamp_min(1e-30))
    lp  = l.clone()
    for _ in range(iters):
        p  = rbm_marginal(lp, alpha, gram, b).clamp_min(1e-30)
        lp = lp + (lt - torch.log(p))
        lp = lp - lp.max(-1, keepdim=True).values
    return lp


def gibbs_last(lp, alpha, C, b, n_burn, n_keep, gen):
    """2-block Gibbs; returns empirical distribution over the n states."""
    R, n = lp.shape
    s = torch.where(torch.rand(R, b, device=dev, generator=gen) < 0.5, -1.0, 1.0).to(DT)
    pw = (2 ** torch.arange(b, device=dev)).to(DT)
    cnt = torch.zeros(R, n, device=dev, dtype=DT)
    one = torch.ones(R, 1, device=dev, dtype=DT)
    for t in range(n_burn + n_keep):
        u = alpha * (s @ C.T - (b - 1)) + lp
        h = (torch.rand(R, n, device=dev, generator=gen) < torch.sigmoid(u)).to(DT)
        f = alpha * (h @ C)
        s = torch.where(torch.rand(R, b, device=dev, generator=gen) < torch.sigmoid(2 * f), 1.0, -1.0)
        if t >= n_burn:
            j = (((s > 0).to(DT)) * pw).sum(-1).long()
            cnt.scatter_add_(1, j[:, None], one)
    return cnt / n_keep


def support_index(S, n_sink, w, device):
    i   = torch.arange(S, device=device)
    sk  = torch.arange(n_sink, device=device).expand(S, n_sink)
    loc = i[:, None] - torch.arange(w - 1, -1, -1, device=device)[None, :]
    idx = torch.cat([sk, loc], 1)
    ok  = (idx >= 0) & (idx <= i[:, None])
    dup = (idx[:, n_sink:, None] == idx[:, None, :n_sink]).any(-1)
    ok = ok.clone(); ok[:, n_sink:] &= ~dup
    return idx.clamp(min=0), ok


MODE = dict(kind='baseline')


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    Bs, H, Sq, D = query.shape
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    logits = (query @ key.transpose(-1, -2)) * sc
    causal = torch.ones(Sq, Sq, device=query.device, dtype=torch.bool).tril()
    logits = logits.masked_fill(~causal, float('-inf'))
    k = MODE['kind']
    if k == 'baseline':
        w = torch.softmax(logits, -1)
        return (w @ value).transpose(1, 2).contiguous(), w

    idx, ok = support_index(Sq, N_SINK, W, query.device)
    okR = ok.expand(Bs, H, Sq, M).reshape(-1, M)
    lf  = logits.gather(-1, idx.expand(Bs, H, Sq, M)).masked_fill(~ok, NEG).clamp_min(NEG).reshape(-1, M)
    R   = lf.shape[0]

    if k == 'exact':
        p = torch.softmax(lf, -1)
    elif k == 'random':
        p = torch.softmax(torch.randn_like(lf).masked_fill(~okR, NEG), -1)
    elif k == 'hier':
        a1, a2 = MODE['a1'], MODE['a2']
        nc, burn, keep = MODE['chains'], MODE['burn'], MODE['keep']
        gen = MODE['gen']
        lg = lf.reshape(R, GRP, GSZ)
        Lg = torch.logsumexp(lg, -1)                                  # (R,GRP) exact group logits
        lp1 = precompensate(Lg, a1, G1, B1)
        lp2 = precompensate(lg.reshape(R * GRP, GSZ), a2, G2, B2).reshape(R, GRP, GSZ)
        # stage 1, replicated across parallel chains
        lp1r = lp1[:, None, :].expand(R, nc, GRP).reshape(-1, GRP)
        pg   = gibbs_last(lp1r, a1, C1, B1, burn, keep, gen).reshape(R, nc, GRP)
        # stage 2: one chain per (row, chain, group); weight by stage-1 mass
        lp2r = lp2[:, None, :, :].expand(R, nc, GRP, GSZ).reshape(-1, GSZ)
        pj   = gibbs_last(lp2r, a2, C2, B2, burn, keep, gen).reshape(R, nc, GRP, GSZ)
        p    = (pg[..., None] * pj).sum(1).reshape(R, M) / nc
    else:
        raise ValueError(k)

    p = (p * okR).reshape(Bs, H, Sq, M)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
    vi = value[:, :, idx, :]
    return (p[..., None] * vi).sum(-2).transpose(1, 2).contiguous(), p


G.eager_attention_forward = attn
tok = GPT2TokenizerFast.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(dev).eval()
ids = tok(open('/tmp/eval.txt', encoding='utf-8').read(), return_tensors='pt').input_ids[:, :S].to(dev)


@torch.no_grad()
def ppl():
    return float(torch.exp(mdl(ids, labels=ids).loss))


print('S=%d  support m=%d (%d sink + %d local)  groups=%d x %d' % (S, M, N_SINK, W, GRP, GSZ))
print('p-bit budget/row: stage1 %d vis + %d hid (deg %d) | stage2 %d vis + %d hid (deg %d)  TOTAL %d'
      % (B1, GRP, GRP, B2, GSZ, GSZ, B1 + GRP + B2 + GSZ))
res = {}
MODE = dict(kind='baseline'); res['baseline'] = ppl(); print('\nbaseline           ppl = %.3f' % res['baseline'], flush=True)
MODE = dict(kind='exact');    res['floor']    = ppl(); print('support exact floor ppl = %.3f  (%.3fx)' % (res['floor'], res['floor']/res['baseline']), flush=True)
MODE = dict(kind='random');   res['control']  = ppl(); print('CONTROL random      ppl = %.1f' % res['control'], flush=True)

print('\nhierarchical TSU sampling (2-stage compiled Gibbs, parallel chains):', flush=True)
print('%-6s%-6s%-9s%-7s%-7s%-10s%-10s%s' % ('a1', 'a2', 'chains', 'burn', 'keep', 'ppl', 'x floor', 'x base'), flush=True)
for a1, a2, nc, burn, keep in [(5.,5.,8,20,20),(5.,5.,16,20,40),(5.,5.,32,30,60),
                               (8.,8.,32,30,60),(5.,8.,32,30,60),(5.,5.,64,30,80)]:
    MODE = dict(kind='hier', a1=a1, a2=a2, chains=nc, burn=burn, keep=keep,
                gen=torch.Generator(device=dev).manual_seed(7))
    t0 = time.time(); v = ppl(); dt = time.time() - t0
    res['hier_a%s_%s_c%d_b%d_k%d' % (a1, a2, nc, burn, keep)] = v
    print('%-6s%-6s%-9d%-7d%-7d%-10.3f%-10.4f%-8.4f (%.0fs)'
          % (a1, a2, nc, burn, keep, v, v / res['floor'], v / res['baseline'], dt), flush=True)

json.dump(res, open('/tmp/tsu_hier.json', 'w'), indent=2)
print('\nwrote /tmp/tsu_hier.json')
