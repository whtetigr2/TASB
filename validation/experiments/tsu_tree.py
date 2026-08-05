# ==========================================================================
#  GENERAL TREE-FACTORED TSU ATTENTION.
#
#  p(j) = prod_k p(g_k | g_1..g_{k-1}),  an EXACT chain-rule factorisation of
#  the attention softmax over a sparse support of size m = prod_k f_k.
#  Every node of the tree is an independent bipartite RBM:
#      b_k = log2(f_k) visible p-bits  +  f_k hidden p-bits
#      visible-degree = f_k,  hidden-degree = b_k
#  So max degree is set by the BRANCHING FACTOR, not by m.  Branching 4 =>
#  degree 4, comfortably inside the paper's stated "in most cases, 12".
#
#  All nodes at all levels are sampled simultaneously (this is what the
#  hardware does -- every node exists in silicon at once), then the path
#  probabilities are multiplied.  Nothing is trained; every parameter is
#  closed-form in the frozen model's logits.
# ==========================================================================
import math, json, time
import torch, torch.nn.functional as Fn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers.models.gpt2.modeling_gpt2 as G

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
DT = torch.float32
NEG = -30.0
S, N_SINK, W = 512, 4, 60
M = N_SINK + W


def codes(n):
    b = int(math.log2(n))
    return torch.tensor([[1.0 if (j >> i) & 1 else -1.0 for i in range(b)]
                         for j in range(n)], device=dev, dtype=DT), b


CACHE = {}


def geom(n):
    if n not in CACHE:
        C, b = codes(n); CACHE[n] = (C, b, C @ C.T)
    return CACHE[n]


def precompensate(l, alpha, n, iters=50):
    C, b, gram = geom(n)
    tgt = torch.softmax(l, -1); lt = torch.log(tgt.clamp_min(1e-30))
    lp = l.clone()
    for _ in range(iters):
        u = alpha * (gram - (b - 1)) + lp[:, None, :]
        p = torch.softmax(Fn.softplus(u).sum(-1), -1).clamp_min(1e-30)
        lp = lp + (lt - torch.log(p)); lp = lp - lp.max(-1, keepdim=True).values
    return lp


def gibbs(lp, alpha, n, n_burn, n_keep, gen):
    C, b, _ = geom(n)
    R = lp.shape[0]
    s = torch.where(torch.rand(R, b, device=dev, generator=gen) < 0.5, -1.0, 1.0).to(DT)
    pw = (2 ** torch.arange(b, device=dev)).to(DT)
    cnt = torch.zeros(R, n, device=dev, dtype=DT); one = torch.ones(R, 1, device=dev, dtype=DT)
    for t in range(n_burn + n_keep):
        u = alpha * (s @ C.T - (b - 1)) + lp
        h = (torch.rand(R, n, device=dev, generator=gen) < torch.sigmoid(u)).to(DT)
        f = alpha * (h @ C)
        s = torch.where(torch.rand(R, b, device=dev, generator=gen) < torch.sigmoid(2 * f), 1.0, -1.0)
        if t >= n_burn:
            cnt.scatter_add_(1, ((((s > 0).to(DT)) * pw).sum(-1).long())[:, None], one)
    return cnt / n_keep


def tree_probs(lf, fs, alpha, nc, burn, keep, gen):
    """lf: (R, M).  Returns (R, M) empirical distribution via the tree."""
    R = lf.shape[0]
    lt = lf.reshape(R, *fs)                      # (R, f1, f2, ..., fL)
    L = len(fs)
    out = torch.ones(R, nc, *fs, device=dev, dtype=DT)
    for k in range(L):
        # logits at level k: LSE over all deeper dims, conditioned on the prefix
        lk = lt
        for _ in range(L - k - 1):
            lk = torch.logsumexp(lk, -1)         # (R, f1..f_{k+1})
        nprefix = int(torch.tensor(fs[:k]).prod().item()) if k else 1
        flat = lk.reshape(R * nprefix, fs[k])
        lp = precompensate(flat, alpha, fs[k])
        lpr = lp[:, None, :].expand(-1, nc, -1).reshape(-1, fs[k])
        pk = gibbs(lpr, alpha, fs[k], burn, keep, gen).reshape(R, nprefix, nc, fs[k])
        # broadcast into (R, nc, f1..fL): prefix dims, then this level, then trailing
        pk = pk.permute(0, 2, 1, 3).reshape(R, nc, *fs[:k], fs[k])
        shape = (R, nc) + fs[:k + 1] + (1,) * (L - k - 1)
        out = out * pk.reshape(shape)
    return out.mean(1).reshape(R, M)


def support_index(S, n_sink, w, device):
    i = torch.arange(S, device=device)
    sk = torch.arange(n_sink, device=device).expand(S, n_sink)
    loc = i[:, None] - torch.arange(w - 1, -1, -1, device=device)[None, :]
    idx = torch.cat([sk, loc], 1)
    ok = (idx >= 0) & (idx <= i[:, None])
    dup = (idx[:, n_sink:, None] == idx[:, None, :n_sink]).any(-1)
    ok = ok.clone(); ok[:, n_sink:] &= ~dup
    return idx.clamp(min=0), ok


MODE = dict(kind='baseline')


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    Bs, H, Sq, D = query.shape
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    logits = ((query @ key.transpose(-1, -2)) * sc).masked_fill(
        ~torch.ones(Sq, Sq, device=query.device, dtype=torch.bool).tril(), float('-inf'))
    if MODE['kind'] == 'baseline':
        w = torch.softmax(logits, -1)
        return (w @ value).transpose(1, 2).contiguous(), w
    idx, ok = support_index(Sq, N_SINK, W, query.device)
    okR = ok.expand(Bs, H, Sq, M).reshape(-1, M)
    lf = logits.gather(-1, idx.expand(Bs, H, Sq, M)).masked_fill(~ok, NEG).clamp_min(NEG).reshape(-1, M)
    if MODE['kind'] == 'exact':
        p = torch.softmax(lf, -1)
    else:
        p = tree_probs(lf, MODE['fs'], MODE['alpha'], MODE['chains'],
                       MODE['burn'], MODE['keep'], MODE['gen'])
    p = (p * okR).reshape(Bs, H, Sq, M)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
    return (p[..., None] * value[:, :, idx, :]).sum(-2).transpose(1, 2).contiguous(), p


G.eager_attention_forward = attn
tok = GPT2TokenizerFast.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(dev).eval()
ids = tok(open('/tmp/eval.txt', encoding='utf-8').read(), return_tensors='pt').input_ids[:, :S].to(dev)


@torch.no_grad()
def ppl():
    return float(torch.exp(mdl(ids, labels=ids).loss))


res = {}
MODE = dict(kind='baseline'); res['baseline'] = ppl()
MODE = dict(kind='exact');    res['floor'] = ppl()
print('S=%d  m=%d (%d sink + %d local)' % (S, M, N_SINK, W))
print('baseline %.3f | support exact floor %.3f (%.3fx)\n' % (res['baseline'], res['floor'], res['floor'] / res['baseline']), flush=True)
print('%-14s%-8s%-9s%-8s%-10s%-10s%s' % ('tree', 'maxdeg', 'pbits/row', 'chains', 'ppl', 'x floor', 'x base'), flush=True)
for fs, nc, burn, keep in [((4, 16), 32, 30, 60), ((8, 8), 32, 30, 60), ((4, 4, 4), 32, 30, 60),
                           ((2, 2, 2, 2, 2, 2), 32, 30, 60), ((4, 4, 4), 64, 30, 60),
                           ((4, 4, 4), 128, 30, 60)]:
    nodes = [int(torch.tensor(fs[:k]).prod().item()) if k else 1 for k in range(len(fs))]
    pbits = sum(n * (int(math.log2(f)) + f) for n, f in zip(nodes, fs))
    MODE = dict(kind='tree', fs=fs, alpha=5.0, chains=nc, burn=burn, keep=keep,
                gen=torch.Generator(device=dev).manual_seed(7))
    t0 = time.time(); v = ppl(); dt = time.time() - t0
    res['tree_%s_c%d' % ('x'.join(map(str, fs)), nc)] = v
    print('%-14s%-8d%-9d%-8d%-10.3f%-10.4f%-8.4f (%.0fs)'
          % ('x'.join(map(str, fs)), max(fs), pbits, nc, v, v / res['floor'], v / res['baseline'], dt), flush=True)

json.dump(res, open('/tmp/tsu_tree.json', 'w'), indent=2)
print('\nwrote /tmp/tsu_tree.json')
