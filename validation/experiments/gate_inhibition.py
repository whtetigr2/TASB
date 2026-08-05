# ==========================================================================
#  RAW-QK GATE + LATERAL INHIBITION.
#
#  The raw gate fails end-to-end (14.9x) because it has no per-row
#  normaliser, and the failure is BIAS not variance (K=8..512 barely moves).
#  top-k rescues it only by supplying tau = a per-row statistic, i.e. the
#  partition function in disguise -- which needs the host.
#
#  But a normaliser does not have to come from the host.  Add a global
#  inhibition term to the ENERGY:
#
#      E(h) = - sum_j f_j h_j  +  (lambda/2) (sum_j h_j)^2 ,   f_j = a*l_j + b
#
#  Gibbs conditional:  p(h_j=1 | rest) = sigmoid( f_j - lambda*(n_-j + 1/2) )
#
#  This is self-normalising ON DEVICE: the more gates that fire, the harder
#  it is for another to fire, so the active count self-regulates to the row's
#  logit scale without anyone computing a max or a log-sum-exp.  It is a
#  uniform all-to-all coupling, implementable with a counter/adder tree
#  rather than S^2 wires.
#
#  Crucially this preserves the property that motivated the whole idea: the
#  field f_j = a*(q.k_j)/sqrt(d) + b is a LOCAL field, so with couplings
#  W_ji = a*k_j[i]/sqrt(d) and q clamped, the device computes the dot product
#  itself.  Softmax cannot be expressed this way; this can.
#
#  Controls: lambda=0 must reproduce the plain raw gate's failure.
# ==========================================================================
import math, json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers.models.gpt2.modeling_gpt2 as G

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
S = 512
CFG = dict(kind='baseline')


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    Bs, H, Sq, D = query.shape
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    scores = (query @ key.transpose(-1, -2)) * sc
    causal = torch.ones(Sq, Sq, device=query.device, dtype=torch.bool).tril()
    if CFG['kind'] == 'baseline':
        p = torch.softmax(scores.masked_fill(~causal, float('-inf')), -1)
        return (p @ value).transpose(1, 2).contiguous(), p

    a, b, lam = CFG['a'], CFG['b'], CFG['lam']
    sweeps, K = CFG['sweeps'], CFG['K']
    f = (a * scores + b).masked_fill(~causal, -1e4)
    acc = torch.zeros_like(f)
    for _ in range(K):
        h = torch.zeros_like(f)
        for _ in range(sweeps):
            n = h.sum(-1, keepdim=True)                       # global count
            # n_-j = n - h_j  (each node sees the others' count)
            p1 = torch.sigmoid(f - lam * (n - h + 0.5))
            h = (torch.rand_like(f) < p1).to(f.dtype) * causal
        acc += h / h.sum(-1, keepdim=True).clamp_min(1.0)
    p = acc / K
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return (p.to(value.dtype) @ value).transpose(1, 2).contiguous(), p


G.eager_attention_forward = attn
tok = GPT2TokenizerFast.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(dev).eval()
ids = tok(open('/tmp/eval.txt', encoding='utf-8').read(), return_tensors='pt').input_ids[:, :S].to(dev)


@torch.no_grad()
def ppl():
    return float(torch.exp(mdl(ids, labels=ids).loss))


res = {}
CFG = dict(kind='baseline'); base = ppl(); res['baseline'] = base
print('S=%d   baseline ppl = %.3f' % (S, base), flush=True)
CFG = dict(kind='inh', a=1.0, b=-4.0, lam=0.0, sweeps=4, K=32)
v = ppl(); res['CONTROL_lambda0'] = v
print('CONTROL lambda=0 (no inhibition)  ppl = %.2f  %.1fx  <- must match the plain raw gate\n'
      % (v, v / base), flush=True)

print('%-7s%-8s%-8s%-9s%-6s%-12s%s' % ('a', 'b', 'lambda', 'sweeps', 'K', 'ppl', 'x baseline'), flush=True)
best = (1e9, None)
for a, b, lam, sw in [(1.0, 0.0, 0.5, 4), (1.0, 0.0, 1.0, 4), (1.0, 2.0, 1.0, 4),
                      (2.0, 0.0, 1.0, 4), (2.0, 2.0, 1.0, 4), (2.0, 4.0, 2.0, 4),
                      (1.0, 4.0, 2.0, 4), (4.0, 4.0, 2.0, 4), (2.0, 2.0, 0.5, 8),
                      (2.0, 4.0, 1.0, 8), (4.0, 8.0, 4.0, 8), (2.0, 8.0, 4.0, 8)]:
    CFG = dict(kind='inh', a=a, b=b, lam=lam, sweeps=sw, K=32)
    v = ppl(); res['inh_a%s_b%s_l%s_s%d' % (a, b, lam, sw)] = v
    if v < best[0]:
        best = (v, (a, b, lam, sw))
    print('%-7s%-8s%-8s%-9d%-6d%-12.3f%.3f' % (a, b, lam, sw, 32, v, v / base), flush=True)

a, b, lam, sw = best[1]
print('\nK / sweep refinement at best (a=%s b=%s lambda=%s):' % (a, b, lam), flush=True)
print('%-9s%-6s%-12s%s' % ('sweeps', 'K', 'ppl', 'x baseline'), flush=True)
for sw2, K in [(4, 128), (8, 128), (16, 128), (8, 512)]:
    CFG = dict(kind='inh', a=a, b=b, lam=lam, sweeps=sw2, K=K)
    v = ppl(); res['ref_s%d_K%d' % (sw2, K)] = v
    print('%-9d%-6d%-12.3f%.3f' % (sw2, K, v, v / base), flush=True)

json.dump(res, open('/tmp/gate_inh.json', 'w'), indent=2)
print('\nwrote /tmp/gate_inh.json')
