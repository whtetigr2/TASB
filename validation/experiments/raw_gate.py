# ==========================================================================
#  RAW-QK GATE, END-TO-END.  The gap in the record.
#
#  Earlier work scored the raw gate only DISTRIBUTIONALLY (KL/TV vs softmax)
#  and scored only the z-score / top-k variants end-to-end.  The z-score
#  variant failed (16-53x) and that failure got generalised to "the gate is
#  refuted" -- which does not follow.  This tests the RAW variant on
#  perplexity, which is the thing that was never measured.
#
#  Why it matters more than the categorical route:
#      g_j = sigmoid( a * (q.k_j)/sqrt(d) + b )
#  A p-bit's field in an Ising model is  sum_i W_ji s_i + b_j,  and the DTCA's
#  RNG bias "is constrained to be a sigmoidal function of an input voltage".
#  So with couplings W_ji = a*k_j[i]/sqrt(d) and a CLAMPED block holding q,
#  the dot product is computed by the device's own local communication.
#  The categorical route cannot do this: softmax needs a normaliser across
#  positions, which is not a local field.
#
#  Readout is self-normalising:  p_j = E[ h_j / sum_k h_k ],  so no partition
#  function is ever formed on-device.
#
#  Controls: exact softmax (baseline), random gate, and a=0 (field ignored).
# ==========================================================================
import math, json, time
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
    k = CFG['kind']
    if k == 'baseline':
        p = torch.softmax(scores.masked_fill(~causal, float('-inf')), -1)
    else:
        a, b, K = CFG['a'], CFG['b'], CFG['K']
        if CFG.get('lencorr'):
            # b_i = b - c*ln(i+1).  Depends only on POSITION INDEX, which is
            # free -- it needs no logits, no sort, no per-row statistic.
            i = torch.arange(Sq, device=query.device, dtype=scores.dtype)
            bb = b - CFG['c'] * torch.log(i + 1.0)
            bb = bb[None, None, :, None]
        else:
            bb = b
        if k == 'randgate':
            g = torch.sigmoid(torch.randn_like(scores) * a + bb)
        else:
            g = torch.sigmoid(a * scores + bb)
        g = g * causal
        acc = torch.zeros_like(g)
        for _ in range(K):
            h = (torch.rand_like(g) < g).to(g.dtype)
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
print('S=%d   baseline (exact softmax, FULL attention) ppl = %.3f\n' % (S, base), flush=True)

print('CONTROLS:', flush=True)
CFG = dict(kind='randgate', a=1.0, b=-4.0, K=32); v = ppl(); res['CONTROL_random_gate'] = v
print('  random gate (field replaced by noise)  ppl = %10.2f  %8.1fx' % (v, v / base), flush=True)
CFG = dict(kind='gate', a=0.0, b=-4.0, K=32); v = ppl(); res['CONTROL_a0'] = v
print('  a=0 (field ignored, uniform gate)      ppl = %10.2f  %8.1fx' % (v, v / base), flush=True)

print('\nRAW-QK GATE  g = sigmoid(a*(q.k)/sqrt(d) + b),  global (a,b), K draws:', flush=True)
print('%-7s%-8s%-6s%-12s%s' % ('a', 'b', 'K', 'ppl', 'x baseline'), flush=True)
for a, b, K in [(1.0, -4.0, 32), (1.0, -6.0, 32), (2.0, -4.0, 32), (2.0, -6.0, 32),
                (1.07, -4.45, 32), (1.0, -8.0, 32), (2.0, -8.0, 32), (4.0, -8.0, 32)]:
    CFG = dict(kind='gate', a=a, b=b, K=K); v = ppl()
    res['raw_a%s_b%s_K%d' % (a, b, K)] = v
    print('%-7s%-8s%-6d%-12.3f%.3f' % (a, b, K, v, v / base), flush=True)

print('\nWITH FREE LENGTH CORRECTION  b_i = b - c*ln(i+1)  (position index only):', flush=True)
print('%-7s%-8s%-7s%-6s%-12s%s' % ('a', 'b', 'c', 'K', 'ppl', 'x baseline'), flush=True)
best = (1e9, None)
for a, b, c, K in [(1.0, -3.0, 1.0, 32), (2.0, -3.0, 1.0, 32), (2.0, -2.0, 1.0, 32),
                   (2.0, -3.0, 0.5, 32), (4.0, -3.0, 1.0, 32), (4.0, -2.0, 1.0, 32),
                   (4.0, -4.0, 1.0, 32), (8.0, -3.0, 1.0, 32)]:
    CFG = dict(kind='gate', a=a, b=b, c=c, K=K, lencorr=True); v = ppl()
    res['len_a%s_b%s_c%s_K%d' % (a, b, c, K)] = v
    if v < best[0]:
        best = (v, (a, b, c))
    print('%-7s%-8s%-7s%-6d%-12.3f%.3f' % (a, b, c, K, v, v / base), flush=True)

a, b, c = best[1]
print('\nK-sweep at the best setting (a=%s, b=%s, c=%s):' % (a, b, c), flush=True)
print('%-6s%-12s%s' % ('K', 'ppl', 'x baseline'), flush=True)
for K in (8, 32, 128, 512):
    CFG = dict(kind='gate', a=a, b=b, c=c, K=K, lencorr=True); v = ppl()
    res['Ksweep_K%d' % K] = v
    print('%-6d%-12.3f%.3f' % (K, v, v / base), flush=True)

json.dump(res, open('/tmp/raw_gate.json', 'w'), indent=2)
print('\nwrote /tmp/raw_gate.json')
