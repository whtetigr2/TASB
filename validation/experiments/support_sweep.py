# ==========================================================================
#  SELF-CHECK: does the "1.09x architectural floor" I reported earlier
#  actually hold at realistic sequence length, or was it an artefact of a
#  short S?  Pure-exact (no sampling) so this isolates the ARCHITECTURE.
#  Sweeps sequence length x n_sink x window.  No Gibbs anywhere in here.
# ==========================================================================
import math, json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers.models.gpt2.modeling_gpt2 as G

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
CFG = dict(kind='baseline', n_sink=0, w=0)


def support_mask(S, n_sink, w, device):
    i = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    return (j <= i) & ((j < n_sink) | ((i - j) < w))


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    Bs, H, Sq, D = query.shape
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    logits = (query @ key.transpose(-1, -2)) * sc
    causal = torch.ones(Sq, Sq, device=query.device, dtype=torch.bool).tril()
    if CFG['kind'] == 'baseline':
        m = causal
    else:
        m = support_mask(Sq, CFG['n_sink'], CFG['w'], query.device)
    logits = logits.masked_fill(~m, float('-inf'))
    w = torch.softmax(logits, -1)
    return (w @ value).transpose(1, 2).contiguous(), w


G.eager_attention_forward = attn
tok = GPT2TokenizerFast.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(dev).eval()
FULL = tok(open('/tmp/eval.txt', encoding='utf-8').read(), return_tensors='pt').input_ids


@torch.no_grad()
def ppl(S):
    ids = FULL[:, :S].to(dev)
    return float(torch.exp(mdl(ids, labels=ids).loss))


@torch.no_grad()
def mass(S, n_sink, w):
    """Fraction of true attention mass captured by the support (layer-averaged)."""
    ids = FULL[:, :S].to(dev)
    CFG.update(kind='baseline')
    o = mdl(ids, output_attentions=True)
    m = support_mask(S, n_sink, w, dev)
    tot, n = 0.0, 0
    for a in o.attentions:                       # (B,H,S,S)
        tot += float((a * m).sum(-1).mean()); n += 1
    return tot / n


res = {}
for S in (128, 256, 512, 1024):
    CFG.update(kind='baseline')
    base = ppl(S)
    print('\n=== S=%d   baseline ppl = %.3f ===' % (S, base), flush=True)
    print('%-8s%-6s%-8s%-12s%-10s%s' % ('n_sink', 'w', 'm', 'mass', 'ppl', 'x baseline'), flush=True)
    for n_sink, w in [(0, 16), (4, 12), (4, 16), (4, 28), (4, 60), (4, 124), (8, 56), (16, 48)]:
        CFG.update(kind='sup', n_sink=n_sink, w=w)
        v = ppl(S)
        CFG.update(kind='sup', n_sink=n_sink, w=w)
        mm = mass(S, n_sink, w)
        res['S%d_s%d_w%d' % (S, n_sink, w)] = dict(ppl=v, ratio=v / base, mass=mm)
        print('%-8d%-6d%-8d%-12.4f%-10.3f%.3f' % (n_sink, w, n_sink + w, mm, v, v / base), flush=True)

json.dump(res, open('/tmp/support_sweep.json', 'w'), indent=2)
print('\nwrote /tmp/support_sweep.json')
