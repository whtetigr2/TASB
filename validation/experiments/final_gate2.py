"""
Does a CHEAP per-row statistic recover the gate, and what does that cost?

The global-b gate failed (12-30x). The working version used a per-row top-k
threshold. Question: is the requirement specifically top-k, or does any per-row
location/scale statistic do? Mean and std need no exp and no sort -- but they are
still a global reduction over the row, so if they work, the honest conclusion is
that the gate does NOT escape the global reduction, only the exponential.

  z-score gate : g = sigmoid(a*(l - mean_row)/std_row + b)   [mean,std: cheap]
  topk gate    : g = sigmoid(l - l_tau)                      [sort: costlier]
  categorical  : thermobridge's current approach
"""
import torch, numpy as np, math
import transformers.models.gpt2.modeling_gpt2 as G
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
TEXT = ("The history of scientific discovery is marked by long periods of incremental "
        "progress punctuated by sudden conceptual shifts. Researchers accumulate anomalies "
        "that resist explanation under the prevailing framework, and eventually a new "
        "model emerges that accounts for what came before while predicting something new. "
        "Thermodynamics followed this pattern, as did statistical mechanics and later "
        "information theory, each reframing quantities the previous era treated as basic.")
ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
CFG = {"mode": "baseline", "K": 32, "a": 2.0, "b": -1.5, "frac": 0.02}
_orig = G.eager_attention_forward

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    m = CFG["mode"]
    if m == "baseline":
        return out, p
    valid = (p > 0); L = p.shape[-1]; K = CFG["K"]
    if m == "categorical":
        flat = p.reshape(-1, L).clamp_min(1e-12)
        idx = torch.multinomial(flat, K, replacement=True)
        q = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K).reshape(p.shape)
    else:
        scores = torch.matmul(query, key.transpose(-1, -2))
        if getattr(module, "scale_attn_weights", True):
            scores = scores / (float(value.size(-1)) ** 0.5)
        vmask = valid.to(scores.dtype)
        if m == "zscore":
            n = vmask.sum(-1, keepdim=True).clamp_min(1.0)
            mu = (scores * vmask).sum(-1, keepdim=True) / n
            var = (((scores - mu) ** 2) * vmask).sum(-1, keepdim=True) / n
            z = (scores - mu) / var.sqrt().clamp_min(1e-6)
            gp = torch.sigmoid(CFG["a"] * z + CFG["b"]) * vmask
        elif m == "topk":
            big = torch.finfo(scores.dtype).min / 2
            sm = scores.masked_fill(~valid, big)
            kt = max(1, int(CFG["frac"] * L))
            tau = sm.topk(min(kt, L), dim=-1).values[..., -1:]
            gp = torch.sigmoid(sm - tau) * vmask
        acc = torch.zeros_like(p)
        for _ in range(K):
            h = (torch.rand_like(gp) < gp).to(p.dtype)
            acc += h / h.sum(-1, keepdim=True).clamp_min(1.0)
        q = acc / K
    q = (q / q.sum(-1, keepdim=True).clamp_min(1e-12)).to(value.dtype)
    return torch.matmul(q, value).transpose(1, 2).contiguous(), q
G.eager_attention_forward = patched

def ppl(n=3):
    with torch.no_grad():
        return float(np.mean([math.exp(model(ids, labels=ids).loss.item()) for _ in range(n)]))

CFG["mode"] = "baseline"; base = ppl(1)
print("=" * 72); print(f"per-row statistic sweep — baseline {base:.2f}"); print("=" * 72)
CFG["K"] = 32
CFG["mode"] = "categorical"; c = ppl()
print(f"  categorical (K=32)            : {c:8.2f}  {c/base:6.2f}x")
CFG["mode"] = "topk"
for f in [0.02, 0.05]:
    CFG["frac"] = f; v = ppl()
    print(f"  topk gate frac={f:<5}          : {v:8.2f}  {v/base:6.2f}x")
CFG["mode"] = "zscore"
for a in [1.0, 2.0, 3.0]:
    for b in [0.0, -1.5, -3.0]:
        CFG["a"], CFG["b"] = a, b
        v = ppl()
        print(f"  zscore gate a={a:<4} b={b:<5}      : {v:8.2f}  {v/base:6.2f}x")
