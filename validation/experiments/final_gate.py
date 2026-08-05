"""
END-TO-END — compiled gate on a frozen model, driven by RAW LOGITS only.

Critical property being checked: the compiled kernel g = sigmoid(a*l + b) is a
function of the raw attention scores. It never forms the partition function.
Earlier patches derived the gate from post-softmax p, which would have smuggled
Z back in; here the scores are recomputed inside the patch and softmax is never
used on the gate path.

Compare, identical text and budget:
  baseline | categorical (thermobridge's current approach) | compiled gate | random floor
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
CFG = {"mode": "baseline", "K": 32, "a": 1.07, "b": -4.3}
_orig = G.eager_attention_forward
USED_SOFTMAX_ON_GATE_PATH = {"flag": False}

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    m = CFG["mode"]
    if m == "baseline":
        return out, p
    valid = (p > 0); L = p.shape[-1]; K = CFG["K"]
    if m == "random":
        q = valid.to(p.dtype)
    elif m == "categorical":
        flat = p.reshape(-1, L).clamp_min(1e-12)
        idx = torch.multinomial(flat, K, replacement=True)
        q = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K).reshape(p.shape)
    elif m == "gate":
        # RAW SCORES -- recomputed here, no softmax, no partition function
        scores = torch.matmul(query, key.transpose(-1, -2))
        if getattr(module, "scale_attn_weights", True):
            scores = scores / (float(value.size(-1)) ** 0.5)
        gp = torch.sigmoid(CFG["a"] * scores + CFG["b"]) * valid.to(scores.dtype)
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
CFG["mode"] = "random";   floor = ppl(1)
print("=" * 76)
print(f"FINAL — baseline {base:.2f}   random floor {floor:.1f}   (a={CFG['a']})")
print("=" * 76)
print(f"{'K':>5} {'categorical':>12} {'gate b=-3.8':>12} {'gate b=-4.3':>12} {'gate b=-4.8':>12}")
for K in [8, 32, 128]:
    CFG["K"] = K
    CFG["mode"] = "categorical"; c = ppl()
    row = []
    for b in [-3.8, -4.3, -4.8]:
        CFG["mode"] = "gate"; CFG["b"] = b
        row.append(ppl())
    print(f"{K:5d} {c:12.2f} {row[0]:12.2f} {row[1]:12.2f} {row[2]:12.2f}")
    print(f"{'':5} {c/base:11.2f}x {row[0]/base:11.2f}x {row[1]/base:11.2f}x {row[2]/base:11.2f}x")
print()
print("gate path uses RAW SCORES only -- Z is never formed.")
