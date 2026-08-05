"""
TEST E — is the gate's failure structural, or just a bad threshold?

Fair-play check on my own refuted hypothesis. If a sharper gate approaches
categorical performance, the idea was under-tuned. If sharpening does not help,
the failure is structural: independent Bernoulli gating cannot produce a PEAKED
distribution, because independent gates give either many positions on (a flat
average) or few (high variance) -- never softmax's concentration.
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
        "model emerges that accounts for what came before while predicting something new.")
ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
CFG = {"frac": 0.10, "K": 32, "mode": "gate"}
_orig = G.eager_attention_forward

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    if CFG["mode"] == "baseline":
        return out, p
    valid = (p > 0); L = p.shape[-1]; K = CFG["K"]
    k_top = max(1, int(CFG["frac"] * L))
    tau = p.topk(min(k_top, L), dim=-1).values[..., -1:]
    gp = torch.sigmoid(CFG.get("sharp", 1.0) *
                       (torch.log(p.clamp_min(1e-20)) - torch.log(tau.clamp_min(1e-20))))
    gp = gp * valid.to(p.dtype)
    acc = torch.zeros_like(p)
    for _ in range(K):
        h = (torch.rand_like(gp) < gp).to(p.dtype)
        acc += h / h.sum(-1, keepdim=True).clamp_min(1.0)
    q = acc / K
    q = (q / q.sum(-1, keepdim=True).clamp_min(1e-12)).to(value.dtype)
    return torch.matmul(q, value).transpose(1, 2).contiguous(), q
G.eager_attention_forward = patched

def ppl():
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())

CFG["mode"] = "baseline"; base = ppl()
print("=" * 62); print("TEST E — gate sharpness sweep (baseline %.2f)" % base); print("=" * 62)
print(f"{'top-frac':>9} {'sharpness':>10} {'K':>4} {'ppl':>10} {'xbase':>8}")
CFG["mode"] = "gate"
for frac in [0.10, 0.02, 0.005]:
    for sharp in [1.0, 4.0, 16.0]:
        CFG["frac"], CFG["sharp"] = frac, sharp
        v = np.mean([ppl() for _ in range(2)])
        print(f"{frac:9.3f} {sharp:10.1f} {CFG['K']:4d} {v:10.2f} {v/base:8.2f}x")
print()
print("reference: categorical K=32 -> 1.36x ;  random floor -> 85x")
