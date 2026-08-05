"""
THE DECIDING QUESTION for fixed-topology hardware.

Attention is sparse (effective support ~8 of 90) with ~45% on a single sink.
Z1's connectivity is FIXED AT FABRICATION (degree 16, grid). So the question is
not "is the support small" -- it is "is the support the SAME set across queries".

  shared support  -> one fixed wiring serves every query -> Z1 can hold it
  per-query support -> wiring must change per token -> needs routing/multi-hop,
                       which fixed silicon cannot do without depth and latency

MEASURED, per layer/head:
  1 overlap of top-k attended positions between different query rows (Jaccard)
  2 how much mass a SINGLE fixed set of k positions captures, if chosen once per
    head (the best fixed wiring) vs chosen per query (the ideal adaptive wiring)
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
TEXT = ("The history of scientific discovery is marked by long periods of incremental "
        "progress punctuated by sudden conceptual shifts. Researchers accumulate anomalies "
        "that resist explanation under the prevailing framework, and eventually a new model "
        "emerges that accounts for what came before while predicting something new. "
        "Thermodynamics followed this pattern, as did statistical mechanics and later "
        "information theory, each reframing quantities the previous era treated as basic. "
        "The same dynamic appears whenever a measurement outruns the theory meant to explain it.")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def analyse(name, label, k=16):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").to(dev).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = m(ids, output_attentions=True)
    S = ids.shape[1]
    jac, fixed_mass, adapt_mass, rel_mass = [], [], [], []
    for A in out.attentions:
        a = A[0].float()                               # (H, S, S)
        H = a.shape[0]
        late = torch.arange(S // 2, S, device=a.device)   # rows with >=S/2 history
        for h in range(H):
            ah = a[h][late]                            # (n, S)
            kk = min(k, S)
            top = ah.topk(kk, dim=-1).indices          # (n, k) per-query support
            # 1 pairwise Jaccard between consecutive query rows
            for i in range(0, top.shape[0] - 1, 4):
                s1 = set(top[i].tolist()); s2 = set(top[i + 1].tolist())
                jac.append(len(s1 & s2) / len(s1 | s2))
            # 2 best FIXED set for this head = top-k by summed mass over queries
            fixed = ah.sum(0).topk(kk).indices
            fixed_mass.append(float(ah[:, fixed].sum(-1).mean()))
            adapt_mass.append(float(ah.gather(-1, top).sum(-1).mean()))
            # relative-position wiring: k most-used OFFSETS (what a grid could wire)
            offs = (torch.arange(len(late), device=a.device)[:, None] + (S // 2)
                    - torch.arange(S, device=a.device)[None, :])
            valid = offs >= 0
            om = torch.zeros(S, device=a.device)
            om.index_add_(0, offs[valid].clamp(0, S - 1), ah[valid])
            topo = om.topk(kk).indices
            sel = torch.isin(offs.clamp(0, S - 1), topo) & valid
            rel_mass.append(float((ah * sel).sum(-1).mean()))
    print(f"\n  {label}  (k={k}, S={S})")
    print(f"    top-k support overlap between adjacent queries (Jaccard) : {np.mean(jac):.3f}")
    print(f"    mass captured, ADAPTIVE per-query wiring                 : {np.mean(adapt_mass):.3f}")
    print(f"    mass captured, best FIXED absolute-position wiring       : {np.mean(fixed_mass):.3f}")
    print(f"    mass captured, best FIXED relative-offset wiring         : {np.mean(rel_mass):.3f}")

print("=" * 76)
print("IS THE SPARSE SUPPORT SHARED (fixed wiring) OR PER-QUERY (needs routing)?")
print("=" * 76)
for name, label in [("gpt2", "GPT-2 (absolute)"), ("HuggingFaceTB/SmolLM2-135M", "SmolLM2 (RoPE)")]:
    try:
        analyse(name, label)
    except Exception as e:
        print(f"  {label}: FAILED {type(e).__name__}: {str(e)[:110]}")
print()
print("Jaccard near 1 -> one wiring serves all queries. Near 0 -> per-token routing.")
print("If FIXED wiring captures nearly as much mass as ADAPTIVE, Z1 topology suffices.")
