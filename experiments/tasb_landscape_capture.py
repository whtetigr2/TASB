"""
tasb_landscape_capture.py
==============================================================================
Captures the full evolving energy landscape across ALL token generation steps.
One J matrix per token. Assembles into a time-series for Blender animation.

This is NOT a snapshot. This is the river — the complete dynamic process
of the model thinking through a response, rendered frame by frame.

USAGE:
    python tasb_landscape_capture.py --prompt "The capital of France is"
    # Outputs: results/landscape_<timestamp>.npz

    Then in Blender:
    python tasb_blender_river.py --input results/landscape_<timestamp>.npz

OUTPUT FORMAT (.npz):
    J_sequence:     (T, n_heads, S_t, S_t) — energy landscape at each step t
                    NOTE: S grows by 1 each step as sequence expands
                    Stored as list of arrays since shapes differ
    P_sequence:     (T, n_heads, S_t, S_t) — Boltzmann distribution
    tokens:         (T,) string array — token generated at each step
    prob_gap:       (T, n_heads, S_t) — well depth per head per position
    entropy:        (T, n_heads, S_t) — attention entropy
    max_well_depth: (T, n_heads) — deepest well per head per step
    top1_key:       (T, n_heads) — most attended key at final query position
    layer_idx:      int
    head_dim:       int
    T_struct:       float — Boltzmann temperature = sqrt(head_dim)
    prompt:         str
==============================================================================
"""

import argparse
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime


def capture_landscape(
    model, tok, prompt: str,
    layer_idx: int = 18,
    max_tokens: int = 50,
    alpha: float = 0.3,
    K: int = 50,
    seed: int = 42,
    use_bridge: bool = True,
) -> dict:
    """
    Run generation and capture the full energy landscape at every token step.
    Returns a dict ready for .npz serialization.
    """
    from tasb_capture_v2 import LlamaAttentionCapture
    from tasb_sampler_v2 import sample as tasb_sample, SamplerConfig
    from tasb_injector_v2 import LlamaAttentionInjector, DispatchEntry

    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors='pt').to(device)
    current_ids = inputs['input_ids'].clone()

    # Storage — list of arrays (shapes differ as S grows)
    J_frames        = []   # energy landscape per frame
    P_frames        = []   # Boltzmann distribution per frame
    pgap_frames     = []   # well depth per frame
    entropy_frames  = []   # attention entropy per frame
    max_well_frames = []   # max well depth per head per frame
    top1_frames     = []   # top-1 attended key at last query pos per frame
    tokens          = []   # token string generated at each step
    token_ids       = []   # token id

    print(f"\n  Capturing energy landscape for: '{prompt}'")
    print(f"  Layer: {layer_idx}  Alpha: {alpha}  Bridge: {use_bridge}")
    print(f"  Generating tokens...")

    with torch.no_grad():
        for step in range(max_tokens):
            seq_len = current_ids.shape[1]

            # ── Vanilla forward pass with capture ────────────────────
            capturer = LlamaAttentionCapture(
                model=model,
                layers_to_capture=[layer_idx],
                strict_verify=False,
            )
            with capturer.capture():
                van_out = model(input_ids=current_ids, use_cache=False)

            van_logits = van_out.logits[:, -1, :]
            cap = capturer.get_capture(layer_idx)

            if cap is None:
                print(f"  WARNING: No capture at step {step}")
                break

            # ── Extract energy landscape ──────────────────────────────
            q     = cap.q_post_rope.float()
            k     = cap.k_post_rope.float()
            scale = float(cap.scaling)
            mask  = cap.attention_mask

            B, n_q, S, head_dim = q.shape
            n_kv  = k.shape[1]
            n_kvg = n_q // n_kv
            k_exp = k.repeat_interleave(n_kvg, dim=1) if n_kvg > 1 else k

            # J matrix — the physical energy landscape
            J = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
            if mask is not None:
                J = J + mask.float()

            # Boltzmann distribution
            P = F.softmax(J, dim=-1)

            # Statistics
            top2     = P.topk(min(2, S), dim=-1).values
            pgap     = (top2[..., 0] - (top2[..., 1]
                        if top2.shape[-1] > 1
                        else torch.zeros_like(top2[..., 0])))
            entropy  = -(P * torch.log(P + 1e-10)).sum(dim=-1)
            max_well = pgap.squeeze(0).max(dim=-1).values  # (n_q,)
            top1_last = P.squeeze(0)[:, -1, :].argmax(dim=-1)  # (n_q,) at last query pos

            # Store as numpy — drop batch dim
            J_np   = J.squeeze(0).cpu().float().numpy()    # (n_q, S, S)
            P_np   = P.squeeze(0).cpu().float().numpy()    # (n_q, S, S)
            pg_np  = pgap.squeeze(0).cpu().float().numpy() # (n_q, S)
            ent_np = entropy.squeeze(0).cpu().float().numpy()
            mw_np  = max_well.cpu().float().numpy()        # (n_q,)
            t1_np  = top1_last.cpu().numpy()               # (n_q,)

            J_frames.append(J_np)
            P_frames.append(P_np)
            pgap_frames.append(pg_np)
            entropy_frames.append(ent_np)
            max_well_frames.append(mw_np)
            top1_frames.append(t1_np)

            # ── Bridge injection (optional) ───────────────────────────
            if use_bridge:
                scfg     = SamplerConfig(backend='exact', K=K,
                                         seed=seed + step)
                p_thermo = tasb_sample(cap, scfg)
                dispatch = {layer_idx: DispatchEntry(cap, p_thermo, alpha)}
                injector = LlamaAttentionInjector(dispatch)
                with injector.inject():
                    bri_out    = model(input_ids=current_ids,
                                       use_cache=False)
                next_logits = bri_out.logits[:, -1, :]
            else:
                next_logits = van_logits

            # ── Sample next token ─────────────────────────────────────
            probs = F.softmax(next_logits.squeeze(0) / 0.8, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum  = torch.cumsum(sorted_p, dim=0)
            mask2 = cum - sorted_p > 0.9
            sorted_p[mask2] = 0.0
            sorted_p = sorted_p / sorted_p.sum()
            next_tok = sorted_i[torch.multinomial(sorted_p, 1)]

            tok_str = tok.decode([next_tok.item()],
                                  skip_special_tokens=True)
            tokens.append(tok_str)
            token_ids.append(next_tok.item())

            print(f"  Step {step+1:3d}: '{tok_str:15s}' "
                  f"S={S:3d} "
                  f"max_well={mw_np.max():.3f} "
                  f"entropy={ent_np[:, -1].mean():.3f}")

            if next_tok.item() == tok.eos_token_id:
                print(f"  EOS at step {step+1}")
                break

            current_ids = torch.cat(
                [current_ids, next_tok.view(1, 1)], dim=1)

            # Stop if we see end of turn
            gen_so_far = tok.decode(token_ids, skip_special_tokens=True)
            if len(gen_so_far) > 200:
                print(f"  Max length reached at step {step+1}")
                break

    print(f"\n  Captured {len(tokens)} frames")
    print(f"  Final sequence length: {J_frames[-1].shape[-1] if J_frames else 0}")

    return {
        'J_frames':       J_frames,         # list of (n_q, S_t, S_t)
        'P_frames':       P_frames,
        'pgap_frames':    pgap_frames,
        'entropy_frames': entropy_frames,
        'max_well_frames':max_well_frames,
        'top1_frames':    top1_frames,
        'tokens':         tokens,
        'token_ids':      token_ids,
        'prompt':         prompt,
        'layer_idx':      layer_idx,
        'head_dim':       head_dim,
        'n_heads':        n_q,
        'T_struct':       float(head_dim ** 0.5),
        'alpha':          alpha,
        'use_bridge':     use_bridge,
    }


def save_landscape(data: dict, path: str) -> str:
    """
    Save landscape data to .npz.
    Since J frames have different shapes (S grows each step),
    we store them as object arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = len(data['tokens'])

    # Store variable-shape arrays as object arrays
    J_obj   = np.empty(T, dtype=object)
    P_obj   = np.empty(T, dtype=object)
    pg_obj  = np.empty(T, dtype=object)
    ent_obj = np.empty(T, dtype=object)

    for i in range(T):
        J_obj[i]   = data['J_frames'][i]
        P_obj[i]   = data['P_frames'][i]
        pg_obj[i]  = data['pgap_frames'][i]
        ent_obj[i] = data['entropy_frames'][i]

    # Fixed-shape arrays
    max_wells = np.stack(data['max_well_frames'])  # (T, n_heads)
    top1s     = np.stack(data['top1_frames'])      # (T, n_heads)

    np.savez_compressed(
        str(path),
        J=J_obj,
        P=P_obj,
        prob_gap=pg_obj,
        entropy=ent_obj,
        max_well=max_wells,
        top1_key=top1s,
        tokens=np.array(data['tokens']),
        token_ids=np.array(data['token_ids']),
        prompt=np.array(data['prompt']),
        layer_idx=np.array(data['layer_idx']),
        n_heads=np.array(data['n_heads']),
        head_dim=np.array(data['head_dim']),
        T_struct=np.array(data['T_struct']),
        alpha=np.array(data['alpha']),
        use_bridge=np.array(data['use_bridge']),
        n_frames=np.array(T),
    )

    size_mb = path.stat().st_size / 1e6
    print(f"\n  Saved: {path}  ({size_mb:.1f} MB)")
    print(f"  Frames: {T}")
    print(f"  Heads:  {data['n_heads']}")
    print(f"  T_struct (Boltzmann temp): {data['T_struct']:.2f}")

    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Capture TASB energy landscape time series")
    parser.add_argument('--prompt',
        default="The capital of France is Paris.",
        help="Prompt to run")
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--max-tokens', type=int, default=30)
    parser.add_argument('--no-bridge', action='store_true',
        help="Capture vanilla (no bridge injection)")
    parser.add_argument('--output', default=None,
        help="Output .npz path (default: results/landscape_<timestamp>.npz)")
    parser.add_argument('--model',
        default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    print("Model ready.")

    # Capture
    data = capture_landscape(
        model=model,
        tok=tok,
        prompt=args.prompt,
        layer_idx=args.layer,
        max_tokens=args.max_tokens,
        alpha=args.alpha,
        use_bridge=not args.no_bridge,
    )

    # Save
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or f"results/landscape_{ts}.npz"
    save_landscape(data, out)

    print("\nDone. Feed this file to tasb_blender_river.py to render.")


if __name__ == "__main__":
    main()
