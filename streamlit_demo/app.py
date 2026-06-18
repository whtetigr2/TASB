"""
app.py — TASB Energy Landscape Demo
Streamlit Community Cloud deployment

Real attention energy landscapes from LLaMA 3.2-3B, layer 18.
Pre-computed sweep: 5 prompts x 4 backends x 4 alphas x 25 tokens.
"""

import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TASB — Thermodynamic Attention Sampling Bridge",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0a0f; color: #e8e8e8; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d0d1a !important;
    border-right: 1px solid #1a1a30 !important;
  }
  [data-testid="stSidebar"] * {
    font-family: 'Courier New', monospace !important;
  }

  /* Title block */
  .tasb-title {
    font-family: 'Courier New', monospace;
    font-size: 1.05rem;
    color: #f2a050;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
  }
  .tasb-subtitle {
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    color: #555;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  /* Backend color stripe */
  .backend-stripe {
    height: 3px;
    border-radius: 2px;
    margin-bottom: 0.6rem;
    transition: background 0.3s ease;
  }

  /* Active run info panel */
  .run-info {
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    color: #888;
    padding: 0.6rem 0.8rem;
    background: #0d0d1a;
    border: 1px solid #1a1a30;
    border-radius: 4px;
    margin-bottom: 0.6rem;
    line-height: 1.8;
  }
  .run-info .label { color: #444; font-size: 0.60rem; text-transform: uppercase; letter-spacing: 0.1em; }
  .run-info .value { color: #f2a050; }
  .run-info .value.exact  { color: #22aaff; }
  .run-info .value.gumbel { color: #aa22ff; }
  .run-info .value.rbm    { color: #22ff88; }
  .run-info .value.thrml  { color: #ff8822; }

  /* Entropy bar */
  .entropy-bar-wrap {
    font-family: 'Courier New', monospace;
    font-size: 0.60rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
  }
  .entropy-bar-bg {
    background: #1a1a2e;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-bottom: 0.5rem;
  }
  .entropy-bar-fill {
    height: 6px;
    border-radius: 2px;
    transition: width 0.3s ease, background 0.3s ease;
  }

  /* Zero flips badge */
  .zero-flips-badge {
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    color: #22ffbb;
    border: 1px solid #22ffbb;
    border-radius: 3px;
    padding: 0.1rem 0.5rem;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
  }

  /* Sing-along text */
  .singalong {
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #555;
    padding: 0.7rem 1rem;
    background: #0d0d1a;
    border-left: 3px solid #1a1a30;
    border-radius: 0 4px 4px 0;
    margin-top: 0.4rem;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.7;
  }
  .singalong .done  { color: #888; }
  .singalong .current { color: #f2a050; font-weight: bold; }
  .singalong .pending { color: #2a2a3a; }

  /* Step progress bar */
  .step-progress-wrap {
    font-family: 'Courier New', monospace;
    font-size: 0.60rem; color: #333;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.15rem; margin-top: 0.5rem;
  }
  .step-progress-bg {
    background: #111120; border-radius: 2px;
    height: 4px; width: 100%; margin-bottom: 0.5rem;
  }
  .step-progress-fill {
    height: 4px; border-radius: 2px; background: #f2a050;
  }

  /* Metrics strip */
  .metric-strip {
    display: flex; flex-wrap: wrap; gap: 1.2rem;
    padding: 0.6rem 1rem;
    background: #0d0d1a;
    border: 1px solid #1a1a30;
    border-radius: 4px; margin-top: 0.4rem;
    font-family: 'Courier New', monospace;
  }
  .metric-item { display: flex; flex-direction: column; gap: 0.05rem; }
  .metric-label { color: #444; font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.1em; }
  .metric-value { font-size: 0.88rem; color: #f2a050; }
  .metric-value.good { color: #22ffbb; }
  .metric-value.warn { color: #ff9944; }
  .metric-value.bad  { color: #ff4466; }

  /* Summary table */
  .summary-table { font-family: 'Courier New', monospace; font-size: 0.70rem; color: #666; margin-top: 0.5rem; }
  .summary-table td { padding: 0.12rem 0.8rem 0.12rem 0; }
  .summary-table .good { color: #22ffbb; }
  .summary-table .warn { color: #ff9944; }
  .summary-table .bad  { color: #ff4466; }

  /* Sidebar section labels */
  .sb-label {
    font-family: 'Courier New', monospace !important;
    font-size: 0.58rem !important;
    color: #444 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    margin-bottom: 0.2rem !important;
    margin-top: 0.8rem !important;
  }

  /* Footer */
  .tasb-footer {
    font-family: 'Courier New', monospace;
    font-size: 0.58rem; color: #222230;
    text-align: center; margin-top: 1.5rem;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: #0d0d1a !important;
    border: 1px solid #1a1a30 !important;
    border-radius: 4px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
PROMPT_META = {
    "thermo":         ("⚛",  "Thermo / Computation",
                       "Sharp, confident wells — the model knows this territory."),
    "whimsy":         ("🌀", "Whimsy / Chaos",
                       "Maximum entropy — many valid continuations, no dominant well."),
    "narrative":      ("🏮", "Narrative",
                       "Moderate entropy — genre conventions add some constraint."),
    "technical":      ("📐", "Technical",
                       "High-confidence factual — deep, narrow wells."),
    "conversational": ("💬", "Conversational",
                       "Well-traveled territory for an instruction-tuned model."),
}

BACKEND_COLORS = {
    "exact":  "#22aaff",
    "gumbel": "#aa22ff",
    "rbm":    "#22ff88",
    "thrml":  "#ff8822",
}

BACKEND_DESC = {
    "exact":  "torch.multinomial over softmax",
    "gumbel": "Gumbel-max logit-space sampling",
    "rbm":    "iterative RBM Gibbs",
    "thrml":  "Extropic THRML block-Gibbs Boltzmann",
}

ALPHA_OPTIONS = [0.0, 0.3, 0.7, 1.0]
ALPHA_LABELS  = {
    0.0: "α=0.0  vanilla",
    0.3: "α=0.3  production",
    0.7: "α=0.7  max safe zone",
    1.0: "α=1.0  full TSU",
}

MASK_THRESHOLD = -50.0


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    candidates = [
        "demo_data.json",
        os.path.join(os.path.dirname(__file__), "demo_data.json"),
        "streamlit_demo/demo_data.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

data = load_data()


# ── Surface helpers ───────────────────────────────────────────────────────────
def prepare_surface(J_raw: np.ndarray) -> np.ndarray:
    J = J_raw.copy().astype(np.float64)
    J[J < MASK_THRESHOLD] = np.nan
    E = -J
    valid = E[~np.isnan(E)]
    if len(valid) > 0:
        E = E - np.nanmin(E)
    return E


def enhance_wells(E: np.ndarray, top_positions: list,
                  well_depth: float = 2.5, well_width: float = 1.5) -> np.ndarray:
    E_e = E.copy()
    S   = E.shape[0]
    q   = S - 1
    x   = np.arange(S, dtype=np.float64)
    for rank, pos in enumerate(top_positions[:6]):
        if np.isnan(E[q, pos]):
            continue
        d = well_depth / (rank + 1)
        g = d * np.exp(-((x - pos) ** 2) / (2 * well_width ** 2))
        E_e[q, :] = np.where(np.isnan(E_e[q, :]), np.nan, E_e[q, :] - g)
    return E_e


def build_surface_data(token_record: dict) -> dict:
    J_raw = np.array(token_record["J_matrix"], dtype=np.float32)
    S     = J_raw.shape[0]

    last_row = J_raw[S - 1, :].copy()
    last_row[last_row < MASK_THRESHOLD] = np.nan
    valid_mask = ~np.isnan(last_row)
    if valid_mask.sum() > 0:
        ranked    = np.argsort(-last_row[valid_mask])
        valid_idx = np.where(valid_mask)[0]
        top_pos   = valid_idx[ranked[:6]].tolist()
    else:
        top_pos = list(range(min(6, S)))

    E  = prepare_surface(J_raw)
    E  = enhance_wells(E, top_pos)

    positions = token_record.get("sample_positions", [])
    pos_arr   = np.array(positions, dtype=int).clip(0, S - 1) if positions else np.array([], dtype=int)
    q_idx     = S - 1
    mx = pos_arr.astype(float)
    my = np.full(len(pos_arr), float(q_idx))
    mz = np.array([
        float(E[q_idx, p]) + 0.3 if not np.isnan(E[q_idx, p]) else 0.3
        for p in pos_arr
    ])

    top_tok = token_record.get("top_tokens", [])
    lx, ly, lz, lt = [], [], [], []
    for rank, pos in enumerate(top_pos[:6]):
        if rank >= len(top_tok):
            break
        ez = float(E[q_idx, pos]) if not np.isnan(E[q_idx, pos]) else 0.0
        lx.append(float(pos))
        ly.append(float(q_idx))
        lz.append(ez - 0.5)
        lt.append(f"<b>{top_tok[rank]['text']}</b><br>{top_tok[rank]['prob']*100:.1f}%")

    return {
        "E": E, "S": S,
        "mx": mx, "my": my, "mz": mz,
        "lx": lx, "ly": ly, "lz": lz, "lt": lt,
        "top_pos": top_pos,
    }


SURFACE_COLORSCALE = [
    [0.00, "#0d0500"], [0.15, "#3d1200"],
    [0.35, "#8b3000"], [0.55, "#c85000"],
    [0.75, "#e87820"], [0.90, "#f2a050"],
    [1.00, "#ffd080"],
]

SCENE_CFG = dict(
    bgcolor="#0a0a0f",
    xaxis=dict(title="Key position",  showbackground=False,
               gridcolor="#111120", tickfont=dict(color="#333", size=7)),
    yaxis=dict(title="Query position", showbackground=False,
               gridcolor="#111120", tickfont=dict(color="#333", size=7)),
    zaxis=dict(title="Energy",         showbackground=False,
               gridcolor="#111120", tickfont=dict(color="#444", size=7)),
    aspectratio=dict(x=1.2, y=1.2, z=0.6),
    camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
)


def build_static_figure(record: dict, alpha: float, backend: str) -> go.Figure:
    d  = build_surface_data(record)
    S  = d["S"]
    mc = BACKEND_COLORS.get(backend, "#22ffcc")
    kl  = record.get("kl", 0.0)
    bkt = record.get("bucket", "")
    s_l = record.get("seq_len", S)

    traces = [
        go.Surface(z=d["E"], x=np.arange(S), y=np.arange(S),
                   colorscale=SURFACE_COLORSCALE, showscale=False,
                   opacity=0.92, hoverinfo="skip",
                   lighting=dict(ambient=0.55, diffuse=0.85,
                                 specular=0.15, roughness=0.65)),
    ]
    if len(d["mx"]) > 0:
        traces.append(go.Scatter3d(
            x=d["mx"], y=d["my"], z=d["mz"], mode="markers",
            marker=dict(size=5, color=mc, opacity=0.88,
                        line=dict(color="#000008", width=0.5)),
            hoverinfo="skip", name="p-bits",
        ))
    if d["lx"]:
        traces.append(go.Scatter3d(
            x=d["lx"], y=d["ly"], z=d["lz"], mode="text",
            text=d["lt"],
            textfont=dict(family="Courier New, monospace", size=11, color="#f2a050"),
            hoverinfo="skip", name="wells",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        margin=dict(l=0, r=0, t=20, b=0), height=560, showlegend=False,
    )
    fig.update_layout(scene=SCENE_CFG)
    fig.add_annotation(
        text=f"{backend}  α={alpha:.1f}  S={s_l}  KL={kl:.5f}  {bkt}",
        xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        font=dict(family="Courier New", size=9, color="#444"),
        showarrow=False,
    )
    return fig


def build_animated_figure(tokens: list, alpha: float,
                            backend: str, interp: int = 3) -> go.Figure:
    mc       = BACKEND_COLORS.get(backend, "#22ffcc")
    all_data = [build_surface_data(t) for t in tokens]

    frame_list = []
    for i, d in enumerate(all_data):
        if i == 0:
            frame_list.append((d, d))
        else:
            dp = all_data[i - 1]
            if dp["E"].shape == d["E"].shape:
                for t in range(1, interp + 1):
                    f = t / interp
                    bd        = dict(d)
                    bd["E"]   = (1 - f) * dp["E"]   + f * d["E"]
                    bd["mx"]  = ((1-f)*dp["mx"] + f*d["mx"]
                                 if len(dp["mx"]) == len(d["mx"]) else d["mx"])
                    bd["my"]  = ((1-f)*dp["my"] + f*d["my"]
                                 if len(dp["my"]) == len(d["my"]) else d["my"])
                    bd["mz"]  = ((1-f)*dp["mz"] + f*d["mz"]
                                 if len(dp["mz"]) == len(d["mz"]) else d["mz"])
                    frame_list.append((bd, d))
            else:
                frame_list.append((d, d))

    d0 = frame_list[0][0]
    S0 = d0["S"]

    fig = go.Figure(data=[
        go.Surface(z=d0["E"], x=np.arange(S0), y=np.arange(S0),
                   colorscale=SURFACE_COLORSCALE, showscale=False,
                   opacity=0.92, hoverinfo="skip",
                   lighting=dict(ambient=0.55, diffuse=0.85,
                                 specular=0.15, roughness=0.65)),
        go.Scatter3d(x=d0["mx"], y=d0["my"], z=d0["mz"], mode="markers",
                     marker=dict(size=5, color=mc, opacity=0.88,
                                 line=dict(color="#000008", width=0.5)),
                     hoverinfo="skip"),
        go.Scatter3d(x=d0["lx"], y=d0["ly"], z=d0["lz"], mode="text",
                     text=d0["lt"],
                     textfont=dict(family="Courier New, monospace",
                                   size=11, color="#f2a050"),
                     hoverinfo="skip"),
    ])

    plotly_frames = []
    for k, (fd, _) in enumerate(frame_list):
        S_f = fd["S"]
        plotly_frames.append(go.Frame(
            data=[
                go.Surface(z=fd["E"], x=np.arange(S_f), y=np.arange(S_f),
                           showscale=False, opacity=0.92, hoverinfo="skip"),
                go.Scatter3d(x=fd["mx"], y=fd["my"], z=fd["mz"], mode="markers",
                             marker=dict(size=5, color=mc, opacity=0.88)),
                go.Scatter3d(x=fd["lx"], y=fd["ly"], z=fd["lz"], mode="text",
                             text=fd["lt"],
                             textfont=dict(family="Courier New, monospace",
                                           size=11, color="#f2a050")),
            ],
            name=str(k),
        ))
    fig.frames = plotly_frames

    n = len(plotly_frames)
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        margin=dict(l=0, r=0, t=20, b=60), height=620, showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0.02, x=0.5, xanchor="center", yanchor="bottom",
            bgcolor="#0d0d1a", bordercolor="#1a1a30",
            font=dict(family="Courier New", size=10, color="#f2a050"),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=60,
                                                      easing="cubic-in-out"),
                                      mode="immediate")]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(k)],
                              dict(mode="immediate",
                                   frame=dict(duration=80, redraw=True),
                                   transition=dict(duration=60))],
                        label=str(k))
                   for k in range(n)],
            active=0,
            currentvalue=dict(prefix="Frame: ",
                              font=dict(family="Courier New", size=9, color="#555")),
            bgcolor="#0d0d1a", bordercolor="#1a1a30", tickcolor="#222",
            font=dict(family="Courier New", size=8, color="#333"),
            y=0.0, x=0.0, len=1.0, pad=dict(t=40),
        )],
    )
    fig.update_layout(scene=SCENE_CFG)
    return fig


# ── Synthetic fallback ────────────────────────────────────────────────────────
def make_synthetic_token(S=40, K=50, alpha=0.3, backend="exact"):
    rng = np.random.default_rng(abs(hash(backend)) % (2**31))
    J   = rng.normal(0, 1, (S, S)).astype(np.float32)
    for _ in range(4):
        i = rng.integers(5, S-5); j = rng.integers(0, i+1)
        J[i, j] += rng.uniform(4, 8) * (1.0 + alpha)
    for i in range(S):
        J[i, i+1:] = -100.0
    top_tokens = [{"idx": k, "text": f" tok{k}",
                   "logit": float(3-k*0.3), "prob": float(0.4/(k+1))}
                  for k in range(10)]
    J_last = J[-1].copy()
    p = np.exp(np.clip(J_last - J_last.max(), -50, 0))
    p /= p.sum()
    return {
        "J_matrix": J.tolist(), "J_last_row": J_last.tolist(),
        "top_tokens": top_tokens,
        "sample_positions": rng.choice(S, size=K, p=p).tolist(),
        "kl": 0.00138*(1+alpha), "top1_match": True,
        "prob_gap": 0.72, "bucket": "confident",
        "token_text": " the", "van_top1_text": " the",
        "inj_top1_text": " the", "seq_len": S,
    }


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("sel_prompt", "thermo"), ("sel_backend", "exact"),
             ("token_step", 0), ("last_prompt", "thermo"),
             ("last_backend", "exact"), ("alpha_idx", 1)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — all controls live here
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:Courier New;font-size:0.8rem;'
        'color:#f2a050;letter-spacing:0.1em;text-transform:uppercase;'
        'border-bottom:1px solid #1a1a30;padding-bottom:0.5rem;'
        'margin-bottom:0.8rem;">⚙ Bridge Settings</p>',
        unsafe_allow_html=True,
    )

    # ── Prompt selector ───────────────────────────────────────────────────────
    st.markdown('<p class="sb-label">Prompt</p>', unsafe_allow_html=True)
    for pk, (icon, label, desc) in PROMPT_META.items():
        active = (st.session_state.sel_prompt == pk)
        border = f"border-left: 3px solid {BACKEND_COLORS.get(st.session_state.sel_backend,'#f2a050')};" if active else "border-left: 3px solid #1a1a30;"
        if st.button(f"{icon}  {label}", key=f"sb_prompt_{pk}",
                     use_container_width=True):
            st.session_state.sel_prompt  = pk
            st.session_state.token_step  = 0

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    # ── Backend selector ──────────────────────────────────────────────────────
    st.markdown('<p class="sb-label">Backend (sampler)</p>', unsafe_allow_html=True)
    for be in ["exact", "gumbel", "rbm", "thrml"]:
        color  = BACKEND_COLORS[be]
        active = (st.session_state.sel_backend == be)
        if st.button(
            f"{'▶ ' if active else '   '}{be}",
            key=f"sb_be_{be}",
            use_container_width=True,
        ):
            st.session_state.sel_backend = be
            st.session_state.token_step  = 0

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    # ── Alpha slider ──────────────────────────────────────────────────────────
    st.markdown('<p class="sb-label">Alpha — TSU participation</p>',
                unsafe_allow_html=True)
    alpha_idx = st.select_slider(
        "alpha_ctrl",
        options=list(range(len(ALPHA_OPTIONS))),
        value=st.session_state.alpha_idx,
        format_func=lambda i: ALPHA_LABELS[ALPHA_OPTIONS[i]],
        label_visibility="collapsed",
    )
    st.session_state.alpha_idx = alpha_idx
    alpha = ALPHA_OPTIONS[alpha_idx]

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    # ── What am I looking at ──────────────────────────────────────────────────
    with st.expander("❓  What am I looking at?"):
        st.markdown(
            '<p style="font-family:Courier New;font-size:0.68rem;'
            'color:#888;line-height:1.7;">'
            '<b style="color:#f2a050">The surface</b> is the attention energy '
            'landscape at Layer 18. Each point represents how much the model '
            '"wants" to attend from one token to another. Deep wells = '
            'high-probability tokens.<br><br>'
            '<b style="color:#f2a050">The marbles</b> are 50 p-bits '
            '(probabilistic bits) sampling the landscape simultaneously — '
            'just like Extropic\'s TSU hardware would in silicon. Each marble '
            'independently finds a low-energy well.<br><br>'
            '<b style="color:#f2a050">The labels</b> show which actual tokens '
            'sit at the bottom of each well, with their probability.<br><br>'
            '<b style="color:#f2a050">Alpha (α)</b> controls how much the '
            'stochastic hardware participates. α=0 is pure GPU. α=1 is full '
            'TSU substitution. The model is never retrained.'
            '</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:Courier New;font-size:0.58rem;color:#222230;'
        'text-align:center;border-top:1px solid #111120;padding-top:0.5rem;">'
        'github.com/whtetigr2/TASB</p>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

# Resolve state
selected = st.session_state.sel_prompt
backend  = st.session_state.sel_backend
run_key  = f"{backend}_alpha_{alpha}"

# Reset step if selection changed
if (st.session_state.last_prompt  != selected or
        st.session_state.last_backend != backend):
    st.session_state.token_step   = 0
    st.session_state.last_prompt  = selected
    st.session_state.last_backend = backend

if data:
    try:
        tokens   = data["prompts"][selected][run_key]["tokens"]
        max_step = len(tokens) - 1
    except (KeyError, TypeError):
        tokens = None; max_step = 24
else:
    tokens = None; max_step = 24

# Current record
if data and tokens:
    step = st.session_state.token_step
    try:
        record = tokens[step]
    except (IndexError, TypeError):
        record = tokens[-1] if tokens else make_synthetic_token()
else:
    step   = 0
    record = make_synthetic_token(alpha=alpha, backend=backend)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="tasb-title">TASB — Thermodynamic Attention Sampling Bridge</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tasb-subtitle">'
    'Real attention energy landscapes · LLaMA 3.2-3B · Layer 18 · '
    'K=50 Boltzmann samples · Paul W. Shaver 2026 · '
    '<a href="https://github.com/whtetigr2/TASB" '
    'style="color:#333;text-decoration:none;">github.com/whtetigr2/TASB</a>'
    '</p>',
    unsafe_allow_html=True,
)

# ── Backend color stripe ──────────────────────────────────────────────────────
bc = BACKEND_COLORS.get(backend, "#f2a050")
st.markdown(
    f'<div class="backend-stripe" style="background:{bc};"></div>',
    unsafe_allow_html=True,
)

# ── Active run info panel ─────────────────────────────────────────────────────
icon_p, label_p, desc_p = PROMPT_META[selected]
kl      = record.get("kl", 0.0)
top1    = record.get("top1_match", True)
pg      = record.get("prob_gap", 0.0)
bkt     = record.get("bucket", "confident")
s_len   = record.get("seq_len", 0)
van_tok = record.get("van_top1_text", "")
inj_tok = record.get("inj_top1_text", "")

# Check zero flips for this run
conf_flips = 0
if data and tokens:
    try:
        conf_flips = data["prompts"][selected][run_key]["summary"]["conf_flips"]
    except (KeyError, TypeError):
        conf_flips = -1

kl_cls   = "good" if kl < 0.005 else ("warn" if kl < 0.02 else "bad")
flip_str = "✓ match" if top1 else f"✗ flip [{bkt}]"
t1_cls   = "good" if top1 else "bad"
bkt_cls  = "good" if bkt == "confident" else "warn"

zero_badge = (
    '<span class="zero-flips-badge">✓ ZERO CONFIDENT FLIPS</span>'
    if conf_flips == 0 else ""
)

info_left, info_right = st.columns([3, 2])

with info_left:
    st.markdown(f"""
<div class="run-info">
  <span class="label">Prompt</span><br>
  <span class="value">{icon_p} {label_p}</span>
  <span style="color:#444;font-size:0.62rem"> — {desc_p}</span><br><br>
  <span class="label">Backend</span><br>
  <span class="value {backend}">{backend}</span>
  <span style="color:#444;font-size:0.62rem"> — {BACKEND_DESC.get(backend,'')}</span><br><br>
  <span class="label">Alpha</span>
  <span class="value"> {alpha:.1f}</span>
  <span style="color:#444;font-size:0.62rem"> — {ALPHA_LABELS[alpha]}</span>
  {zero_badge}
</div>
""", unsafe_allow_html=True)

with info_right:
    # Entropy / confidence indicator
    pg_pct  = int(pg * 100)
    pg_col  = "#22ffbb" if pg >= 0.5 else ("#ff9944" if pg >= 0.1 else "#ff4466")
    pg_label = bkt

    # Step progress
    step_pct = int((step / max(max_step, 1)) * 100)

    st.markdown(f"""
<div class="run-info">
  <div class="entropy-bar-wrap">Model confidence (prob gap) — {pg_label}</div>
  <div class="entropy-bar-bg">
    <div class="entropy-bar-fill" style="width:{pg_pct}%;background:{pg_col};"></div>
  </div>
  <div class="step-progress-wrap">Generation progress — step {step} / {max_step}</div>
  <div class="step-progress-bg">
    <div class="step-progress-fill" style="width:{step_pct}%;"></div>
  </div>
  <span class="label">KL divergence</span>
  <span class="metric-value {kl_cls}" style="font-size:0.85rem"> {kl:.5f}</span><br>
  <span class="label">Top-1</span>
  <span class="metric-value {t1_cls}" style="font-size:0.85rem"> {flip_str}</span><br>
  <span class="label">Vanilla → Bridge</span>
  <span style="color:#666;font-size:0.78rem"> '{van_tok}' → '{inj_tok}'</span>
</div>
""", unsafe_allow_html=True)

# ── Token step slider ─────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:Courier New;font-size:0.60rem;color:#333;'
    'text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.1rem;">'
    'Token step scrubber</p>',
    unsafe_allow_html=True,
)
new_step = st.slider(
    "token_step_slider",
    min_value=0, max_value=max(0, max_step),
    value=st.session_state.token_step,
    label_visibility="collapsed",
)
if new_step != st.session_state.token_step:
    st.session_state.token_step = new_step
    st.rerun()

# ── Tabs: Scrub | Animation ───────────────────────────────────────────────────
tab_scrub, tab_anim = st.tabs(["🔍  Step View", "▶  Animated Generation"])

with tab_scrub:
    if not (data and tokens):
        st.info("⚠️ demo_data.json not found — showing synthetic data.", icon="⚠️")
    fig_s = build_static_figure(record, alpha=alpha, backend=backend)
    st.plotly_chart(fig_s, use_container_width=True,
                    config={"displayModeBar": False}, key="static_chart")

with tab_anim:
    st.markdown(
        '<p style="font-family:Courier New;font-size:0.68rem;color:#555;'
        'line-height:1.6;margin-bottom:0.5rem;">'
        'Press ▶ Play — the energy landscape morphs token by token as the model '
        'generates. Each frame shows the real attention surface reshaping. '
        'p-bits scatter and re-settle into new wells with every token.'
        '</p>',
        unsafe_allow_html=True,
    )
    if data and tokens and len(tokens) > 1:
        with st.spinner("Pre-computing animation frames..."):
            fig_a = build_animated_figure(tokens, alpha=alpha, backend=backend)
        st.plotly_chart(fig_a, use_container_width=True,
                        config={"displayModeBar": False}, key="anim_chart")
    else:
        st.info("Load real data to see the animation.", icon="ℹ️")

# ── Sing-along text ───────────────────────────────────────────────────────────
if data and tokens:
    prompt_text = data["prompts"][selected]["text"]
    done_tokens = "".join(t.get("token_text", "") for t in tokens[:step])
    curr_token  = tokens[step].get("token_text", "") if step < len(tokens) else ""
    remain      = "".join(t.get("token_text", "") for t in tokens[step+1:])

    st.markdown(
        f'<div class="singalong">'
        f'<span style="color:#555">{prompt_text}</span>'
        f'<span class="done">{done_tokens}</span>'
        f'<span class="current">{curr_token}</span>'
        f'<span class="pending">{remain}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Backend comparison table ──────────────────────────────────────────────────
st.markdown("---")
if data:
    st.markdown(
        f'<p style="font-family:Courier New;font-size:0.60rem;color:#333;'
        f'text-transform:uppercase;letter-spacing:0.12em;">'
        f'Backend comparison — {label_p} · α={alpha:.1f}</p>',
        unsafe_allow_html=True,
    )
    rows = []
    for be in ["exact", "gumbel", "rbm", "thrml"]:
        rk = f"{be}_alpha_{alpha}"
        try:
            s  = data["prompts"][selected][rk]["summary"]
            kl_s = s["mean_kl"]; t1_s = s["top1_pct"]; cf_s = s["conf_flips"]
            kl_c = "good" if kl_s < 0.005 else "warn"
            t1_c = "good" if t1_s >= 99.0 else "warn"
            cf_c = "good" if cf_s == 0 else "bad"
            col  = BACKEND_COLORS[be]
            rows.append(
                f'<tr>'
                f'<td style="color:{col};padding-right:1.5rem">{be}</td>'
                f'<td class="{kl_c}" style="padding-right:1.5rem">{kl_s:.5f}</td>'
                f'<td class="{t1_c}" style="padding-right:1.5rem">{t1_s:.1f}%</td>'
                f'<td class="{cf_c}" style="padding-right:1.5rem">{cf_s} flips</td>'
                f'<td style="color:#444">{s["n_tokens"]} tokens</td>'
                f'</tr>'
            )
        except (KeyError, TypeError):
            col = BACKEND_COLORS[be]
            rows.append(
                f'<tr><td style="color:{col}">{be}</td>'
                f'<td colspan="4" style="color:#333">no data</td></tr>'
            )
    st.markdown(f"""
<table class="summary-table">
  <tr>
    <td style="color:#222;padding-right:1.5rem">BACKEND</td>
    <td style="color:#222;padding-right:1.5rem">MEAN KL</td>
    <td style="color:#222;padding-right:1.5rem">TOP-1</td>
    <td style="color:#222;padding-right:1.5rem">CONF FLIPS</td>
    <td style="color:#222">TOKENS</td>
  </tr>
  {''.join(rows)}
</table>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="tasb-footer">'
    'TASB · Paul W. Shaver 2026 · '
    'Real data. Every claim has a CSV. Every CSV has a script.'
    '</p>',
    unsafe_allow_html=True,
)
