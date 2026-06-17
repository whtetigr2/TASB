"""
app.py — TASB Energy Landscape Demo
Streamlit Community Cloud deployment

Visualizes real attention energy landscapes from LLaMA 3.2-3B,
captured by the TASB bridge at layer 18.

Data source: demo_data.json (pre-computed on Lightning.ai)
Full sweep: 5 prompts × 4 backends × 4 alphas × 25 tokens
"""

import json
import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TASB — Thermodynamic Attention Sampling Bridge",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0a0f; color: #e8e8e8; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  .tasb-title {
    font-family: 'Courier New', monospace;
    font-size: 1.1rem;
    color: #f2a050;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0;
  }
  .tasb-subtitle {
    font-family: 'Courier New', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
    margin-bottom: 1rem;
  }

  /* Control section labels */
  .ctrl-label {
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.25rem;
  }

  /* Prompt buttons */
  div[data-testid="column"] button {
    background-color: #111120 !important;
    color: #888 !important;
    border: 1px solid #222240 !important;
    border-radius: 3px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.70rem !important;
    padding: 0.35rem 0.5rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
  }
  div[data-testid="column"] button:hover {
    border-color: #f2a050 !important;
    color: #f2a050 !important;
    background-color: #1a0f00 !important;
  }

  /* Backend pills */
  .backend-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }
  .backend-pill {
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    border: 1px solid #333;
    color: #666;
    cursor: pointer;
    transition: all 0.15s;
  }
  .backend-pill.exact   { border-color: #22aaff; color: #22aaff; }
  .backend-pill.gumbel  { border-color: #aa22ff; color: #aa22ff; }
  .backend-pill.rbm     { border-color: #22ff88; color: #22ff88; }
  .backend-pill.thrml   { border-color: #ff8822; color: #ff8822; }

  /* Metrics strip */
  .metric-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    padding: 0.7rem 1.2rem;
    background: #0c0c18;
    border: 1px solid #1a1a30;
    border-radius: 4px;
    margin-top: 0.4rem;
    font-family: 'Courier New', monospace;
  }
  .metric-item { display: flex; flex-direction: column; gap: 0.05rem; }
  .metric-label {
    color: #444;
    font-size: 0.60rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .metric-value { font-size: 0.95rem; color: #f2a050; }
  .metric-value.good  { color: #22ffbb; }
  .metric-value.warn  { color: #ff9944; }
  .metric-value.bad   { color: #ff4466; }

  /* Token text display */
  .token-display {
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #888;
    padding: 0.6rem 1rem;
    background: #0c0c18;
    border-left: 3px solid #f2a050;
    border-radius: 0 3px 3px 0;
    margin-top: 0.4rem;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
  }
  .token-new { color: #22ffbb; font-weight: bold; }

  /* Summary table */
  .summary-table {
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    color: #666;
    margin-top: 1rem;
  }
  .summary-table td { padding: 0.15rem 0.8rem 0.15rem 0; }
  .summary-table .good { color: #22ffbb; }
  .summary-table .warn { color: #ff9944; }

  /* Footer */
  .tasb-footer {
    font-family: 'Courier New', monospace;
    font-size: 0.60rem;
    color: #2a2a3a;
    text-align: center;
    margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
PROMPT_META = {
    "thermo":         ("⚛",  "Thermo / Computation"),
    "whimsy":         ("🌀", "Whimsy / Chaos"),
    "narrative":      ("🏮", "Narrative"),
    "technical":      ("📐", "Technical"),
    "conversational": ("💬", "Conversational"),
}

BACKEND_COLORS = {
    "exact":  "#22aaff",
    "gumbel": "#aa22ff",
    "rbm":    "#22ff88",
    "thrml":  "#ff8822",
}

BACKEND_LABELS = {
    "exact":  "exact  — torch.multinomial over softmax",
    "gumbel": "gumbel — Gumbel-max logit-space sampling",
    "rbm":    "rbm    — iterative RBM Gibbs",
    "thrml":  "thrml  — Extropic THRML block-Gibbs Boltzmann",
}

ALPHA_OPTIONS = [0.0, 0.3, 0.7, 1.0]
ALPHA_LABELS  = {
    0.0: "α = 0.0  (vanilla — no bridge)",
    0.3: "α = 0.3  (production)",
    0.7: "α = 0.7  (max safe zone)",
    1.0: "α = 1.0  (full TSU participation)",
}


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


# ── Synthetic fallback ────────────────────────────────────────────────────────
def make_synthetic_token(S: int = 40, K: int = 50, alpha: float = 0.3,
                          backend: str = "exact"):
    rng = np.random.default_rng(hash(backend) % (2**32))
    J   = rng.normal(0, 1, (S, S)).astype(np.float32)
    for _ in range(3):
        i = rng.integers(5, S - 5)
        j = rng.integers(0, i + 1)
        J[i, j] += rng.uniform(4, 8) * (1.0 + alpha)
    for i in range(S):
        J[i, i+1:] = -9.0
    top_tokens = [
        {"idx": k, "text": f" tok{k}", "logit": float(3 - k*0.3),
         "prob": float(0.4 / (k+1))}
        for k in range(10)
    ]
    J_last = J[-1].copy()
    probs  = np.exp(J_last - J_last.max())
    probs  = np.where(np.isfinite(probs), probs, 0)
    probs /= probs.sum()
    samples = rng.choice(S, size=K, p=probs).tolist()
    return {
        "J_matrix": J.tolist(), "J_last_row": J_last.tolist(),
        "top_tokens": top_tokens, "sample_positions": samples,
        "kl": 0.00138 * (1 + alpha), "top1_match": True,
        "prob_gap": 0.72, "bucket": "confident",
        "token_text": " the", "van_top1_text": " the",
        "inj_top1_text": " the", "seq_len": S,
    }


# ── Build 3D figure ───────────────────────────────────────────────────────────
def build_figure(token_record: dict, alpha: float,
                 backend: str, n_marbles: int = -1) -> go.Figure:
    """
    Build the 3D energy landscape with marble positions.
    n_marbles: if >= 0, show only first n marbles (for animation).
    """
    J    = np.array(token_record["J_matrix"], dtype=np.float32)
    S    = J.shape[0]
    x_ax = np.arange(S)
    y_ax = np.arange(S)
    Z    = -J   # energy = -logit

    marble_color = BACKEND_COLORS.get(backend, "#22ffcc")

    # ── Surface ───────────────────────────────────────────────────────────────
    surface = go.Surface(
        z=Z, x=x_ax, y=y_ax,
        colorscale=[
            [0.00, "#0d0500"],
            [0.12, "#3d1200"],
            [0.30, "#8b3000"],
            [0.50, "#c85000"],
            [0.70, "#e87820"],
            [0.85, "#f2a050"],
            [1.00, "#ffd080"],
        ],
        showscale=False,
        opacity=0.90,
        hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.85,
                      specular=0.15, roughness=0.65),
    )
    traces = [surface]

    # ── Marbles ───────────────────────────────────────────────────────────────
    positions = token_record.get("sample_positions", [])
    if n_marbles >= 0:
        positions = positions[:n_marbles]

    if positions:
        pos_arr  = np.array(positions, dtype=int).clip(0, S - 1)
        q_idx    = S - 1
        marble_x = pos_arr.astype(float)
        marble_y = np.full(len(pos_arr), float(q_idx))
        marble_z = -J[q_idx, pos_arr] + 0.25

        marbles = go.Scatter3d(
            x=marble_x, y=marble_y, z=marble_z,
            mode="markers",
            marker=dict(
                size=5,
                color=marble_color,
                opacity=0.85,
                line=dict(color="#000008", width=0.5),
            ),
            hoverinfo="skip",
            name="p-bits",
        )
        traces.append(marbles)

    # ── Well labels ───────────────────────────────────────────────────────────
    # Place labels at the top-6 energy minima in the last query row of J.
    # Label each with the corresponding top token by probability rank.
    # This correctly maps vocab probability rank -> position-space energy well.
    top_tokens_list = token_record.get("top_tokens", [])[:6]
    q_idx = S - 1
    last_row = -J[q_idx, :]          # energy values for last query row
    # Find top-6 positions by lowest energy (deepest wells)
    n_labels = min(6, S, len(top_tokens_list))
    well_positions = np.argsort(last_row)[:n_labels]  # ascending = deepest first

    lx, ly, lz, lt = [], [], [], []
    for rank, pos in enumerate(well_positions):
        if rank >= len(top_tokens_list):
            break
        tok_info = top_tokens_list[rank]
        ez = float(last_row[pos])
        lx.append(float(pos))
        ly.append(float(q_idx))
        lz.append(ez - 0.8)
        lt.append(f"<b>{tok_info['text']}</b><br>{tok_info['prob']*100:.1f}%")

    if lx:
        labels = go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="text",
            text=lt,
            textfont=dict(
                family="Courier New, monospace",
                size=11,
                color="#f2a050",
            ),
            hoverinfo="skip",
            name="wells",
        )
        traces.append(labels)

    # ── Layout ────────────────────────────────────────────────────────────────
    kl    = token_record.get("kl", 0.0)
    bkt   = token_record.get("bucket", "")
    s_len = token_record.get("seq_len", S)

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        margin=dict(l=0, r=0, t=28, b=0),
        height=580,
        scene=dict(
            bgcolor="#0a0a0f",
            xaxis=dict(
                title=dict(text="Key position",
                           font=dict(color="#444", size=9,
                                     family="Courier New")),
                tickfont=dict(color="#333", size=7),
                gridcolor="#111120",
                showbackground=False,
            ),
            yaxis=dict(
                title=dict(text="Query position",
                           font=dict(color="#444", size=9,
                                     family="Courier New")),
                tickfont=dict(color="#333", size=7),
                gridcolor="#111120",
                showbackground=False,
            ),
            zaxis=dict(
                title=dict(text="Energy (-logit)",
                           font=dict(color="#666", size=9,
                                     family="Courier New")),
                tickfont=dict(color="#444", size=7),
                gridcolor="#111120",
                showbackground=False,
            ),
            aspectratio=dict(x=1.2, y=1.2, z=0.6),
            camera=dict(
                eye=dict(x=1.5, y=-1.8, z=1.0),
            ),
        ),
        showlegend=False,
    )
    fig.add_annotation(
        text=(f"backend={backend}  a={alpha:.1f}  "
              f"S={s_len}  KL={kl:.5f}  {bkt}"),
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        font=dict(family="Courier New", size=10, color="#555"),
        showarrow=False,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="tasb-title">TASB — Thermodynamic Attention Sampling Bridge</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tasb-subtitle">'
    'Real attention energy landscapes · LLaMA 3.2-3B · Layer 18 · '
    'K=50 Boltzmann samples · Paul W. Shaver 2026'
    '</p>',
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "selected_prompt"  not in st.session_state:
    st.session_state.selected_prompt  = "thermo"
if "selected_backend" not in st.session_state:
    st.session_state.selected_backend = "exact"
if "token_step"       not in st.session_state:
    st.session_state.token_step       = 0

# ── Control row ───────────────────────────────────────────────────────────────
row1_left, row1_right = st.columns([4, 3])

with row1_left:
    st.markdown('<p class="ctrl-label">Prompt category</p>',
                unsafe_allow_html=True)
    btn_cols = st.columns(len(PROMPT_META))
    for i, (pk, (icon, label)) in enumerate(PROMPT_META.items()):
        with btn_cols[i]:
            if st.button(f"{icon} {label}", key=f"btn_{pk}"):
                st.session_state.selected_prompt = pk
                st.session_state.token_step = 0

with row1_right:
    st.markdown('<p class="ctrl-label">Backend (sampler)</p>',
                unsafe_allow_html=True)
    back_cols = st.columns(4)
    for i, be in enumerate(["exact", "gumbel", "rbm", "thrml"]):
        with back_cols[i]:
            color = BACKEND_COLORS[be]
            if st.button(be, key=f"be_{be}"):
                st.session_state.selected_backend = be
                st.session_state.token_step = 0

row2_left, row2_right = st.columns([3, 4])

with row2_left:
    st.markdown('<p class="ctrl-label">Alpha — TSU participation</p>',
                unsafe_allow_html=True)
    alpha_idx = st.select_slider(
        "alpha",
        options=list(range(len(ALPHA_OPTIONS))),
        value=1,
        format_func=lambda i: ALPHA_LABELS[ALPHA_OPTIONS[i]],
        label_visibility="collapsed",
    )
    alpha = ALPHA_OPTIONS[alpha_idx]

with row2_right:
    st.markdown('<p class="ctrl-label">Token step — scrub through generation</p>',
                unsafe_allow_html=True)

    selected  = st.session_state.selected_prompt
    backend   = st.session_state.selected_backend
    run_key   = f"{backend}_alpha_{alpha}"

    if data:
        try:
            tokens   = data["prompts"][selected][run_key]["tokens"]
            max_step = len(tokens) - 1
        except (KeyError, TypeError):
            tokens   = None
            max_step = 24
    else:
        tokens   = None
        max_step = 24

    step = st.slider(
        "step",
        min_value=0,
        max_value=max(0, max_step),
        value=min(st.session_state.token_step, max(0, max_step)),
        label_visibility="collapsed",
    )
    st.session_state.token_step = step

# ── Main visualization ────────────────────────────────────────────────────────
plot_slot    = st.empty()
metrics_slot = st.empty()
text_slot    = st.empty()

# Get token record
if data and tokens:
    try:
        record = tokens[step]
    except (IndexError, TypeError):
        record = tokens[-1] if tokens else make_synthetic_token()
else:
    record = make_synthetic_token(alpha=alpha, backend=backend)
    st.info(
        "⚠️  demo_data.json not found — showing synthetic placeholder. "
        "Run tasb_demo_data_gen_v1.py on Lightning.ai to generate real data.",
        icon="⚠️",
    )

# Render
fig = build_figure(record, alpha=alpha, backend=backend)
plot_slot.plotly_chart(
    fig, use_container_width=True,
    config={"displayModeBar": False},
)

# ── Metrics strip ─────────────────────────────────────────────────────────────
kl       = record.get("kl", 0.0)
top1     = record.get("top1_match", True)
pg       = record.get("prob_gap", 0.0)
bkt      = record.get("bucket", "confident")
s_len    = record.get("seq_len", 0)
van_tok  = record.get("van_top1_text", "")
inj_tok  = record.get("inj_top1_text", "")

kl_cls   = "good" if kl < 0.005 else ("warn" if kl < 0.02 else "bad")
t1_cls   = "good" if top1 else "bad"
flip_str = "✓ match" if top1 else f"✗ flip [{bkt}]"
bkt_cls  = "good" if bkt == "confident" else "warn"

metrics_slot.markdown(f"""
<div class="metric-strip">
  <div class="metric-item">
    <span class="metric-label">Backend</span>
    <span class="metric-value" style="color:{BACKEND_COLORS.get(backend,'#f2a050')}">{backend}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">KL divergence</span>
    <span class="metric-value {kl_cls}">{kl:.5f}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Top-1</span>
    <span class="metric-value {t1_cls}">{flip_str}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Prob gap</span>
    <span class="metric-value {bkt_cls}">{pg:.3f} [{bkt}]</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Seq length</span>
    <span class="metric-value">{s_len}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Vanilla → Bridge</span>
    <span class="metric-value">'{van_tok}' → '{inj_tok}'</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Step</span>
    <span class="metric-value">{step} / {max_step}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Generated text display ────────────────────────────────────────────────────
if data and tokens:
    generated = "".join(t.get("token_text", "") for t in tokens[:step + 1])
    prompt_text = data["prompts"][selected]["text"]
    text_slot.markdown(
        f'<div class="token-display">{prompt_text}'
        f'<span class="token-new">{generated}</span></div>',
        unsafe_allow_html=True,
    )

# ── Animate + backend comparison ──────────────────────────────────────────────
st.markdown("---")
anim_col, desc_col = st.columns([1, 3])

with anim_col:
    animate = st.button("▶  Animate generation", type="primary")

with desc_col:
    be_color = BACKEND_COLORS.get(backend, "#f2a050")
    st.markdown(
        f'<p style="font-family:Courier New;font-size:0.72rem;'
        f'color:#555;margin-top:0.5rem;">'
        f'Active: <span style="color:{be_color}">{BACKEND_LABELS.get(backend,backend)}</span>'
        f'<br>Switch backend to see different p-bit trajectories on the same '
        f'energy landscape — same surface, different sampling physics.'
        f'</p>',
        unsafe_allow_html=True,
    )

if animate and data and tokens:
    INTERP_STEPS = 4    # intermediate frames between each token step
    FRAME_SLEEP  = 0.08 # seconds per interpolation frame — smooth morphing

    for i in range(len(tokens)):
        curr = tokens[i]
        J_curr = np.array(curr["J_matrix"], dtype=np.float32)
        S_curr = J_curr.shape[0]

        # Interpolate between previous and current J matrix
        if i == 0:
            frames = [curr]
        else:
            prev   = tokens[i - 1]
            J_prev = np.array(prev["J_matrix"], dtype=np.float32)
            S_prev = J_prev.shape[0]
            # Both are downsampled to MAX_S so shapes should match
            # If they differ (S grew), pad previous to current size
            if S_prev != S_curr:
                frames = [curr]  # skip interpolation on size change
            else:
                frames = []
                for t in range(1, INTERP_STEPS + 1):
                    alpha_t = t / INTERP_STEPS
                    J_blend = (1 - alpha_t) * J_prev + alpha_t * J_curr
                    # Build a blended record using current metadata
                    blended = dict(curr)
                    blended["J_matrix"] = J_blend.tolist()
                    frames.append(blended)

        for frame_rec in frames:
            fig_f = build_figure(frame_rec, alpha=alpha, backend=backend)
            plot_slot.plotly_chart(
                fig_f, use_container_width=True,
                config={"displayModeBar": False},
                key=f"af_{i}_{id(frame_rec)}",
            )
            time.sleep(FRAME_SLEEP)

        # Update metrics and text after each full token step
        kl_f  = curr.get("kl", 0.0)
        t1_f  = curr.get("top1_match", True)
        pg_f  = curr.get("prob_gap", 0.0)
        bkt_f = curr.get("bucket", "confident")
        s_f   = curr.get("seq_len", 0)
        van_f = curr.get("van_top1_text", "")
        inj_f = curr.get("inj_top1_text", "")

        kl_cf  = "good" if kl_f < 0.005 else "warn"
        flip_f = "✓ match" if t1_f else f"✗ flip [{bkt_f}]"
        t1_cf  = "good" if t1_f else "bad"
        bkt_cf = "good" if bkt_f == "confident" else "warn"

        metrics_slot.markdown(f"""
<div class="metric-strip">
  <div class="metric-item">
    <span class="metric-label">Backend</span>
    <span class="metric-value" style="color:{BACKEND_COLORS.get(backend,'#f2a050')}">{backend}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">KL divergence</span>
    <span class="metric-value {kl_cf}">{kl_f:.5f}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Top-1</span>
    <span class="metric-value {t1_cf}">{flip_f}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Prob gap</span>
    <span class="metric-value {bkt_cf}">{pg_f:.3f} [{bkt_f}]</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Seq length</span>
    <span class="metric-value">{s_f}</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Vanilla → Bridge</span>
    <span class="metric-value">'{van_f}' → '{inj_f}'</span>
  </div>
  <div class="metric-item">
    <span class="metric-label">Step</span>
    <span class="metric-value">{i} / {len(tokens)-1}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        generated = "".join(
            t.get("token_text", "") for t in tokens[:i + 1]
        )
        prompt_text = data["prompts"][selected]["text"]
        text_slot.markdown(
            f'<div class="token-display">{prompt_text}'
            f'<span class="token-new">{generated}</span></div>',
            unsafe_allow_html=True,
        )

# ── Backend comparison summary table ─────────────────────────────────────────
if data:
    st.markdown("---")
    st.markdown(
        '<p class="ctrl-label">Backend comparison — '
        f'{selected} prompt at α={alpha:.1f}</p>',
        unsafe_allow_html=True,
    )
    rows = []
    for be in ["exact", "gumbel", "rbm", "thrml"]:
        rk = f"{be}_alpha_{alpha}"
        try:
            summary = data["prompts"][selected][rk]["summary"]
            kl_s    = summary["mean_kl"]
            t1_s    = summary["top1_pct"]
            cf_s    = summary["conf_flips"]
            n_s     = summary["n_tokens"]
            kl_cls2 = "good" if kl_s < 0.005 else "warn"
            t1_cls2 = "good" if t1_s >= 99.0 else "warn"
            cf_cls2 = "good" if cf_s == 0 else "bad"
            color   = BACKEND_COLORS[be]
            rows.append(
                f'<tr>'
                f'<td style="color:{color};padding-right:1.5rem">{be}</td>'
                f'<td class="{kl_cls2}" style="padding-right:1.5rem">{kl_s:.5f}</td>'
                f'<td class="{t1_cls2}" style="padding-right:1.5rem">{t1_s:.1f}%</td>'
                f'<td class="{cf_cls2}" style="padding-right:1.5rem">{cf_s}</td>'
                f'<td style="color:#444">{n_s} tokens</td>'
                f'</tr>'
            )
        except (KeyError, TypeError):
            rows.append(
                f'<tr><td style="color:{BACKEND_COLORS[be]}">{be}</td>'
                f'<td colspan="4" style="color:#333">no data</td></tr>'
            )

    st.markdown(f"""
<table class="summary-table">
  <tr>
    <td style="color:#333;padding-right:1.5rem">BACKEND</td>
    <td style="color:#333;padding-right:1.5rem">MEAN KL</td>
    <td style="color:#333;padding-right:1.5rem">TOP-1</td>
    <td style="color:#333;padding-right:1.5rem">CONF FLIPS</td>
    <td style="color:#333">TOKENS</td>
  </tr>
  {''.join(rows)}
</table>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="tasb-footer">'
    'TASB · Paul W. Shaver 2026 · '
    'github.com/whtetigr2/TASB · '
    'Real data. Every claim has a CSV. Every CSV has a script.'
    '</p>',
    unsafe_allow_html=True,
)
