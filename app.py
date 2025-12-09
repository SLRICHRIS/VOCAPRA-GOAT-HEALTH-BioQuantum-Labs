#!/usr/bin/env python3
"""
VOCAPRA Streamlit App — Full corrected single-file app (Elite Dark)
- Glossy/high-contrast metric cards
- Goat-specific tagline
- Robust audio loading & inference, Grad-CAM, prototype similarity
Save as app.py and run: streamlit run app.py
"""

from __future__ import annotations
import json
import io
import math
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

# Optional Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# components for safe HTML embedding
from streamlit.components.v1 import html as st_html

# -----------------------
# CONFIG
# -----------------------
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80
ARTIFACT_DIR = Path("vocapra_project")
PROTOTYPES_DIR = ARTIFACT_DIR / "prototypes"

PALETTE = {
    "bg": "#030406",
    "panel_alpha": 0.02,
    "neon": "#00f3ff",
    "accent": "#bc13fe",
    "muted": "#8b94a6",
    "fg": "#e8eef6",
    "bar_cmap": "plasma",
    "mfcc_cmap": "magma",
    "cam_cmap": "inferno"
}

# Matplotlib dark tweaks
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": PALETTE["fg"],
    "axes.labelcolor": PALETTE["fg"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "axes.edgecolor": "#2b2f33",
    "font.family": "DejaVu Sans",
})

# -----------------------
# HELPERS: artifacts, features, model load
# -----------------------
def resolve_artifact(pattern: str) -> Optional[Path]:
    if not ARTIFACT_DIR.exists():
        return None
    matches: List[Path] = sorted(ARTIFACT_DIR.glob(pattern))
    return matches[0] if matches else None

def compute_mfcc_with_deltas(y: np.ndarray, sr: int = SR) -> np.ndarray:
    n_fft = max(256, int(WIN_LEN * sr))
    hop_length = max(64, int(HOP_LEN * sr))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    feats = np.concatenate([mf, d1, d2], axis=0).T.astype(np.float32)  # (T, F)
    return feats

def to_fixed_frames(seq: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    T, F = seq.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = seq[:target_frames]
    else:
        out[-T:, :] = seq
    return out

@st.cache_resource(show_spinner=False)
def load_model_and_gradcam():
    model_path = resolve_artifact("best_model*.h5")
    if model_path is None:
        return None, None, None, None
    model = tf.keras.models.load_model(str(model_path))
    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break
    grad_model = None
    if conv_layer_name is not None:
        try:
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])
        except Exception:
            grad_model = None
    return model, grad_model, conv_layer_name, model_path

@st.cache_resource(show_spinner=False)
def load_label_map():
    json_path = resolve_artifact("label_to_idx*.json")
    if json_path is None:
        return {}, {}, None
    with open(json_path, "r") as f:
        label_to_idx_raw = json.load(f)
    if all(isinstance(v, (int, str)) for v in label_to_idx_raw.values()):
        label_to_idx = {str(k): int(v) for k, v in label_to_idx_raw.items()}
        idx_to_label = {int(v): str(k) for k, v in label_to_idx_raw.items()}
    else:
        idx_to_label = {int(k): str(v) for k, v in label_to_idx_raw.items()}
        label_to_idx = {v: k for k, v in idx_to_label.items()}
    return idx_to_label, label_to_idx, json_path

# -----------------------
# Grad-CAM
# -----------------------
def run_gradcam(grad_model, sample):
    if grad_model is None:
        raise ValueError("grad_model unavailable")
    sample_tf = tf.convert_to_tensor(sample.astype(np.float32))
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(sample_tf)
        preds_tensor = preds[-1] if isinstance(preds, (list, tuple)) else preds
        class_idx = int(tf.argmax(preds_tensor[0]).numpy())
        loss = preds_tensor[:, class_idx]
    grads = tape.gradient(loss, conv_outs)
    if grads is None:
        return np.zeros(sample.shape[1]), class_idx
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    T_in = sample.shape[1]
    cam_resized = np.interp(np.linspace(0, cam.shape[0]-1, T_in), np.arange(cam.shape[0]), cam)
    return cam_resized, class_idx

# -----------------------
# UNSUPERVISED METRICS HELPERS
# -----------------------
def softmax_safe(probs):
    p = np.array(probs, dtype=np.float32)
    p = np.maximum(p, 1e-12)
    return p / p.sum()

def entropy_of_probs(probs):
    p = softmax_safe(probs)
    return -float(np.sum(p * np.log(p + 1e-12)))

def confidence_margin(probs):
    top = np.sort(probs)[::-1]
    return float(top[0] - (top[1] if len(top) > 1 else 0.0))

def snr_estimate(signal):
    s = np.asarray(signal, dtype=np.float32)
    if s.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(s**2))
    noise = s[np.abs(s) < np.percentile(np.abs(s), 25)]
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-12
    return 20 * math.log10((rms + 1e-12) / noise_rms)

def rms_energy(signal):
    s = np.asarray(signal, dtype=np.float32)
    return float(np.sqrt(np.mean(s**2)))

def zero_crossing_rate(signal):
    z = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)
    return float(np.mean(z))

def augment_noise(signal, snr_db=20):
    sig = signal.astype(np.float32)
    rms = np.sqrt(np.mean(sig**2))
    noise_rms = rms / (10**(snr_db/20))
    noise = np.random.normal(0, noise_rms, size=sig.shape)
    return sig + noise

def augment_time_stretch(signal, rate=1.05):
    try:
        return librosa.effects.time_stretch(signal, rate=rate)
    except Exception:
        return signal

def augment_pitch_shift(signal, sr, n_steps=1):
    try:
        return librosa.effects.pitch_shift(signal, sr, n_steps=n_steps)
    except Exception:
        return signal

def mc_dropout_predict(model, x_input, n=10):
    preds = []
    for _ in range(n):
        try:
            p = model(x_input, training=True).numpy()
        except Exception:
            p = model.predict(x_input, verbose=0)
        preds.append(p[0])
    return np.vstack(preds)

def get_embedding(model, x_input, layer_name: Optional[str]=None):
    try:
        if layer_name:
            layer = model.get_layer(layer_name)
            emb_model = tf.keras.models.Model(model.inputs, layer.output)
            emb = emb_model.predict(x_input, verbose=0)
            return emb.reshape((emb.shape[0], -1))[0]
        else:
            if len(model.layers) >= 2:
                pen = model.layers[-2].output
                emb_model = tf.keras.models.Model(model.inputs, pen)
                emb = emb_model.predict(x_input, verbose=0)
                return emb.reshape((emb.shape[0], -1))[0]
    except Exception:
        return None
    return None

def cosine_similarity(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

# -----------------------
# PROTOTYPE BUILD
# -----------------------
@st.cache_resource(show_spinner=False)
def build_prototypes_from_dir(protos_dir: Path, model, target_frames=TARGET_FRAMES):
    if not protos_dir.exists():
        return {}
    protos: Dict[str, np.ndarray] = {}
    for cls in sorted([d for d in protos_dir.iterdir() if d.is_dir()]):
        embs = []
        for wav in sorted(cls.glob("*.wav")):
            try:
                data, sr = sf.read(str(wav))
                if data.ndim == 2:
                    data = np.mean(data, axis=1)
                if sr != SR:
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SR)
                feats = compute_mfcc_with_deltas(data, sr=SR)
                fixed = to_fixed_frames(feats, target_frames)
                x_in = np.expand_dims(fixed, axis=0).astype(np.float32)
                emb = get_embedding(model, x_in, layer_name=None)
                if emb is not None:
                    embs.append(emb)
            except Exception:
                continue
        if embs:
            protos[cls.name] = np.mean(np.vstack(embs), axis=0)
    return protos

# -----------------------
# Plot helpers (Plotly + Matplotlib fallbacks)
# -----------------------
def plotly_waveform(y, sr, title="Waveform", neon_color=PALETTE["neon"]):
    if not PLOTLY_AVAILABLE:
        return None
    t = np.linspace(0, len(y)/sr, len(y))
    max_points = 6000
    if t.size > max_points:
        idx = np.linspace(0, t.size-1, max_points).astype(int)
        t = t[idx]
        y_plot = y[idx]
    else:
        y_plot = y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_plot, mode='lines',
                             line=dict(color=neon_color, width=1.6),
                             hovertemplate='time: %{x:.3f}s<br>amp: %{y:.3f}<extra></extra>'))
    for w, a in [(3, 0.06), (6, 0.03), (12, 0.02)]:
        fig.add_trace(go.Scatter(x=t, y=y_plot, mode='lines',
                                 line=dict(color=neon_color, width=w),
                                 opacity=a, hoverinfo='skip', showlegend=False))
    fig.update_layout(margin=dict(l=6, r=6, t=28, b=6),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      title=dict(text=title, font=dict(color=PALETTE["fg"], size=12)),
                      xaxis=dict(title='Time (s)', color=PALETTE["muted"]),
                      yaxis=dict(showticklabels=False))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.02)')
    return fig

def plotly_prob_bars(probs, idx_to_label, top_k=8):
    if not PLOTLY_AVAILABLE:
        return None
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    labels = [idx_to_label[i].upper() for i in sorted_idx]
    vals = probs[sorted_idx]
    color_seq = px.colors.sequential.Plasma
    if len(vals) > 1:
        colors = [color_seq[int(i*(len(color_seq)-1)/(len(vals)-1))] for i in range(len(vals))]
    else:
        colors = [color_seq[0]]
    fig = go.Figure(go.Bar(x=vals[::-1], y=[l for l in labels[::-1]],
                           orientation='h', text=[f"{v*100:.2f}%" for v in vals[::-1]],
                           textposition='outside',
                           marker=dict(color=colors[::-1], line=dict(color='rgba(0,0,0,0.3)', width=0.6)),
                           hovertemplate='%{y}<br>Prob: %{x:.3f}<extra></extra>'))
    fig.update_layout(margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(range=[0, 1], tickformat='.0%', color=PALETTE["muted"]),
                      yaxis=dict(autorange='reversed', color=PALETTE["fg"]))
    return fig

def make_neon_plot_matplotlib(x, y, color=PALETTE["neon"], title="Waveform"):
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=100)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.plot(x, y, color=color, linewidth=1.6)
    for n in range(1, 5):
        ax.plot(x, y, color=color, linewidth=1.6 + n*0.6, alpha=0.12/n)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#2b2f33')
    ax.tick_params(axis='x', colors=PALETTE["muted"])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", color=PALETTE["muted"], fontsize=9)
    ax.set_title(title, color=PALETTE["fg"], fontsize=11)
    ax.grid(axis='x', color='#111214', linestyle='--', linewidth=0.4, alpha=0.25)
    plt.tight_layout()
    return fig

def plot_probability_bars_matplotlib(probs, idx_to_label, top_k=8):
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    top_labels = [idx_to_label[i] for i in sorted_idx]
    top_vals = probs[sorted_idx]
    fig, ax = plt.subplots(figsize=(5.2, 3), dpi=100)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    cmap = plt.get_cmap(PALETTE["bar_cmap"])
    colors = cmap(np.linspace(0.15, 0.85, len(top_vals)))
    y_pos = np.arange(len(top_vals))
    bars = ax.barh(y_pos, top_vals[::-1], color=colors[::-1], height=0.6, edgecolor='#111214', linewidth=0.35)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.upper() for l in top_labels[::-1]], fontsize=9, color=PALETTE["fg"])
    ax.set_xlabel("Probability", color=PALETTE["muted"], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
    ax.grid(axis='x', color='#0f1112', linestyle='--', linewidth=0.5, alpha=0.6)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#2b2f33')
    for bar in bars:
        width = bar.get_width()
        xpos = width + 0.02 if width < 0.18 else width - 0.02
        ha = 'left' if width < 0.18 else 'right'
        color = '#111214' if width < 0.18 else '#fff'
        ax.text(xpos, bar.get_y() + bar.get_height()/2.0, f"{width:.2f}", ha=ha, va='center', fontsize=8, color=color, fontfamily='monospace')
    plt.tight_layout()
    return fig

# -----------------------
# ELITE CSS (glossy metric cards)
# -----------------------
ELITE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Manrope:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
:root {{
  --bg: {PALETTE['bg']};
  --neon: {PALETTE['neon']};
  --accent: {PALETTE['accent']};
  --muted: {PALETTE['muted']};
  --fg: {PALETTE['fg']};
}}
body, .stApp {{
  background: radial-gradient(1200px 500px at 10% 10%, rgba(0,243,255,0.03), transparent 6%),
              radial-gradient(1000px 400px at 90% 20%, rgba(188,19,254,0.03), transparent 6%),
              var(--bg) !important;
  color: var(--fg) !important;
  font-family: 'Inter', 'Manrope', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}
.hud-card {{
  background: linear-gradient(180deg, rgba(255,255,255,{PALETTE['panel_alpha']}), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 12px;
  padding: 12px;
  backdrop-filter: blur(8px) saturate(120%);
  box-shadow: 0 8px 30px rgba(0,0,0,0.6), 0 0 40px rgba(0,243,255,0.02) inset;
}}
.hud-title {{ font-family: 'Manrope', Inter, sans-serif; font-weight:700; font-size:1.8rem; color:var(--fg);}}
.hud-hero {{ font-family: 'Manrope', Inter, sans-serif; font-weight:800; font-size:2.8rem; background: linear-gradient(90deg, var(--fg), var(--neon), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; position: relative; display:inline-block;}}
.hud-hero::after{{ content:''; position:absolute; left:-40%; top:0; height:100%; width:40%; background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.12), rgba(255,255,255,0.0)); transform:skewX(-18deg); animation: hero-shimmer 4s linear infinite; pointer-events:none; }}
@keyframes hero-shimmer{{ 0%{{ left:-40% }} 50% {{ left:120% }} 100% {{ left:120% }} }}
.confidence-pulse {{ font-family: 'Inter', monospace; font-weight:700; font-size:1.6rem; color:var(--fg); text-shadow: 0 0 12px rgba(0,243,255,0.12); animation: pulse 2.4s infinite; }}
@keyframes pulse {{ 0% {{ transform: scale(1); opacity:1; }} 50% {{ transform: scale(1.03); opacity:0.95; text-shadow: 0 0 18px rgba(0,243,255,0.22); }} 100% {{ transform: scale(1); opacity:1; }} }}
.label-small {{ font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; }}

.elite-row {{ display:flex; gap:14px; margin-bottom:14px; }}
.elite-card {{
  flex:1;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.005));
  border-radius:14px;
  padding:14px 16px;
  min-height:92px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  position:relative;
  overflow:hidden;
  transition: transform 0.22s cubic-bezier(.2,.9,.3,1), box-shadow 0.22s;
  border: 1px solid rgba(255,255,255,0.03);
}}
.elite-card::before {{
  content:'';
  position:absolute;
  left:0; top:0; bottom:0; width:6px;
  background: linear-gradient(180deg, var(--neon), var(--accent));
  box-shadow: 0 0 18px rgba(0,243,255,0.06);
  border-top-left-radius:14px;
  border-bottom-left-radius:14px;
}}
.elite-card::after {{
  content:'';
  position:absolute;
  left:-30%; top:-40%;
  width:160%; height:100%;
  background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.12), rgba(255,255,255,0.06));
  transform: rotate(-18deg);
  filter: blur(12px);
  opacity:0.35;
  pointer-events:none;
}}
.elite-card:hover {{
  transform: translateY(-6px);
  box-shadow: 0 18px 40px rgba(0,0,0,0.7), 0 0 40px rgba(0,243,255,0.03);
}}
.elite-label {{ font-size:0.70rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; }}
.elite-value {{
  font-size:1.92rem;
  color:var(--fg);
  font-weight:800;
  font-family: 'JetBrains Mono', 'Manrope', monospace;
  margin-top:6px;
  line-height:1;
  text-shadow: 0 2px 8px rgba(0,0,0,0.6);
}}
.elite-bar-wrap {{ margin-top:8px; width:100%; height:8px; background: rgba(255,255,255,0.02); border-radius:8px; overflow:hidden; }}
.elite-bar {{
  height:100%;
  width:0%;
  border-radius:8px;
  background: linear-gradient(90deg, var(--neon), var(--accent));
  box-shadow: 0 6px 18px rgba(0,243,255,0.06) inset;
  transition: width 0.6s cubic-bezier(.2,.9,.3,1);
}}
@media (max-width:900px) {{
  .elite-row {{ flex-direction:column; }}
  .elite-value {{ font-size:1.6rem; }}
}}
</style>
"""

# -----------------------
# render_elite_metrics (components)
# -----------------------
def render_elite_metrics(metrics: Dict[str, str], cols: int = 3,
                         neon=PALETTE["neon"], accent=PALETTE["accent"],
                         muted=PALETTE["muted"], fg=PALETTE["fg"]):
    """
    Render glossy neon metric cards via components HTML.
    Each card shows a value and a tiny progress bar scaled between 0-1 when possible.
    """
    items = list(metrics.items())
    rows = [items[i:i+cols] for i in range(0, len(items), cols)]

    def safe_ratio(v):
        try:
            s = str(v).strip()
            if s.endswith('%'):
                num = float(s.replace('%','').strip())
                return max(0.0, min(100.0, num)) / 100.0
            num = float(s)
            if abs(num) <= 1.0:
                return max(0.0, min(1.0, num))
            # compress larger values (arctan squashing)
            return float((2/math.pi) * math.atan(num/10.0))
        except Exception:
            return 0.0

    body = "<div>"
    for row in rows:
        body += "<div class='elite-row'>"
        for label, value in row:
            safe_label = str(label)
            safe_value = str(value)
            ratio = safe_ratio(value)
            pct = int(round(ratio * 100))
            body += f"""
            <div class='elite-card' role="group" aria-label="{safe_label}: {safe_value}">
              <div class='elite-label'>{safe_label}</div>
              <div class='elite-value'>{safe_value}</div>
              <div class='elite-bar-wrap' aria-hidden="true">
                <div class='elite-bar' style='width:{pct}%;'></div>
              </div>
            </div>
            """
        body += "</div>"
    body += "</div>"

    html_full = ELITE_CSS + body  # include CSS to be safe
    height = max(140, 120 * len(rows))
    st_html(html_full, height=height, scrolling=True)

# -----------------------
# Glossy Grad-CAM + MFCC plot function (drop-in)
# -----------------------
def plot_mfcc_gradcam_glossy(fixed, cam, audio, sr, title="Feature map (MFCC + deltas)",
                             mfcc_cmap=PALETTE["mfcc_cmap"], cam_cmap=PALETTE["cam_cmap"],
                             bg_color=PALETTE["bg"], fg_color=PALETTE["fg"], muted=PALETTE["muted"]):
    """
    Glossy, technical Grad-CAM + MFCC panel.
    Inputs:
      - fixed: (T, F) feature matrix (MFCC + deltas)
      - cam: (T,) cam vector aligned to time frames
      - audio, sr: for duration ticks
    Returns matplotlib.figure.Figure
    """
    T_frames = fixed.shape[0]
    F_bins = fixed.shape[1]
    duration_s = len(audio) / sr

    # Figure with strong dark background and inner glossy card
    fig = plt.figure(figsize=(12, 4.2), dpi=120, facecolor="none")
    fig.patch.set_alpha(0)

    # Main rounded panel as backdrop (glossy)
    ax_back = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    ax_back.axis("off")
    rect = patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02,rounding_size=14",
        linewidth=0,
        transform=ax_back.transAxes,
        facecolor=(0.05, 0.05, 0.06, 0.65),  # slightly translucent panel
        edgecolor=(0,0,0,0)
    )
    ax_back.add_patch(rect)

    # Create two axes inside the card with proper spacing
    ax_mfcc = fig.add_axes([0.06, 0.34, 0.88, 0.56])  # left, bottom, width, height
    ax_cam  = fig.add_axes([0.06, 0.14, 0.88, 0.14], sharex=ax_mfcc)

    # MFCC heatmap
    im = ax_mfcc.imshow(
        fixed.T,
        origin="lower",
        aspect="auto",
        cmap=mfcc_cmap,
        interpolation='nearest',
        norm=Normalize(vmin=np.percentile(fixed, 5), vmax=np.percentile(fixed, 99))
    )

    # Styling axes
    ax_mfcc.set_ylabel("Feature bins", color=muted, fontsize=10)
    ax_mfcc.set_yticks(np.linspace(0, F_bins - 1, min(6, F_bins)).astype(int))
    ax_mfcc.set_xticks(np.linspace(0, T_frames - 1, 5))
    ax_mfcc.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color=muted)
    ax_mfcc.tick_params(axis='y', colors=muted)
    ax_mfcc.set_title(title, color=fg_color, fontsize=12, pad=8, weight='600')

    # Thin inner frame for the MFCC axis for a technical look
    for spine in ax_mfcc.spines.values():
        spine.set_edgecolor("#121314")
        spine.set_linewidth(0.7)

    # CAM heatstrip under MFCC
    ax_cam.imshow(
        np.tile(cam, (F_bins, 1)),
        origin='lower',
        aspect='auto',
        cmap=cam_cmap,
        interpolation='bilinear',
        alpha=0.95,
        extent=[0, T_frames, 0, F_bins]
    )
    ax_cam.set_xlabel("Time (s)", color=muted, fontsize=10)
    ax_cam.set_yticks([])  # no y ticks for the heatstrip
    ax_cam.set_xticks(np.linspace(0, T_frames - 1, 5))
    ax_cam.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color=muted)
    ax_cam.tick_params(axis='x', colors=muted)

    # Add a slender colorbar with glossy border on the right
    cax = fig.add_axes([0.92, 0.18, 0.02, 0.58])  # x, y, w, h
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', pad=0.01)
    cbar.set_label("Feature magnitude", color=muted, fontsize=9, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color=muted)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=muted)

    # Add subtle gridlines (horizontal guide lines) to give technical feel
    ax_mfcc.grid(axis='x', color='#0b0c0d', linestyle='--', linewidth=0.4, alpha=0.4)
    ax_cam.grid(axis='x', color='#0b0c0d', linestyle='--', linewidth=0.4, alpha=0.4)

    # Add a neon marker indicating the top activation frame
    max_idx = int(np.argmax(cam))
    tx = max_idx
    ax_cam.axvline(x=tx, color=PALETTE["neon"], linewidth=1.1, alpha=0.9, linestyle='-')
    ax_mfcc.axvline(x=tx, color=PALETTE["neon"], linewidth=0.9, alpha=0.8, linestyle='-')

    # Overlay a small rounded legend card at top-right inside the panel
    legend_box = patches.FancyBboxPatch(
        (0.68, 0.78), 0.28, 0.18,
        boxstyle="round,pad=0.02,rounding_size=8",
        transform=ax_back.transAxes,
        linewidth=0.5,
        edgecolor=(1,1,1,0.03),
        facecolor=(1,1,1,0.02)
    )
    ax_back.add_patch(legend_box)
    ax_back.text(0.70, 0.85, "Activation", transform=ax_back.transAxes, color=muted, fontsize=9)
    ax_back.text(0.70, 0.82, f"Peak frame: {max_idx}", transform=ax_back.transAxes, color=fg_color, fontsize=10, weight='600')

    # Tight layout and return
    plt.subplots_adjust(left=0.03, right=0.91, top=0.95, bottom=0.08, hspace=0.08)
    return fig

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="VOCAPRA HUD (Corrected)", page_icon="💠", layout="wide")
st.markdown(ELITE_CSS, unsafe_allow_html=True)

# Load artifacts
idx_to_label, label_to_idx, label_json_path = load_label_map()
model, grad_model, conv_name, model_path = load_model_and_gradcam()

with st.sidebar:
    st.markdown("### SYSTEM LOG")
    st.code(f"""
> INIT_SEQ... OK
> MODEL: {model_path.name if model_path else 'ERR'}
> SR: {SR} Hz
> FRAMES: {TARGET_FRAMES}
> STATUS: {'ONLINE' if model_path and idx_to_label else 'OFFLINE'}
""", language="yaml")
    st.markdown("---")
    st.caption("Conv1D Stack / GlobalAvgPool / Softmax")

if model is None or not idx_to_label:
    st.sidebar.warning("Artifacts missing in vocapra_project/ — UI will load but inference is disabled.")

# build prototypes (if available)
prototypes = {}
if model is not None and PROTOTYPES_DIR.exists():
    prototypes = build_prototypes_from_dir(PROTOTYPES_DIR, model, target_frames=TARGET_FRAMES)
    if prototypes:
        st.sidebar.success(f"Loaded {len(prototypes)} class prototypes.")
    else:
        st.sidebar.info("No prototypes found in vocapra_project/prototypes/")

# Sidebar controls
st.sidebar.markdown("### Unsupervised options")
enable_mc = st.sidebar.checkbox("Enable MC Dropout (uncertainty)", value=False)
mc_runs = st.sidebar.number_input("MC runs", min_value=5, max_value=50, value=10, step=1) if enable_mc else 10
enable_aug = st.sidebar.checkbox("Check Augmentation Consistency", value=True)
ood_threshold = st.sidebar.slider("OOD softmax-max threshold", 0.0, 1.0, 0.2, 0.01)
embed_layer = st.sidebar.text_input("Embedding layer name (optional)", value="")

# Header (goat tagline)
st.markdown("<div class='hud-title'>VOCAPRA <span style='color:#00f3ff'>.AI</span></div>", unsafe_allow_html=True)
st.markdown("<div class='label-small'>// Real-time Acoustic Intelligence for Goats — Detect. Explain. Act.</div>", unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader("Upload WAV", type=["wav"])
if uploaded is None:
    st.stop()

# Read audio robustly
uploaded.seek(0)
file_bytes = uploaded.read()
try:
    data, sr_read = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    if sr_read != SR:
        y = librosa.resample(data.astype(np.float32), orig_sr=sr_read, target_sr=SR)
    else:
        y = data.astype(np.float32)
    sr = SR
except Exception:
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=SR, mono=True)

# audio player
st.audio(file_bytes, format='audio/wav')

# features & model input
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0).astype(np.float32)

if model is None:
    st.error("Model not found — place best_model*.h5 in vocapra_project/ and reload.")
    st.stop()

# inference (with channel fallback)
try:
    probs = model.predict(x_in, verbose=0)[0]
except Exception:
    try:
        x_try = np.expand_dims(x_in, axis=-1)
        probs = model.predict(x_try, verbose=0)[0]
        x_in = x_try
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

probs = softmax_safe(probs)
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, "UNKNOWN").upper()
conf_top = float(probs[pred_idx])
ent = entropy_of_probs(probs)
margin = confidence_margin(probs)
snr_db = snr_estimate(y)
rms_val = rms_energy(y)
zcr = zero_crossing_rate(y)
is_ood = conf_top < ood_threshold

# MC dropout
mc_std = None
mc_mean_conf = None
if enable_mc:
    try:
        mc_preds = mc_dropout_predict(model, x_in, n=mc_runs)
        mc_mean = mc_preds.mean(axis=0)
        mc_std = float(mc_preds.std(axis=0).mean())
        mc_mean_conf = float(np.max(mc_mean))
    except Exception:
        mc_std = None
        mc_mean_conf = None

# augmentation consistency
consistency = None
if enable_aug:
    try:
        a1 = augment_noise(y, snr_db=15)
        a2 = augment_pitch_shift(y, sr=sr, n_steps=1)
        a3 = augment_time_stretch(y, rate=0.95)
        aug_preds = []
        for sig in [a1, a2, a3]:
            f = compute_mfcc_with_deltas(sig, sr=sr)
            fx = to_fixed_frames(f, TARGET_FRAMES)
            xi = np.expand_dims(fx, axis=0).astype(np.float32)
            try:
                p = model.predict(xi, verbose=0)[0]
            except Exception:
                p = model.predict(np.expand_dims(xi, axis=-1), verbose=0)[0]
            aug_preds.append(int(np.argmax(softmax_safe(p))))
        consistency = float(sum(1 for p in aug_preds if p == pred_idx) / len(aug_preds))
    except Exception:
        consistency = None

# embedding & prototype similarity
embedding = None
embedding_norm = None
proto_sims = {}
try:
    embedding = get_embedding(model, x_in, layer_name=embed_layer if embed_layer.strip() else None)
    if embedding is not None:
        embedding_norm = float(np.linalg.norm(embedding))
        for cls, proto in prototypes.items():
            proto_sims[cls] = cosine_similarity(embedding, proto)
        proto_sorted = sorted(proto_sims.items(), key=lambda x: -x[1]) if proto_sims else []
    else:
        proto_sorted = []
except Exception:
    proto_sorted = []

# Primary card (elite)
st.markdown(f"""
<div class='hud-card' style='display:flex; justify-content:space-between; align-items:center;'>
  <div>
    <div class='label-small'>PRIMARY DETECTION</div>
    <div class='hud-hero'>{pred_label}</div>
  </div>
  <div style='text-align:right;'>
    <div class='label-small'>CONFIDENCE</div>
    <div class='confidence-pulse'>{conf_top*100:05.2f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Visuals: Plotly (if available) with Matplotlib fallback
g1, g2 = st.columns([1.7, 1.3])
with g1:
    st.markdown("<div class='label-small'>SIGNAL OSCILLOSCOPE</div>", unsafe_allow_html=True)
    plotted = False
    if PLOTLY_AVAILABLE:
        try:
            fig_wave = plotly_waveform(y, sr, title="Waveform", neon_color=PALETTE["neon"])
            if fig_wave is not None:
                st.plotly_chart(fig_wave, use_container_width=True)
                plotted = True
        except Exception:
            plotted = False
    if not plotted:
        fig_w = make_neon_plot_matplotlib(np.linspace(0, len(y)/sr, len(y)), y, color=PALETTE["neon"], title="Waveform")
        st.pyplot(fig_w)
        plt.close(fig_w)

with g2:
    st.markdown("<div class='label-small'>PROBABILITIES</div>", unsafe_allow_html=True)
    plotted = False
    if PLOTLY_AVAILABLE:
        try:
            fig_p = plotly_prob_bars(probs, idx_to_label, top_k=8)
            if fig_p is not None:
                st.plotly_chart(fig_p, use_container_width=True)
                plotted = True
        except Exception:
            plotted = False
    if not plotted:
        fig_p = plot_probability_bars_matplotlib(probs, idx_to_label, top_k=8)
        st.pyplot(fig_p)
        plt.close(fig_p)

# ===== High-contrast metrics (replacement) =====
metrics_map = {
    "Top confidence": f"{conf_top*100:05.2f}%",
    "Entropy": f"{ent:.3f}",
    "Confidence margin": f"{margin:.3f}",
    "Estimated SNR (dB)": f"{snr_db:.2f}",
    "RMS energy": f"{rms_val:.5f}",
    "Zero-cross rate": f"{zcr:.4f}"
}
st.markdown("### Unsupervised metrics (no ground-truth required)", unsafe_allow_html=True)
render_elite_metrics(metrics_map, cols=3)
# ==============================================

if enable_mc and mc_std is not None:
    st.markdown(f"**MC Dropout** — mean top-conf: {mc_mean_conf:.4f}  •  avg std across classes: {mc_std:.4f}")

if enable_aug:
    st.markdown(f"**Augmentation consistency**: {consistency if consistency is not None else 'N/A'}")

if embedding_norm is not None:
    st.markdown(f"**Embedding norm**: {embedding_norm:.3f}")

if proto_sorted:
    st.markdown("#### Similarity to class prototypes")
    for cls, sim in proto_sorted[:6]:
        st.write(f"{cls.upper():12s}  —  cosine_sim: {sim:.3f}")

if is_ood:
    st.warning(f"Softmax max ({conf_top:.3f}) < OOD threshold ({ood_threshold:.3f}) — sample may be Out-Of-Distribution")

# Grad-CAM (use glossy function)
if grad_model is not None:
    try:
        cam, _ = run_gradcam(grad_model, x_in)
        fig = plot_mfcc_gradcam_glossy(fixed, cam, y, sr,
                                      title="Feature map (MFCC + deltas)",
                                      mfcc_cmap=PALETTE["mfcc_cmap"],
                                      cam_cmap=PALETTE["cam_cmap"])
        st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# Footer
st.markdown(f"<div style='position:fixed; bottom:0; left:0; width:100%; background:{PALETTE['bg']}; padding:8px 0; text-align:center; color:#666;'>&copy; 2025 Rights Reserved by BioQuantum Labs</div>", unsafe_allow_html=True)
