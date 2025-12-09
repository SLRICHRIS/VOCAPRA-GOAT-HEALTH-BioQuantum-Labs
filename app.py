#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v4.4 (Dark HUD)
Dark-theme visuals: transparent matplotlib, neon accents, dark background.
Includes unsupervised metrics, audio playback, prototypes, Grad-CAM.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict
import io, math

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Elite dark HUD palette
PALETTE = {
    "bg": "#05060A",        # deep near-black HUD
    "panel": "rgba(10,12,15,0.6)",
    "neon": "#00f3ff",      # cyan neon
    "accent": "#bc13fe",    # magenta accent
    "muted": "#7d8790",
    "fg": "#e6eef6",
    "bar_cmap": "plasma",
    "mfcc_cmap": "magma",
    "cam_cmap": "inferno"
}

# Matplotlib global dark style adjustments
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
# HELPERS
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
    feats = np.concatenate([mf, d1, d2], axis=0).T.astype(np.float32)
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
    if model_path is None: return None, None, None, None
    model = tf.keras.models.load_model(str(model_path))
    conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_name = layer.name
            break
    grad_model = None
    if conv_name is not None:
        try:
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_name).output, model.output])
        except Exception:
            grad_model = None
    return model, grad_model, conv_name, model_path

@st.cache_resource(show_spinner=False)
def load_label_map():
    json_path = resolve_artifact("label_to_idx*.json")
    if json_path is None: return {}, {}, None
    with open(json_path, "r") as f:
        label_to_idx_raw = json.load(f)
    if all(isinstance(v, (int, str)) for v in label_to_idx_raw.values()):
        label_to_idx = {str(k): int(v) for k, v in label_to_idx_raw.items()}
        idx_to_label = {int(v): str(k) for k, v in label_to_idx_raw.items()}
    else:
        idx_to_label = {int(k): str(v) for k, v in label_to_idx_raw.items()}
        label_to_idx = {v: k for k, v in idx_to_label.items()}
    return idx_to_label, label_to_idx, json_path

# Grad-CAM
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

# Unsup helpers
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
    if s.size == 0: return 0.0
    rms = np.sqrt(np.mean(s**2))
    noise = s[np.abs(s) < np.percentile(np.abs(s), 25)]
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-12
    return 20 * math.log10((rms + 1e-12) / noise_rms)

def rms_energy(signal):
    s = np.asarray(signal, dtype=np.float32); return float(np.sqrt(np.mean(s**2)))

def zero_crossing_rate(signal):
    z = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)
    return float(np.mean(z))

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
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.size==0 or b.size==0: return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))

@st.cache_resource(show_spinner=False)
def build_prototypes_from_dir(protos_dir: Path, model, target_frames=TARGET_FRAMES):
    if not protos_dir.exists(): return {}
    protos: Dict[str, np.ndarray] = {}
    for cls in sorted([d for d in protos_dir.iterdir() if d.is_dir()]):
        embs = []
        for wav in sorted(cls.glob("*.wav")):
            try:
                data, sr = sf.read(str(wav))
                if data.ndim==2: data = np.mean(data, axis=1)
                if sr != SR:
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SR)
                feats = compute_mfcc_with_deltas(data, sr=SR)
                fixed = to_fixed_frames(feats, target_frames)
                x_in = np.expand_dims(fixed, axis=0).astype(np.float32)
                emb = get_embedding(model, x_in, layer_name=None)
                if emb is not None: embs.append(emb)
            except Exception:
                continue
        if embs:
            protos[cls.name] = np.mean(np.vstack(embs), axis=0)
    return protos

# Plot helpers tuned for dark theme
def make_neon_plot(x, y, color=PALETTE["neon"], title="Waveform", sr=SR):
    fig, ax = plt.subplots(figsize=(9,2.6), dpi=120)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.plot(x, y, color=color, linewidth=1.6)
    for n in range(1,5): ax.plot(x, y, color=color, linewidth=1.6+n*0.6, alpha=0.12/n)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#2b2f33'); ax.tick_params(axis='x', colors=PALETTE["muted"])
    ax.set_yticks([]); ax.set_xlabel("Time (s)", color=PALETTE["muted"], fontsize=9)
    ax.set_title(title, color=PALETTE["fg"], fontsize=11)
    ax.grid(axis='x', color='#111214', linestyle='--', linewidth=0.4, alpha=0.25)
    plt.tight_layout(); return fig

def plot_probability_bars(probs, idx_to_label, top_k=8):
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    top_labels = [idx_to_label[i] for i in sorted_idx]; top_vals = probs[sorted_idx]
    fig, ax = plt.subplots(figsize=(5.2,3), dpi=120); fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    cmap = plt.get_cmap(PALETTE["bar_cmap"]); colors = cmap(np.linspace(0.15,0.85,len(top_vals)))
    y_pos = np.arange(len(top_vals))
    bars = ax.barh(y_pos, top_vals[::-1], color=colors[::-1], height=0.6, edgecolor='#111214', linewidth=0.35)
    ax.set_yticks(y_pos); ax.set_yticklabels([l.upper() for l in top_labels[::-1]], fontsize=9, color=PALETTE["fg"])
    ax.set_xlabel("Probability", color=PALETTE["muted"], fontsize=9); ax.set_xlim(0,1.0); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda x,pos: f"{x*100:.0f}%")
    ax.grid(axis='x', color='#0f1112', linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#2b2f33')
    for bar in bars:
        width = bar.get_width(); xpos = width + 0.02 if width < 0.18 else width - 0.02
        ha = 'left' if width < 0.18 else 'right'; color = '#111214' if width < 0.18 else '#fff'
        ax.text(xpos, bar.get_y()+bar.get_height()/2.0, f"{width:.2f}", ha=ha, va='center', fontsize=8, color=color, fontfamily='monospace')
    plt.tight_layout(); return fig

def plot_gradcam_and_mfcc(fixed, cam, y, sr):
    T_frames = fixed.shape[0]; F_bins = fixed.shape[1]; duration_s = len(y)/sr
    fig, (ax_mfcc, ax_cam) = plt.subplots(2,1, figsize=(12,4), gridspec_kw={'height_ratios':[1,0.25]}, dpi=120)
    fig.patch.set_alpha(0); ax_mfcc.patch.set_alpha(0)
    im = ax_mfcc.imshow(fixed.T, origin='lower', aspect='auto', cmap=PALETTE["mfcc_cmap"])
    ax_mfcc.set_ylabel("Feature bins", color=PALETTE["muted"], fontsize=9)
    ax_mfcc.set_xticks(np.linspace(0, T_frames-1, 5)); ax_mfcc.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color=PALETTE["muted"])
    ax_mfcc.set_title("Feature map (MFCC + deltas)", color=PALETTE["fg"], fontsize=10)
    ax_cam.imshow(np.tile(cam, (F_bins,1)), origin='lower', aspect='auto', cmap=PALETTE["cam_cmap"], alpha=0.95, extent=[0,T_frames,0,F_bins])
    ax_cam.set_xlabel("Time (s)", color=PALETTE["muted"], fontsize=9)
    ax_cam.set_xticks(np.linspace(0, T_frames-1, 5)); ax_cam.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color=PALETTE["muted"])
    ax_cam.set_yticks([])
    cbar = fig.colorbar(im, ax=[ax_mfcc, ax_cam], orientation='vertical', pad=0.02); cbar.set_label("Feature magnitude", color=PALETTE["muted"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"]); plt.setp(plt.getp(cbar.ax.axes,'yticklabels'), color=PALETTE["muted"])
    plt.tight_layout(); return fig

# -----------------------
# UI (dark)
# -----------------------
st.set_page_config(page_title="VOCAPRA HUD (Dark)", page_icon="💠", layout="wide")
st.markdown(f"""
    <style>
      body {{ background: {PALETTE['bg']}; color: {PALETTE['fg']}; }}
      .stApp {{ background: linear-gradient(180deg, {PALETTE['bg']}, #07080c); }}
      .label-small {{ font-size:0.85rem; color:{PALETTE['muted']}; }}
      .hud-title {{ font-size:1.8rem; font-weight:700; color:{PALETTE['fg']}; }}
      .fixed-footer {{ position:fixed; bottom:0; left:0; width:100%; background:{PALETTE['bg']}; padding:6px 0; text-align:center; color:#666; }}
      .st-bf {{ background: rgba(255,255,255,0.02) !important; }}
    </style>
""", unsafe_allow_html=True)

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

# Build prototypes if available
prototypes = {}
if model is not None and PROTOTYPES_DIR.exists():
    prototypes = build_prototypes_from_dir(PROTOTYPES_DIR, model, target_frames=TARGET_FRAMES)
    if prototypes:
        st.sidebar.success(f"Loaded {len(prototypes)} class prototypes.")
    else:
        st.sidebar.info("No prototypes found in vocapra_project/prototypes/")

# Sidebar options
st.sidebar.markdown("### Unsupervised options")
enable_mc = st.sidebar.checkbox("Enable MC Dropout (uncertainty)", value=False)
mc_runs = st.sidebar.number_input("MC runs", min_value=5, max_value=50, value=10, step=1) if enable_mc else 10
enable_aug = st.sidebar.checkbox("Check Augmentation Consistency", value=True)
ood_threshold = st.sidebar.slider("OOD softmax-max threshold", 0.0, 1.0, 0.2, 0.01)
embed_layer = st.sidebar.text_input("Embedding layer name (optional)", value="")

st.markdown("<div class='hud-title'>VOCAPRA <span style='color:#00f3ff'>.AI</span></div>", unsafe_allow_html=True)
st.markdown("<div class='label-small'>// Acoustic Event Recognition — dark HUD</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload WAV", type=["wav"])
if uploaded is None:
    st.stop()

uploaded.seek(0); file_bytes = uploaded.read()
try:
    data, sr_read = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if data.ndim == 2: data = np.mean(data, axis=1)
    if sr_read != SR: y = librosa.resample(data.astype(np.float32), orig_sr=sr_read, target_sr=SR)
    else: y = data.astype(np.float32)
    sr = SR
except Exception:
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=SR, mono=True)

# audio player
st.audio(file_bytes, format='audio/wav')

# features, input
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0).astype(np.float32)

if model is None:
    st.error("Model not found — add best_model*.h5 to vocapra_project/")
    st.stop()

# inference
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
mc_std = None; mc_mean_conf = None
if enable_mc:
    try:
        mc_preds = mc_dropout_predict(model, x_in, n=mc_runs)
        mc_mean = mc_preds.mean(axis=0)
        mc_std = float(mc_preds.std(axis=0).mean())
        mc_mean_conf = float(np.max(mc_mean))
    except Exception:
        mc_std = None; mc_mean_conf = None

# augmentation consistency
consistency = None
if enable_aug:
    try:
        a1 = augment_noise(y, snr_db=15); a2 = augment_pitch_shift(y, sr=sr, n_steps=1); a3 = augment_time_stretch(y, rate=0.95)
        aug_preds = []
        for sig in [a1,a2,a3]:
            f = compute_mfcc_with_deltas(sig, sr=sr); fx = to_fixed_frames(f, TARGET_FRAMES); xi = np.expand_dims(fx, axis=0).astype(np.float32)
            try:
                p = model.predict(xi, verbose=0)[0]
            except Exception:
                p = model.predict(np.expand_dims(xi, axis=-1), verbose=0)[0]
            aug_preds.append(int(np.argmax(softmax_safe(p))))
        consistency = float(sum(1 for p in aug_preds if p==pred_idx)/len(aug_preds))
    except Exception:
        consistency = None

# embedding & prototypes
embedding = None; embedding_norm = None; proto_sims = {}
try:
    embedding = get_embedding(model, x_in, layer_name=embed_layer if embed_layer.strip() else None)
    if embedding is not None:
        embedding_norm = float(np.linalg.norm(embedding))
        for cls, proto in prototypes.items():
            proto_sims[cls] = cosine_similarity(embedding, proto)
        proto_sorted = sorted(proto_sims.items(), key=lambda x:-x[1]) if proto_sims else []
    else:
        proto_sorted = []
except Exception:
    proto_sorted = []

# primary card
st.markdown(f"""
<div style="padding:10px; border-left:4px solid {PALETTE['accent']}; background: rgba(255,255,255,0.02);">
 <div style="display:flex; justify-content:space-between; align-items:center;">
  <div><div style="color:{PALETTE['accent']}; font-weight:700">PRIMARY DETECTION</div>
  <div style="font-size:2rem; color:{PALETTE['fg']};">{pred_label}</div></div>
  <div style="text-align:right"><div style="color:{PALETTE['muted']}">CONFIDENCE</div><div style="font-family:monospace; color:{PALETTE['fg']};">{conf_top*100:05.2f}%</div></div>
 </div></div>
""", unsafe_allow_html=True)

# visuals
g1, g2 = st.columns([1.7,1.3])
with g1:
    st.markdown("<div class='label-small'>SIGNAL OSCILLOSCOPE</div>", unsafe_allow_html=True)
    fig_w = make_neon_plot(np.linspace(0, len(y)/sr, len(y)), y, title="Waveform", sr=sr)
    st.pyplot(fig_w); plt.close(fig_w)
with g2:
    st.markdown("<div class='label-small'>PROBABILITIES</div>", unsafe_allow_html=True)
    fig_p = plot_probability_bars(probs, idx_to_label, top_k=8); st.pyplot(fig_p); plt.close(fig_p)

# metrics
st.markdown("### Unsupervised metrics (no ground-truth required)")
c1,c2,c3 = st.columns(3)
c1.metric("Top confidence", f"{conf_top*100:05.2f}%"); c2.metric("Entropy", f"{ent:.3f}"); c3.metric("Confidence margin", f"{margin:.3f}")
d1,d2,d3 = st.columns(3)
d1.metric("Estimated SNR (dB)", f"{snr_db:.2f}"); d2.metric("RMS energy", f"{rms_val:.5f}"); d3.metric("Zero-cross rate", f"{zcr:.4f}")

if enable_mc and mc_std is not None:
    st.markdown(f"**MC Dropout** — mean top-conf: {mc_mean_conf:.4f} • avg std across classes: {mc_std:.4f}")
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

# Grad-CAM
if grad_model is not None:
    try:
        cam, _ = run_gradcam(grad_model, x_in)
        fig_gc = plot_gradcam_and_mfcc(fixed, cam, y, sr); st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)
        st.pyplot(fig_gc); plt.close(fig_gc)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# footer
st.markdown(f"<div style='position:fixed; bottom:0; left:0; width:100%; background:{PALETTE['bg']}; padding:6px 0; text-align:center; color:#666;'>&copy; 2025 Rights Reserved by BioQuantum Labs</div>", unsafe_allow_html=True)
