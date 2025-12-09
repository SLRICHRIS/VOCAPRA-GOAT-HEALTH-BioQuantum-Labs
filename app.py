#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v4.2 (Cyber-HUD Edition)
Full app with unsupervised metrics and improved visualisations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List
import io
import math

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80
ARTIFACT_DIR = Path("vocapra_project")

# =============================================================================
# HELPERS: artifacts, features, model load
# =============================================================================
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
    # Normalize
    if all(isinstance(v, (int, str)) for v in label_to_idx_raw.values()):
        label_to_idx = {str(k): int(v) for k, v in label_to_idx_raw.items()}
        idx_to_label = {int(v): str(k) for k, v in label_to_idx_raw.items()}
    else:
        idx_to_label = {int(k): str(v) for k, v in label_to_idx_raw.items()}
        label_to_idx = {v: k for k, v in idx_to_label.items()}
    return idx_to_label, label_to_idx, json_path

# =============================================================================
# Grad-CAM
# =============================================================================
def run_gradcam(grad_model, sample):
    """
    Compute Grad-CAM for a single sample (shape: (1, T, F) or (1, T, F, 1)).
    Returns cam_resized (T,) and class_idx (int).
    """
    if grad_model is None:
        raise ValueError("grad_model unavailable")
    sample_tf = tf.convert_to_tensor(sample.astype(np.float32))
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(sample_tf)
        if isinstance(preds, (list, tuple)):
            preds_tensor = preds[-1]
        else:
            preds_tensor = preds
        class_idx_tensor = tf.argmax(preds_tensor[0])
        class_idx = int(class_idx_tensor.numpy())
        loss = preds_tensor[:, class_idx]
    grads = tape.gradient(loss, conv_outs)
    if grads is None:
        T_in = sample.shape[1]
        return np.zeros(T_in), class_idx
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    T_in = sample.shape[1]
    T_cam = cam.shape[0]
    cam_resized = np.interp(np.linspace(0, T_cam - 1, T_in), np.arange(T_cam), cam)
    return cam_resized, class_idx

# =============================================================================
# UNSUPERVISED METRICS HELPERS
# =============================================================================
def softmax_safe(probs):
    p = np.array(probs, dtype=np.float32)
    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    return p

def entropy_of_probs(probs):
    p = softmax_safe(probs)
    return -float(np.sum(p * np.log(p + 1e-12)))

def confidence_margin(probs):
    top = np.sort(probs)[::-1]
    if len(top) < 2:
        return float(top[0])
    return float(top[0] - top[1])

def snr_estimate(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(signal**2))
    noise_floor = np.sqrt(np.mean((signal[np.abs(signal) < np.percentile(np.abs(signal), 25)])**2) + 1e-12)
    if noise_floor == 0:
        return float('inf') if rms > 0 else 0.0
    return 20 * math.log10(rms / noise_floor + 1e-12)

def rms_energy(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.sqrt(np.mean(signal**2)))

def zero_crossing_rate(signal, sr=SR):
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

def get_embedding(model, x_input, layer_name=None):
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
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def mc_dropout_predict(model, x_input, n=10):
    preds = []
    for _ in range(n):
        try:
            p = model(x_input, training=True).numpy()
        except Exception:
            p = model.predict(x_input, verbose=0)
        preds.append(p[0])
    return np.vstack(preds)

# =============================================================================
# PLOTTING: neon waveform, probability bar, gradcam block
# =============================================================================
def make_neon_plot(x, y, color='#00f3ff', title="Waveform", sr=SR):
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=120)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.plot(x, y, color=color, linewidth=1.6, alpha=1.0, label='Waveform')
    for n in range(1, 6):
        ax.plot(x, y, color=color, linewidth=1.6 + n * 0.9, alpha=0.12 / n)

    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(axis='x', colors='#bbbbbb', labelsize=9)
    ax.set_yticks([])

    ax.set_xlabel("Time (s)", color='#bbbbbb', fontsize=9, fontfamily='monospace')
    ax.set_title(title, color='#dddddd', fontsize=11, pad=6)
    ax.grid(axis='x', color='#222222', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc='upper right', fontsize=8, labelcolor='#dddddd')

    plt.tight_layout()
    return fig

def plot_probability_bars(probs, idx_to_label, top_k=8):
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    top_labels = [idx_to_label[i] for i in sorted_idx]
    top_vals = probs[sorted_idx]

    fig, ax = plt.subplots(figsize=(5.2, 3), dpi=120)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.15, 0.85, len(top_vals)))

    y_pos = np.arange(len(top_vals))
    bars = ax.barh(y_pos, top_vals[::-1], color=colors[::-1], height=0.6, edgecolor='black', linewidth=0.35)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.upper() for l in top_labels[::-1]], fontsize=9, fontfamily='monospace', color='#e8e8e8')
    ax.set_xlabel("Probability", color='#bbbbbb', fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")

    ax.grid(axis='x', color='#2a2a2a', linestyle='--', linewidth=0.5, alpha=0.6)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#444444')

    for bar in bars:
        width = bar.get_width()
        xpos = width + 0.02 if width < 0.18 else width - 0.02
        ha = 'left' if width < 0.18 else 'right'
        color = '#222' if width < 0.18 else '#ffffff'
        ax.text(xpos, bar.get_y() + bar.get_height()/2.0, f"{width:.2f}", ha=ha, va='center', fontsize=8, color=color, fontfamily='monospace')

    plt.tight_layout()
    return fig

def plot_gradcam_and_mfcc(fixed, cam, y, sr):
    T_frames = fixed.shape[0]
    F_bins = fixed.shape[1]
    duration_s = len(y) / sr

    fig, (ax_mfcc, ax_cam) = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 0.25]}, dpi=120)
    fig.patch.set_alpha(0)
    ax_mfcc.patch.set_alpha(0)

    im = ax_mfcc.imshow(fixed.T, origin='lower', aspect='auto', cmap='magma')
    ax_mfcc.set_ylabel("Feature bins", color='#bbbbbb', fontsize=9)
    ax_mfcc.set_xticks(np.linspace(0, T_frames - 1, 5))
    ax_mfcc.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color='#bbbbbb')
    ax_mfcc.set_title("Feature map (MFCC + deltas)", color='#e8e8e8', fontsize=10)

    ax_cam.imshow(np.tile(cam, (F_bins, 1)), origin='lower', aspect='auto', cmap='inferno', alpha=0.95, extent=[0, T_frames, 0, F_bins])
    ax_cam.set_xlabel("Time (s)", color='#bbbbbb', fontsize=9)
    ax_cam.set_xticks(np.linspace(0, T_frames - 1, 5))
    ax_cam.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration_s, 5)], color='#bbbbbb')
    ax_cam.set_yticks([])

    cbar = fig.colorbar(im, ax=[ax_mfcc, ax_cam], orientation='vertical', pad=0.02)
    cbar.set_label("Feature magnitude", color='#bbbbbb', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#bbbbbb')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#bbbbbb')

    plt.tight_layout()
    return fig

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="VOCAPRA HUD", page_icon="💠", layout="wide")
st.markdown(
    """
    <style>
    body { background: #030508; color: #e8e8e8; }
    .label-small { font-size:0.85rem; color:#9aa; }
    .hud-title { font-size:1.8rem; font-weight:700; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load artifacts
idx_to_label, label_to_idx, label_json_path = load_label_map()
model, grad_model, conv_name, model_path = load_model_and_gradcam()

with st.sidebar:
    st.markdown("### SYSTEM LOG")
    st.code(
        f"""
        > INIT_SEQ... OK
        > MODEL: {model_path.name if model_path else 'ERR'}
        > SR: {SR} Hz
        > FRAMES: {TARGET_FRAMES}
        > STATUS: {'ONLINE' if model_path and idx_to_label else 'OFFLINE'}
        """, language="yaml"
    )
    st.markdown("---")
    st.caption("Conv1D Stack / GlobalAvgPool / Softmax")

if model is None or not idx_to_label:
    st.warning("Artifacts missing in vocapra_project/ — UI will load but inference is disabled.")

st.markdown("<div class='hud-title'>VOCAPRA <span style='color:#00f3ff'>.AI</span></div>", unsafe_allow_html=True)
st.markdown("<div class='label-small'>// Acoustic Event Recognition + Unsupervised Metrics</div>", unsafe_allow_html=True)

# Unsupervised options
right = st.sidebar
enable_mc_dropout = right.checkbox("Enable MC Dropout (uncertainty)", value=False)
mc_n = right.number_input("MC runs", min_value=5, max_value=50, value=10, step=1) if enable_mc_dropout else 10
enable_aug_consistency = right.checkbox("Check Augmentation Consistency", value=True)
ood_threshold = right.slider("OOD softmax-max threshold", 0.0, 1.0, 0.2, 0.01)
embedding_layer = right.text_input("Embedding layer name (optional)", value="")

# File uploader
st.markdown("### Upload WAV")
uploaded = st.file_uploader("", type=["wav"], label_visibility="hidden")

if uploaded is None:
    st.stop()

# Read audio reliably
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

# Features and model input
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0).astype(np.float32)

if model is None:
    st.error("Model not found — place best_model*.h5 in vocapra_project/ and reload.")
    st.stop()

# Inference with channel fallback
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
rms = rms_energy(y)
zcr = zero_crossing_rate(y, sr=sr)
is_ood = conf_top < ood_threshold

# MC Dropout
mc_std = None
mc_mean_conf = None
if enable_mc_dropout:
    try:
        mc_preds = mc_dropout_predict(model, x_in, n=mc_n)
        mc_mean = mc_preds.mean(axis=0)
        mc_std = float(mc_preds.std(axis=0).mean())
        mc_mean_conf = float(np.max(mc_mean))
    except Exception:
        mc_std = None
        mc_mean_conf = None

# Augmentation consistency
consistency_score = None
if enable_aug_consistency:
    try:
        a1 = augment_noise(y, snr_db=15)
        a2 = augment_pitch_shift(y, sr=sr, n_steps=1)
        a3 = augment_time_stretch(y, rate=0.95)
        aug_signals = [a1, a2, a3]
        preds = []
        for sig in aug_signals:
            f = compute_mfcc_with_deltas(sig, sr=sr)
            fx = to_fixed_frames(f, TARGET_FRAMES)
            xi = np.expand_dims(fx, axis=0).astype(np.float32)
            try:
                p = model.predict(xi, verbose=0)[0]
            except Exception:
                p = model.predict(np.expand_dims(xi, axis=-1), verbose=0)[0]
            preds.append(int(np.argmax(softmax_safe(p))))
        consistency_score = float(sum(1 for p in preds if p == pred_idx) / len(preds))
    except Exception:
        consistency_score = None

# Embedding (optional)
embedding_sim = None
if embedding_layer or True:
    try:
        emb = get_embedding(model, x_in, layer_name=embedding_layer if embedding_layer.strip() else None)
        if emb is not None:
            embedding_sim = float(np.linalg.norm(emb))
    except Exception:
        embedding_sim = None

# Display primary card
st.markdown(f"<div style='padding:8px; border-left:4px solid #bc13fe; background:rgba(10,15,25,0.6)'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div><div style='color:#bc13fe;font-weight:700'>PRIMARY DETECTION</div><div style='font-size:2.1rem'>{pred_label}</div></div>"
            f"<div style='text-align:right'><div style='color:#aaa'>CONFIDENCE</div><div style='font-family:monospace'>{conf_top*100:05.2f}%</div></div>"
            f"</div></div>", unsafe_allow_html=True)

# Visuals
g1, g2 = st.columns([1.7, 1.3])
with g1:
    st.markdown("<div class='label-small'>SIGNAL OSCILLOSCOPE</div>", unsafe_allow_html=True)
    fig_w = make_neon_plot(np.linspace(0, len(y)/sr, len(y)), y, title="Waveform")
    st.pyplot(fig_w)
    plt.close(fig_w)
with g2:
    st.markdown("<div class='label-small'>PROBABILITIES</div>", unsafe_allow_html=True)
    fig_p = plot_probability_bars(probs, idx_to_label, top_k=8)
    st.pyplot(fig_p)
    plt.close(fig_p)

# Metrics table
st.markdown("### Unsupervised metrics (no ground-truth required)")
cols = st.columns(3)
cols[0].metric("Top confidence", f"{conf_top*100:05.2f}%")
cols[1].metric("Entropy", f"{ent:.3f}")
cols[2].metric("Confidence margin", f"{margin:.3f}")

cols2 = st.columns(3)
cols2[0].metric("Estimated SNR (dB)", f"{snr_db:.2f}")
cols2[1].metric("RMS energy", f"{rms:.5f}")
cols2[2].metric("Zero-cross rate", f"{zcr:.4f}")

if enable_mc_dropout and mc_std is not None:
    st.markdown(f"**MC Dropout** — mean top-conf: {mc_mean_conf:.4f}  •  avg std across classes: {mc_std:.4f}")

if enable_aug_consistency:
    st.markdown(f"**Augmentation consistency**: {consistency_score if consistency_score is not None else 'N/A'}")

if embedding_sim is not None:
    st.markdown(f"**Embedding norm** (proxy): {embedding_sim:.3f}")

if is_ood:
    st.warning(f"Softmax max ({conf_top:.3f}) < OOD threshold ({ood_threshold:.3f}) — sample may be Out-Of-Distribution")

# Grad-CAM if available
if grad_model is not None:
    try:
        st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)
        cam, _ = run_gradcam(grad_model, x_in)
        fig_gc = plot_gradcam_and_mfcc(fixed, cam, y, sr)
        st.pyplot(fig_gc)
        plt.close(fig_gc)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# Footer
st.markdown(
    """
    <div style="position:fixed; bottom:0; left:0; width:100%; background:#030508; padding:6px 0; text-align:center; color:#666;">
        &copy; 2025 Rights Reserved by BioQuantum Labs
    </div>
    """,
    unsafe_allow_html=True,
)
