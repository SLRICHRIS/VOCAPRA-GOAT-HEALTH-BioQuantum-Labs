#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v4.2 (Cyber-HUD Edition)
Added unsupervised evaluation metrics (no true labels required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import io
import math

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# CONFIG & SETUP
# =============================================================================
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80
ARTIFACT_DIR = Path("vocapra_project")

# =============================================================================
# UTILS & PIPELINE (Model and Feature Extraction Logic)
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
    if all(isinstance(v, (int, str)) for v in label_to_idx_raw.values()):
        label_to_idx = {str(k): int(v) for k, v in label_to_idx_raw.items()}
        idx_to_label = {int(v): str(k) for k, v in label_to_idx_raw.items()}
    else:
        idx_to_label = {int(k): str(v) for k, v in label_to_idx_raw.items()}
        label_to_idx = {v: k for k, v in idx_to_label.items()}
    return idx_to_label, label_to_idx, json_path

def run_gradcam(grad_model, sample):
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
# NEW: UNSUPERVISED METRICS HELPERS
# =============================================================================
def softmax(probs):
    # assumes model output is already softmax — just safe normalization
    p = np.array(probs, dtype=np.float32)
    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    return p

def entropy_of_probs(probs):
    p = softmax(probs)
    return -float(np.sum(p * np.log(p + 1e-12)))

def confidence_margin(probs):
    top = np.sort(probs)[::-1]
    if len(top) < 2:
        return float(top[0])
    return float(top[0] - top[1])

def snr_estimate(signal):
    # rough SNR estimate: ratio of signal RMS to noise floor (using median filtering)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(signal**2))
    # estimate noise as low-amplitude quantile (not perfect but ok)
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
    # add gaussian noise to achieve approx snr_db
    sig = signal.astype(np.float32)
    rms = np.sqrt(np.mean(sig**2))
    noise_rms = rms / (10**(snr_db/20))
    noise = np.random.normal(0, noise_rms, size=sig.shape)
    return sig + noise

def augment_time_stretch(signal, rate=1.05):
    return librosa.effects.time_stretch(signal, rate=rate)

def augment_pitch_shift(signal, sr, n_steps=1):
    return librosa.effects.pitch_shift(signal, sr, n_steps=n_steps)

def get_embedding(model, x_input, layer_name=None):
    # If layer_name provided, try to extract that layer's output as embedding
    try:
        if layer_name:
            layer = model.get_layer(layer_name)
            emb_model = tf.keras.models.Model(model.inputs, layer.output)
            emb = emb_model.predict(x_input, verbose=0)
            return emb.reshape((emb.shape[0], -1))[0]
        else:
            # fallback: use penultimate layer if possible
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
    """Run model.predict with training=True to enable dropout if present.
       Returns array (n, num_classes)."""
    preds = []
    for _ in range(n):
        try:
            p = model(x_input, training=True).numpy()
        except Exception:
            p = model.predict(x_input, verbose=0)
        preds.append(p[0])
    return np.vstack(preds)

# =============================================================================
# ELITE PLOTTING (NEON GLOW EFFECTS)
# =============================================================================
def make_neon_plot(x, y, color='#00f3ff', title="Waveform"):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.plot(x, y, color=color, linewidth=1.2, alpha=1.0)
    for n in range(1, 6):
        ax.plot(x, y, color=color, linewidth=1.2 + n * 0.8, alpha=0.15 / n)
    ax.set_facecolor("none")
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#666666', labelsize=8)
    ax.set_yticks([])
    ax.set_xlabel("TIME DOMAIN", color='#444444', fontfamily='monospace', fontsize=8)
    return fig

# =============================================================================
# STREAMLIT UI CONFIG & CSS INJECTION
# =============================================================================
st.set_page_config(page_title="VOCAPRA HUD", page_icon="💠", layout="wide")

st.markdown(
    """
    <style> /* minimal kept for brevity */ .label-small{font-size:0.75rem;color:#888}</style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOAD ARTIFACTS & SIDEBAR
# =============================================================================
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

# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.markdown("<div class='hud-title'>VOCAPRA <span style='color:#00f3ff'>.AI</span></div>", unsafe_allow_html=True)
st.markdown("<div class='hud-subtitle'>// Acoustic Event Recognition + Unsupervised Metrics</div>", unsafe_allow_html=True)

# Controls for unsupervised metrics
left_col, right_col = st.columns([1.7, 1])
with right_col:
    st.markdown("### Unsupervised options")
    enable_mc_dropout = st.checkbox("Enable MC Dropout (n=10)", value=False)
    mc_n = st.number_input("MC runs", min_value=5, max_value=50, value=10, step=1) if enable_mc_dropout else 0
    enable_aug_consistency = st.checkbox("Check Augmentation Consistency", value=True)
    ood_threshold = st.slider("OOD softmax-max threshold (flag as unknown if below)", 0.0, 1.0, 0.2, 0.01)
    embedding_layer = st.text_input("Embedding layer name (optional)", value="")

c1, c2 = st.columns([1.5, 1])
with c1:
    st.markdown("<div class='label-small'>INPUT STREAM</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload WAV", type=["wav"], label_visibility="collapsed")
with c2:
    num_classes = len(idx_to_label) if idx_to_label else 0
    st.markdown(f"<div class='label-small'>CLASSES: {num_classes}</div>", unsafe_allow_html=True)

if uploaded is None:
    st.stop()

# =============================================================================
# INFERENCE + UNSUPERVISED METRICS
# =============================================================================
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

feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0).astype(np.float32)

if model is None:
    st.error("Model not found — put best_model*.h5 into vocapra_project/ and reload.")
    st.stop()

# attempt inference (try channel dim if needed)
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

probs = softmax(probs)
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, "UNKNOWN").upper()
conf_top = float(probs[pred_idx])
ent = entropy_of_probs(probs)
margin = confidence_margin(probs)
snr_db = snr_estimate(y)
rms = rms_energy(y)
zcr = zero_crossing_rate(y, sr=sr)
is_ood = conf_top < ood_threshold

# MC Dropout (optional)
mc_std = None
mc_mean_conf = None
if enable_mc_dropout:
    try:
        mc_preds = mc_dropout_predict(model, x_in, n=mc_n)
        mc_mean = mc_preds.mean(axis=0)
        mc_std = float(mc_preds.std(axis=0).mean())  # aggregate std across classes
        mc_mean_conf = float(np.max(mc_mean))
    except Exception as e:
        mc_std = None
        mc_mean_conf = None

# Augmentation consistency
consistency_score = None
if enable_aug_consistency:
    try:
        a1 = augment_noise(y, snr_db=15)
        a2 = augment_pitch_shift(y, sr=sr, n_steps=1)
        # time-stretch may change length; handle carefully
        try:
            a3 = augment_time_stretch(y, rate=0.95)
        except Exception:
            a3 = y
        # extract features & predict for each
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
            preds.append(int(np.argmax(softmax(p))))
        # consistency = fraction of augment preds equal to original pred
        consistency_score = float(sum(1 for p in preds if p == pred_idx) / len(preds))
    except Exception:
        consistency_score = None

# Embedding similarity (if embedding layer exists)
embedding_sim = None
embedding_vec = None
if embedding_layer or True:
    try:
        emb = get_embedding(model, x_in, layer_name=embedding_layer if embedding_layer.strip() else None)
        if emb is not None:
            embedding_vec = emb
            # naive similarity: cosine with class prototype if available
            # compute prototype as average embedding of class outputs (not available here),
            # so we just show embedding norm and leave prototype matching for dataset-time
            embedding_sim = float(np.linalg.norm(emb))
    except Exception:
        embedding_sim = None

# Display results
st.markdown(f"<div class='label-small blink'>Analyzing... COMPLETE</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="border-left:4px solid #bc13fe; padding:12px; background: rgba(10,15,25,0.6);">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="color:#bc13fe; font-size:0.9rem; font-weight:700;">PRIMARY DETECTION</div>
          <div style="font-size:2.5rem; font-weight:700; margin-top:6px;">{pred_label}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.8rem; color:#aaa;">CONFIDENCE</div>
          <div style="font-family:monospace; font-size:1.5rem;">{conf_top*100:05.2f}%</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Visualization
g1, g2 = st.columns([1.7, 1.3])
with g1:
    st.markdown("<div class='label-small'>SIGNAL OSCILLOSCOPE</div>", unsafe_allow_html=True)
    fig = make_neon_plot(np.linspace(0, len(y)/sr, len(y)), y)
    st.pyplot(fig)
    plt.close(fig)
with g2:
    st.markdown("<div class='label-small'>PROBABILITIES</div>", unsafe_allow_html=True)
    sorted_idx = np.argsort(probs)[::-1][:6]
    top_labels = [idx_to_label[i] for i in sorted_idx]
    top_vals = probs[sorted_idx]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.barh(range(len(top_vals)), top_vals[::-1], height=0.5, color='#bc13fe')
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels([l.upper() for l in top_labels[::-1]])
    ax.set_xlim(0,1)
    ax.invert_yaxis()
    for i, rect in enumerate(ax.patches):
        ax.text(rect.get_width()+0.01, rect.get_y()+rect.get_height()/2, f"{rect.get_width():.2f}", va='center')
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# Metrics panel
st.markdown("### Unsupervised metrics (no true labels required)")
cols = st.columns(3)
cols[0].metric("Top confidence", f"{conf_top*100:05.2f}%")
cols[1].metric("Entropy", f"{ent:.3f}")
cols[2].metric("Confidence margin", f"{margin:.3f}")

cols2 = st.columns(3)
cols2[0].metric("Estimated SNR (dB)", f"{snr_db:.2f}")
cols2[1].metric("RMS energy", f"{rms:.5f}")
cols2[2].metric("Zero-cross rate", f"{zcr:.4f}")

if enable_mc_dropout:
    st.markdown(f"**MC Dropout** — mean top-conf: {mc_mean_conf:.4f}  •  avg std across classes: {mc_std:.4f}")

if enable_aug_consistency:
    st.markdown(f"**Augmentation consistency**: {consistency_score if consistency_score is not None else 'N/A'} (fraction of augmentations with same top class)")

if embedding_sim is not None:
    st.markdown(f"**Embedding norm** (proxy for representation strength): {embedding_sim:.3f}")

if is_ood:
    st.warning(f"Softmax max ({conf_top:.3f}) < OOD threshold ({ood_threshold:.3f}) — sample may be Out-Of-Distribution / unknown class")

# Grad-CAM (if available)
if grad_model is not None:
    try:
        st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)
        cam, _ = run_gradcam(grad_model, x_in)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.imshow(fixed.T, origin="lower", aspect="auto", cmap='ocean', alpha=0.2)
        extent = [0, fixed.shape[0], 0, fixed.shape[1]]
        ax.imshow(np.tile(cam, (fixed.shape[1], 1)), origin="lower", aspect="auto", alpha=0.8, cmap='inferno', extent=extent)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# Footer
st.markdown(
    """
    <div style="position:fixed; bottom:0; left:0; width:100%; background:#030508; padding:5px 0; text-align:center; color:#666;">
        &copy; 2025 Rights Reserved by BioQuantum Labs
    </div>
    """,
    unsafe_allow_html=True
)
