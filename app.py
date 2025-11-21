#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v2 (Glassmorphism Edition)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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
# UTILS & PIPELINE (Logic remains the same)
# =============================================================================
def resolve_artifact(pattern: str) -> Optional[Path]:
    if not ARTIFACT_DIR.exists():
        return None
    matches: List[Path] = sorted(ARTIFACT_DIR.glob(pattern))
    return matches[0] if matches else None

def compute_mfcc_with_deltas(y: np.ndarray, sr: int = SR) -> np.ndarray:
    n_fft = int(WIN_LEN * sr)
    hop_length = int(HOP_LEN * sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length)
    mf = mf.T
    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T
    feats = np.concatenate([mf, d1, d2], axis=1).astype(np.float32)
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
    model = tf.keras.models.load_model(model_path)
    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break
    grad_model = None
    if conv_layer_name is not None:
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])
    return model, grad_model, conv_layer_name, model_path

@st.cache_resource(show_spinner=False)
def load_label_map():
    json_path = resolve_artifact("label_to_idx*.json")
    if json_path is None:
        return {}, {}, None
    with open(json_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return idx_to_label, label_to_idx, json_path

def run_gradcam(grad_model, sample):
    sample_tf = tf.convert_to_tensor(sample)
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(sample_tf)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outs)
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    T_in = sample.shape[1]
    T_cam = cam.shape[0]
    cam_resized = np.interp(np.linspace(0, T_cam - 1, T_in), np.arange(T_cam), cam)
    return cam_resized, int(class_idx.numpy())

# =============================================================================
# HELPER: STYLISH PLOTTING
# =============================================================================
def style_axis(ax):
    """Applies a clean, dark-mode style to matplotlib axes."""
    ax.set_facecolor("none") # Transparent
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#cccccc')
    ax.tick_params(axis='y', colors='#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="VOCAPRA Explorer",
    page_icon="🐐",
    layout="wide",
)

# ---- UPDATED CSS FOR GLASSMORPHISM & VIBRANCY ----
st.markdown(
    """
    <style>
    /* Background */
    .main {
        background: linear-gradient(145deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Typography */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.02em; }
    
    /* Header Gradient Text */
    .gradient-text {
        background: linear-gradient(to right, #2dd4bf, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(45, 212, 191, 0.3);
    }

    /* Metric Styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f9ff;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
    }

    /* Pills */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(45, 212, 191, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
    }
    
    /* File Uploader Customization */
    .stFileUploader {
        padding: 1rem;
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
idx_to_label, label_to_idx, label_json_path = load_label_map()
model, grad_model, conv_name, model_path = load_model_and_gradcam()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("### ⚙️ System Config")
    st.info(f"Model: {model_path.name if model_path else 'Not Found'}")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.caption("• Input: (80, 39)\n• Conv1D Stack\n• Global Avg Pool")
    st.markdown("---")
    st.markdown("**Audio Spec**")
    st.caption(f"• Rate: {SR} Hz\n• Window: {WIN_LEN*1000:.0f}ms")

if model is None or not idx_to_label:
    st.error("⚠️ Artifacts missing. Please check `vocapra_project/` folder.")
    st.stop()

# =============================================================================
# MAIN UI
# =============================================================================

# Header
st.markdown('<div class="gradient-text">VOCAPRA Explorer</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: -10px; color: #94a3b8;">Advanced Acoustic Event Detection & Interpretability Console</div>', unsafe_allow_html=True)
st.write("") 

# Layout: Upload & Stats
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">1. AUDIO INPUT</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop WAV file", type=["wav"], label_visibility="collapsed")
    st.caption("Supports automatic resampling to 16kHz mono.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{len(idx_to_label)} <span style='font-size:1rem; color:#64748b'>classes</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status-pill'>READY TO INFERENCE</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    st.stop()

# =============================================================================
# PROCESSING & PLOTTING
# =============================================================================

# Load Audio
try:
    y, sr = librosa.load(uploaded, sr=SR, mono=True)
except Exception:
    uploaded.seek(0)
    data, sr_raw = sf.read(uploaded)
    if data.ndim == 2: data = np.mean(data, axis=1)
    y = librosa.resample(data, orig_sr=sr_raw, target_sr=SR)
    sr = SR

# Feature Extraction
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0)

# Prediction
probs = model.predict(x_in, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, str(pred_idx))
confidence = probs[pred_idx]

# --- RESULTS SECTION ---
st.markdown("### Analysis Results")

# 1. Prediction Banner
st.markdown(
    f"""
    <div class="glass-card" style="border-left: 6px solid #2dd4bf;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="metric-label">DETECTED EVENT</div>
                <div class="metric-value">{pred_label}</div>
            </div>
            <div style="text-align:right;">
                <div class="metric-label">CONFIDENCE</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Plots (Waveform + Probabilities)
c_wave, c_probs = st.columns([1.8, 1.2])

with c_wave:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">SIGNAL WAVEFORM</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 2.5))
    # Make background transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # Plot line with cyan color
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#2dd4bf', linewidth=0.8, alpha=0.9)
    style_axis(ax)
    ax.set_xlabel("Time (s)")
    ax.grid(color='white', alpha=0.05)
    
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with c_probs:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
    
    sorted_indices = np.argsort(probs)[::-1][:5]
    top_labels = [idx_to_label[i] for i in sorted_indices]
    top_vals = probs[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # Horizontal bars with gradient-like colors
    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color='#38bdf8', alpha=0.8)
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_labels[::-1], color='#e2e8f0', fontsize=9)
    style_axis(ax)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Grad-CAM
st.markdown("### Model Introspection")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(f'<div class="metric-label">TEMPORAL ACTIVATION MAP (Grad-CAM) FOR: <span style="color:#2dd4bf">{pred_label}</span></div>', unsafe_allow_html=True)

if grad_model:
    cam, _ = run_gradcam(grad_model, x_in)
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # Background: The Features (MFCC)
    # Using 'gray' cmap for features to make the CAM pop
    ax.imshow(fixed.T, origin="lower", aspect="auto", cmap='gray', alpha=0.3)
    
    # Overlay: The Grad-CAM heatmap
    # Using 'magma' or 'inferno' for that "Cyber" look (better than jet)
    extent = [0, fixed.shape[0], 0, fixed.shape[1]]
    im = ax.imshow(
        np.tile(cam, (fixed.shape[1], 1)),
        origin="lower", 
        aspect="auto", 
        alpha=0.65, 
        cmap='magma', 
        extent=extent
    )
    
    style_axis(ax)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Coefficients")
    
    # Add a colorbar that fits the theme
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')
    cbar.outline.set_edgecolor('#444444')
    
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Grad-CAM unavailable (Model architecture incompatible).")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #475569; font-size: 0.8rem; margin-top: 2rem;">VOCAPRA AI · 2025</div>', unsafe_allow_html=True)
