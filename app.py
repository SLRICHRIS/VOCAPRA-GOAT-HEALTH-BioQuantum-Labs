#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite Edition

AI for Speech / Goat Vocalisation:
- Upload 16 kHz mono WAV
- MFCC (+Δ +Δ²) features
- Tiny Conv1D model (best_model.h5)
- Class probabilities
- Grad-CAM-like time saliency

Artifacts expected in:
  vocapra_project/best_model*.h5
  vocapra_project/label_to_idx*.json
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

# ----------------- PIPELINE CONFIG (match training) ----------------- #
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80

ARTIFACT_DIR = Path("vocapra_project")


# ----------------- UTILS: ARTIFACT RESOLUTION ----------------- #
def _find_first_matching(patterns: List[str]) -> Optional[Path]:
    """Return first existing path in ARTIFACT_DIR matching any of patterns."""
    for pat in patterns:
        for p in ARTIFACT_DIR.glob(pat):
            if p.is_file():
                return p
    return None


def resolve_model_path() -> Optional[Path]:
    """
    Handle typical download names like:
      best_model.h5
      best_model (1).h5
    """
    return _find_first_matching(["best_model*.h5"])


def resolve_label_map_path() -> Optional[Path]:
    return _find_first_matching(["label_to_idx*.json"])


# ----------------- FEATURE PIPELINE ----------------- #
def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """MFCC + Δ + Δ² as used in training."""
    n_fft = int(win_len * sr)
    hop_length = int(hop_len * sr)

    mf = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )  # (n_mfcc, T)
    mf = mf.T  # (T, n_mfcc)
    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T
    feats = np.concatenate([mf, d1, d2], axis=1).astype(np.float32)
    return feats


def to_fixed_frames(seq: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Right-pad or truncate (T,F) -> (target_frames, F)."""
    T, F = seq.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = seq[:target_frames]
    else:
        out[-T:, :] = seq
    return out


# ----------------- MODEL + LABEL MAP LOADING ----------------- #
@st.cache_resource(show_spinner=True)
def load_model_and_gradcam() -> Tuple[Optional[tf.keras.Model], Optional[tf.keras.Model], Optional[str]]:
    model_path = resolve_model_path()
    if model_path is None:
        return None, None, None

    model = tf.keras.models.load_model(model_path)

    # Find last Conv1D layer for Grad-CAM
    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break

    grad_model = None
    if conv_layer_name is not None:
        conv_layer = model.get_layer(conv_layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )

    return model, grad_model, conv_layer_name


@st.cache_resource(show_spinner=True)
def load_label_map() -> Tuple[Dict[int, str], Dict[str, int]]:
    label_path = resolve_label_map_path()
    if label_path is None:
        return {}, {}

    with open(label_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return idx_to_label, label_to_idx


def run_gradcam(
    grad_model: tf.keras.Model,
    sample: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM along time axis for sample of shape (1, T, F).
    Returns:
      cam_resized: (T,) normalized [0,1]
      class_idx:   int
    """
    x = tf.convert_to_tensor(sample)

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(x)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outs)          # (1, T', C)
    weights = tf.reduce_mean(grads, axis=1)         # (1, C)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
    cam = tf.nn.relu(cam).numpy()[0]               # (T',)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    T_in = sample.shape[1]
    T_cam = cam.shape[0]
    cam_resized = np.interp(
        np.linspace(0, T_cam - 1, T_in),
        np.arange(T_cam),
        cam,
    )
    return cam_resized, int(class_idx.numpy())


# ----------------- STREAMLIT PAGE CONFIG + STYLING ----------------- #
st.set_page_config(
    page_title="VOCAPRA Audio Event Explorer",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Full-page subtle background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    /* Hero banner */
    .vocapra-hero {
        border-radius: 1.5rem;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(236,72,153,0.08));
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 50px rgba(15,23,42,0.7);
    }
    .vocapra-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }
    .vocapra-sub {
        font-size: 0.98rem;
        color: #e5e7eb;
        opacity: 0.86;
    }
    .vocapra-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        padding: 0.18rem 0.62rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.6);
        color: #cbd5f5;
    }
    .metric-card {
        background: rgba(15,23,42,0.72);
        border-radius: 1rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(51,65,85,0.8);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- HERO ----------------- #
st.markdown(
    """
    <div class="vocapra-hero">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
        <div>
          <div class="vocapra-title">🎧 VOCAPRA Audio Event Explorer</div>
          <div class="vocapra-sub">
            Tiny Conv1D model for goat vocalisation events – MFCC (+Δ +Δ²), TFLite-ready,
            with Grad-CAM time saliency. Upload WAV, see predictions, explore what the model attends to.
          </div>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.4rem; min-width:210px;">
          <div class="vocapra-pill">16 kHz • MFCC x 39 • 80 frames</div>
          <div class="vocapra-pill">Conv1D → GlobalAvgPool → Dense Softmax</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR ----------------- #
with st.sidebar:
    st.subheader("Pipeline snapshot")
    st.markdown(
        """
        **Front-end**  
        • Streamlit, Matplotlib, advanced layout  

        **Acoustics**  
        • 16 kHz mono  
        • 25 ms window, 10 ms hop  
        • MFCC (13) + Δ + Δ² → 39-D vectors  
        • Fixed 80 frames per clip  

        **Model**  
        • Conv1D(32) → BN → MaxPool  
        • Conv1D(48) → BN → MaxPool  
        • GlobalAveragePooling1D  
        • Dense softmax (multi-class)  
        """
    )
    st.markdown("---")
    st.caption(
        "Artifacts are resolved from:\n"
        f"`{ARTIFACT_DIR}/best_model*.h5`\n"
        f"`{ARTIFACT_DIR}/label_to_idx*.json`"
    )

# ----------------- LOAD ARTIFACTS ----------------- #
idx_to_label, label_to_idx = load_label_map()
model, grad_model, conv_name = load_model_and_gradcam()

if model is None or not idx_to_label:
    st.error(
        "Model or label map not found.\n\n"
        "Please ensure the following files exist in the repo:\n"
        "- `vocapra_project/best_model.h5` (or `best_model (1).h5`)\n"
        "- `vocapra_project/label_to_idx.json` (or `label_to_idx (1).json`)"
    )
    st.stop()

# ----------------- MAIN LAYOUT ----------------- #
left, right = st.columns([2.2, 1.1])

with left:
    st.markdown("#### 1. Upload audio")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse a 16 kHz mono WAV file",
        type=["wav"],
        label_visibility="collapsed",
    )

with right:
    st.markdown("#### 2. Class set")
    st.markdown(
        ", ".join(f"`{idx_to_label[i]}`" for i in sorted(idx_to_label.keys()))
    )

if uploaded_file is None:
    st.info("⬆️ Upload a WAV file to run the pipeline.")
    st.stop()

# Keep raw bytes for playback
raw_bytes = uploaded_file.getvalue()

# ----------------- AUDIO LOADING ----------------- #
try:
    y, sr = librosa.load(uploaded_file, sr=SR, mono=True)
except Exception:
    uploaded_file.seek(0)
    data, sr_raw = sf.read(uploaded_file)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    y = librosa.resample(data, orig_sr=sr_raw, target_sr=SR)
    sr = SR

duration = len(y) / sr

overview_tab, pred_tab, feature_tab = st.tabs(
    ["🔎 Overview", "📊 Predictions & Explanations", "🧬 Raw Features"]
)

# ----------------- TAB: OVERVIEW ----------------- #
with overview_tab:
    c1, c2 = st.columns([2.5, 1.2])
    with c1:
        st.subheader("Waveform")
        t = np.linspace(0, duration, num=len(y))
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(t, y, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        ax.grid(alpha=0.15)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader("Audio")
        st.audio(raw_bytes, format="audio/wav")
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Duration</div>
              <div class="metric-value">{duration:.2f} s</div>
              <div class="metric-sub">{sr} Hz • mono</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- FEATURE EXTRACTION + MODEL PREDICTION ----------------- #
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0)  # (1, T, F)

probs = model.predict(x_in, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, str(pred_idx))
pred_conf = float(probs[pred_idx])

# ----------------- TAB: PREDICTIONS & EXPLANATIONS ----------------- #
with pred_tab:
    st.subheader("Prediction summary")

    mc1, mc2, mc3 = st.columns([1.4, 1.4, 1.4])
    with mc1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Top class</div>
              <div class="metric-value">{pred_label}</div>
              <div class="metric-sub">index {pred_idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Confidence</div>
              <div class="metric-value">{pred_conf*100:.1f}%</div>
              <div class="metric-sub">softmax probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Feature frames</div>
              <div class="metric-value">{TARGET_FRAMES}</div>
              <div class="metric-sub">MFCC+Δ+Δ² time bins</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Class probabilities")

    sorted_idx = np.argsort(probs)[::-1]
    top_k = min(8, len(sorted_idx))
    top_idx = sorted_idx[:top_k]
    top_labels = [idx_to_label[i] for i in top_idx]
    top_vals = probs[top_idx]

    bc1, bc2 = st.columns([2.3, 1.7])
    with bc1:
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.barh(range(top_k), top_vals[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_labels[::-1])
        ax.set_xlabel("Probability")
        ax.set_title("Top class probabilities")
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with bc2:
        st.write("Full distribution:")
        st.json({idx_to_label[i]: float(probs[i]) for i in range(len(probs))})

    st.markdown("---")
    st.subheader("Grad-CAM over time")

    if grad_model is None:
        st.warning("Grad-CAM disabled: no Conv1D layer detected in model.")
    else:
        cam, cam_class_idx = run_gradcam(grad_model, x_in)
        cam_label = idx_to_label.get(cam_class_idx, str(cam_class_idx))

        st.caption(
            f"Grad-CAM computed for class **{cam_label}** (index {cam_class_idx}). "
            "Heat intensity corresponds to time segments that influenced this class."
        )

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.imshow(fixed.T, origin="lower", aspect="auto")
        ax.imshow(
            np.tile(cam, (fixed.shape[1], 1)),
            origin="lower",
            aspect="auto",
            alpha=0.45,
            cmap="jet",
        )
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Feature bins (MFCC + Δ + Δ²)")
        ax.set_title("Grad-CAM overlay on feature map")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ----------------- TAB: RAW FEATURES ----------------- #
with feature_tab:
    st.subheader("Feature tensor")
    st.write(
        f"Fixed feature shape: `{fixed.shape}` (frames × feature dims). "
        "Each frame is a 39-D vector: 13 MFCC + 13 Δ + 13 Δ²."
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(fixed.T, origin="lower", aspect="auto")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature index")
    ax.set_title("MFCC(+Δ+Δ²) feature map")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.caption(
    "VOCAPRA • Tiny Conv1D + MFCC pipeline • Designed for TFLite deployment and explainable goat vocalisation analytics."
)
