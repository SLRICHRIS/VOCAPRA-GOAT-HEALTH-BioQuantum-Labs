#!/usr/bin/env python3
"""
VOCAPRA Audio Explorer
----------------------

Streamlit web app for your VOCAPRA pipeline:

- Upload a WAV file
- Run the trained Conv1D model (best_model.h5)
- See top-k class probabilities
- Visualize MFCC features and a Grad-CAM-like saliency map

Designed for deployment on Streamlit Cloud.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Model

# =========================
# CONFIG (must match training)
# =========================

SR = 16000
N_MFCC = 13
WIN_LEN = 0.025   # seconds
HOP_LEN = 0.010   # seconds
TARGET_FRAMES = 80

MODEL_DIR = Path("vocapra_project")   # change to Path(".") if you move files to root
MODEL_PATH = MODEL_DIR / "best_model.h5"
LABEL_MAP_PATH = MODEL_DIR / "label_to_idx.json"


# =========================
# Audio / Feature utilities
# =========================

def load_audio_from_uploaded(file) -> Tuple[np.ndarray, int]:
    """
    Load audio from a Streamlit UploadedFile.
    Returns mono float32 waveform and sampling rate.
    """
    data = file.read()
    # use soundfile to read from bytes
    y, sr = sf.read(io.BytesIO(data), always_2d=False)

    # Convert to mono if stereo
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    # Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    return y, sr


def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """
    Compute MFCC + Δ + Δ² features for waveform y.
    Returns (T, 3*n_mfcc).
    """
    n_fft = int(win_len * sr)
    hop_length = int(hop_len * sr)

    mf = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # (n_mfcc, T)

    mf = mf.T  # (T, n_mfcc)
    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T

    feats = np.concatenate([mf, d1, d2], axis=1)
    return feats.astype(np.float32)


def pad_or_truncate(feats: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Pad or truncate a (T, F) feature sequence to (target_frames, F).
    Uses right alignment (same as your training script).
    """
    T, F = feats.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = feats[:target_frames]
    else:
        out[-T:, :] = feats
    return out


# =========================
# Model loading & Grad-CAM
# =========================

@st.cache_resource(show_spinner=True)
def load_model_and_metadata() -> Tuple[Model, Model, List[str]]:
    """
    Load Keras model and build a Grad-CAM helper model.
    Returns:
        model      : full classifier
        grad_model : model that outputs conv activations + predictions
        idx_to_lbl : list mapping class index -> label string
    """
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    if not LABEL_MAP_PATH.exists():
        st.error(f"Label map JSON not found at: {LABEL_MAP_PATH}")
        st.stop()

    # Load label map
    import json
    with LABEL_MAP_PATH.open("r") as f:
        label_to_idx: Dict[str, int] = json.load(f)

    idx_to_lbl = [None] * len(label_to_idx)
    for lbl, idx in label_to_idx.items():
        idx_to_lbl[idx] = lbl

    # Load Keras model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Find last Conv1D layer for Grad-CAM
    conv_layer_name: Optional[str] = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break

    if conv_layer_name is None:
        st.warning("No Conv1D layer found for Grad-CAM. Saliency will be disabled.")
        grad_model = None
    else:
        conv_layer = model.get_layer(conv_layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [conv_layer.output, model.output],
            name="grad_cam_helper",
        )

    return model, grad_model, idx_to_lbl


def run_model_on_feats(model: Model, feats: np.ndarray) -> np.ndarray:
    """
    Run the classifier on a single (T, F) feature tensor.
    Returns probabilities of shape (num_classes,).
    """
    x = feats[np.newaxis, ...]  # (1, T, F)
    probs = model.predict(x, verbose=0)[0]
    return probs


def compute_gradcam(
    grad_model: Model,
    feats: np.ndarray,
) -> np.ndarray:
    """
    Compute a 1D Grad-CAM activation vector aligned with time frames.

    Args:
        grad_model: model that outputs (conv_activations, predictions)
        feats: (T, F) feature matrix

    Returns:
        cam_resized: (T,) saliency in [0, 1]
    """
    x = feats[np.newaxis, ...]  # (1, T, F)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.convert_to_tensor(x))
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)          # (1, T', C)
    weights = tf.reduce_mean(grads, axis=1)        # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, tf.newaxis, :], axis=-1)[0]  # (T',)

    cam = tf.nn.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    # Resize CAM to input feature time length
    T_in = feats.shape[0]
    T_cam = cam.shape[0]
    cam_resized = np.interp(
        np.linspace(0, T_cam - 1, T_in),
        np.arange(T_cam),
        cam,
    )
    return cam_resized


# =========================
# Plotting helpers
# =========================

def plot_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mfcc(feats: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        feats.T,
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature bins (MFCC + Δ + Δ²)")
    ax.set_title("MFCC Feature Map")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def plot_gradcam_overlay(feats: np.ndarray, cam: np.ndarray) -> plt.Figure:
    """
    Overlay Grad-CAM saliency over feature map.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(
        feats.T,
        origin="lower",
        aspect="auto",
    )
    ax.imshow(
        np.tile(cam, (feats.shape[1], 1)),
        origin="lower",
        aspect="auto",
        cmap="jet",
        alpha=0.45,
    )
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature bins")
    ax.set_title("Grad-CAM Time Saliency Overlay")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=ax,
        fraction=0.025,
        pad=0.02,
        label="Saliency",
    )
    fig.tight_layout()
    return fig


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="VOCAPRA Audio Intelligence",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 VOCAPRA Audio Intelligence Dashboard")
st.caption(
    "Elite CRNN-inspired Conv1D model · MFCC + Δ + Δ² features · "
    "Keras inference + Grad-CAM saliency."
)

with st.sidebar:
    st.header("Pipeline Status")
    st.write(f"**Sampling rate:** {SR} Hz")
    st.write(f"**MFCCs:** {N_MFCC} (with Δ and Δ² → {3 * N_MFCC} dims)")
    st.write(f"**Target frames:** {TARGET_FRAMES}")
    st.markdown("---")
    st.info(
        "Upload a single `.wav` file. The app will resample to 16 kHz, "
        "compute MFCC(+Δ,+Δ²), run the trained model, and render Grad-CAM."
    )

# Load model & metadata once
model, grad_model, idx_to_lbl = load_model_and_metadata()
num_classes = len(idx_to_lbl)

uploaded = st.file_uploader(
    "Upload a WAV file for analysis",
    type=["wav"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.warning("Waiting for an audio file…")
    st.stop()

# ---------------------------------------------------------
# 1) Raw audio + waveform
# ---------------------------------------------------------

st.subheader("1. Raw Audio")

col_a, col_b = st.columns([2, 3])

with col_a:
    st.audio(uploaded, format="audio/wav")

with col_b:
    y, sr = load_audio_from_uploaded(uploaded)
    fig_wav = plot_waveform(y, sr)
    st.pyplot(fig_wav, use_container_width=True)

# ---------------------------------------------------------
# 2) Feature extraction
# ---------------------------------------------------------

st.subheader("2. MFCC Feature Extraction")

feats_seq = compute_mfcc_with_deltas(y, sr)         # (T, F)
feats_fixed = pad_or_truncate(feats_seq, TARGET_FRAMES)  # (TARGET_FRAMES, F)

st.write(f"**Original frames:** {feats_seq.shape[0]} → "
         f"**Padded/trimmed to:** {feats_fixed.shape[0]} frames")
st.write(f"**Feature dimension:** {feats_fixed.shape[1]}")

fig_mfcc = plot_mfcc(feats_fixed)
st.pyplot(fig_mfcc, use_container_width=True)

# ---------------------------------------------------------
# 3) Model prediction
# ---------------------------------------------------------

st.subheader("3. Model Prediction")

probs = run_model_on_feats(model, feats_fixed)  # (num_classes,)

top_k = min(5, num_classes)
top_indices = np.argsort(probs)[::-1][:top_k]
top_labels = [idx_to_lbl[i] for i in top_indices]
top_probs = probs[top_indices]

df_pred = pd.DataFrame(
    {
        "Rank": np.arange(1, top_k + 1),
        "Class": top_labels,
        "Probability": top_probs,
    }
)
df_pred["Probability"] = df_pred["Probability"].map(lambda x: f"{x:.3f}")

st.table(df_pred)

pred_idx = int(np.argmax(probs))
pred_label = idx_to_lbl[pred_idx]
st.success(f"**Predicted class:** `{pred_label}` (index {pred_idx})")

# ---------------------------------------------------------
# 4) Grad-CAM saliency
# ---------------------------------------------------------

st.subheader("4. Grad-CAM Time Saliency")

if grad_model is None:
    st.warning("Grad-CAM is disabled because no Conv1D layer was found.")
else:
    cam = compute_gradcam(grad_model, feats_fixed)   # (T,)

    with st.expander("Show Grad-CAM raw vector", expanded=False):
        st.write(cam)

    fig_cam = plot_gradcam_overlay(feats_fixed, cam)
    st.pyplot(fig_cam, use_container_width=True)

st.markdown("---")
st.caption(
    "Backend: TF 2.x Conv1D classifier · INT8-ready architecture · "
    "Feature pipeline synchronized with training script."
)
