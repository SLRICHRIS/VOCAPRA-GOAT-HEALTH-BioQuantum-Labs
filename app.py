#!/usr/bin/env python3
"""
VOCAPRA Streamlit App

Elite UI for:
- Uploading a WAV file
- Running VOCAPRA MFCC pipeline
- Predicting with best_model.h5 (Conv1D tiny CRNN-like model)
- Showing class probabilities
- Displaying Grad-CAM-like saliency over the feature map

Artifacts expected under vocapra_project/ :

  - best_model.h5           OR  best_model (1).h5
  - label_to_idx.json       OR  label_to_idx (1).json
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

# ----------------- CONFIG (must match training) ----------------- #
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80

ARTIFACT_DIR = Path("vocapra_project")


def _find_first(patterns: List[str]) -> Optional[Path]:
    """Return first existing file in ARTIFACT_DIR matching any glob pattern."""
    for pat in patterns:
        for p in ARTIFACT_DIR.glob(pat):
            if p.is_file():
                return p
    return None


# Accept both clean names and “(1)” variants
MODEL_PATH = _find_first(["best_model.h5", "best_model*.h5"])
LABEL_MAP_PATH = _find_first(["label_to_idx.json", "label_to_idx*.json"])

# ----------------- FEATURE PIPELINE ----------------- #
def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """MFCC + Δ + Δ², exactly as in training."""
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
    """Right-pad / truncate a (T, F) feature seq to (target_frames, F)."""
    T, F = seq.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = seq[:target_frames]
    else:
        out[-T:, :] = seq
    return out


# ----------------- MODEL LOADING + GRADCAM ----------------- #
@st.cache_resource(show_spinner=False)
def load_model_and_gradcam() -> Tuple[Optional[tf.keras.Model], Optional[tf.keras.Model], Optional[str]]:
    """Load Keras model and build grad_model (for last Conv1D layer)."""

    if MODEL_PATH is None or not MODEL_PATH.exists():
        return None, None, None

    model = tf.keras.models.load_model(MODEL_PATH)

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


@st.cache_resource(show_spinner=False)
def load_label_map() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load label_to_idx json and also return idx_to_label."""
    if LABEL_MAP_PATH is None or not LABEL_MAP_PATH.exists():
        return {}, {}
    with open(LABEL_MAP_PATH, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return idx_to_label, label_to_idx


def run_gradcam(
    grad_model: tf.keras.Model,
    sample: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM along time axis for a single sample (1, T, F).
    Returns CAM (T,) and the class index used.
    """
    sample_tf = tf.convert_to_tensor(sample)  # (1, T, F)

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(sample_tf)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outs)          # (1, T', C)
    weights = tf.reduce_mean(grads, axis=1)         # (1, C)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
    cam = tf.nn.relu(cam).numpy()[0]               # (T',)

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    # Resize CAM to input time axis T
    T_in = sample.shape[1]
    T_cam = cam.shape[0]
    cam_resized = np.interp(
        np.linspace(0, T_cam - 1, T_in),
        np.arange(T_cam),
        cam,
    )
    return cam_resized, int(class_idx.numpy())


# ----------------- STREAMLIT UI ----------------- #
st.set_page_config(
    page_title="VOCAPRA Audio Event Explorer",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-pred {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎧 VOCAPRA Audio Event Explorer")
st.caption(
    "Upload a WAV file, see the tiny Conv1D model's prediction, "
    "and explore which time regions mattered via a Grad-CAM heatmap."
)

# Sidebar info
with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        """
        **Model**  
        • Conv1D → BN → MaxPool (x2)  
        • GlobalAveragePooling → Dense softmax  

        **Features**  
        • MFCC (13) + Δ + Δ² = 39 dims  
        • 16 kHz mono, 25 ms window, 10 ms hop  
        • Fixed **80 frames** via right padding.
        """
    )
    st.markdown("---")
    st.info(
        "Artifacts are loaded from:\n"
        "`vocapra_project/best_model*.h5`\n"
        "`vocapra_project/label_to_idx*.json`"
    )

idx_to_label, label_to_idx = load_label_map()
model, grad_model, conv_name = load_model_and_gradcam()

if model is None or not idx_to_label:
    st.error(
        "Model or label map not found.\n\n"
        "Make sure these exist in the repo:\n\n"
        "• `vocapra_project/best_model.h5` **or** `best_model (1).h5`\n"
        "• `vocapra_project/label_to_idx.json` **or** `label_to_idx (1).json`"
    )
    st.stop()

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a 16 kHz mono WAV file",
        type=["wav"],
        help="It will be resampled to 16 kHz and converted to mono if needed.",
    )

with col_info:
    st.subheader("Classes")
    st.write(", ".join(idx_to_label[i] for i in sorted(idx_to_label.keys())))

if uploaded is None:
    st.info("👆 Upload a WAV file to start.")
    st.stop()

# ----------------- Load audio ----------------- #
try:
    # Using librosa directly from file-like object
    y, sr = librosa.load(uploaded, sr=SR, mono=True)
except Exception:
    # Fallback via soundfile
    uploaded.seek(0)
    data, sr_raw = sf.read(uploaded)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    y = librosa.resample(data, orig_sr=sr_raw, target_sr=SR)
    sr = SR

duration = len(y) / sr

st.success(f"Loaded audio: {duration:.2f} seconds @ {sr} Hz")

with st.expander("🔊 Waveform preview", expanded=False):
    t = np.linspace(0, duration, num=len(y))
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    st.pyplot(fig)
    plt.close(fig)

# ----------------- Feature extraction + prediction ----------------- #
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0)  # (1, T, F)

probs = model.predict(x_in, verbose=0)[0]  # (C,)
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, str(pred_idx))

st.markdown(
    f"<div class='big-pred'>Predicted event: {pred_label}</div>",
    unsafe_allow_html=True,
)

# ----------------- Probability chart ----------------- #
st.subheader("Class probabilities")

sorted_indices = np.argsort(probs)[::-1]
top_k = min(8, len(sorted_indices))
top_indices = sorted_indices[:top_k]
top_labels = [idx_to_label[i] for i in top_indices]
top_values = probs[top_indices]

prob_cols = st.columns([3, 2])
with prob_cols[0]:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(range(top_k), top_values[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_labels[::-1])
    ax.set_xlabel("Probability")
    ax.set_title("Top class probabilities")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with prob_cols[1]:
    st.metric("Top class", pred_label, f"{probs[pred_idx]*100:.1f}%")
    st.write("Full distribution:")
    st.json({idx_to_label[i]: float(probs[i]) for i in range(len(probs))})

# ----------------- Grad-CAM ----------------- #
st.subheader("Grad-CAM: time segments that mattered")

if grad_model is None:
    st.warning("Grad-CAM disabled: no Conv1D layer detected in the model.")
else:
    cam, cam_class_idx = run_gradcam(grad_model, x_in)
    cam_class_label = idx_to_label.get(cam_class_idx, str(cam_class_idx))

    st.caption(
        f"Grad-CAM computed for predicted class: **{cam_class_label}** "
        f"(index {cam_class_idx})"
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(fixed.T, origin="lower", aspect="auto")
    ax.imshow(
        np.tile(cam, (fixed.shape[1], 1)),
        origin="lower",
        aspect="auto",
        alpha=0.45,
        cmap="jet",
    )
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature bins (MFCC+Δ+Δ²)")
    ax.set_title("Grad-CAM overlay on feature map")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.caption("Built with ❤️ on a tiny Conv1D VOCAPRA pipeline, ready for TFLite deployment.")
