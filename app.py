#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI

World-class UI for:
- Uploading a WAV file
- Running VOCAPRA MFCC (+Δ +Δ²) pipeline
- Predicting with a tiny Conv1D model (best_model*.h5)
- Visualising class probabilities
- Viewing Grad-CAM-like saliency over the feature map

Expected artifacts (any of these patterns is accepted):
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

# =============================================================================
# CONFIG (must match training pipeline)
# =============================================================================
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80

ARTIFACT_DIR = Path("vocapra_project")

# =============================================================================
# SMALL UTILS
# =============================================================================


def resolve_artifact(pattern: str) -> Optional[Path]:
    """
    Return the first file in ARTIFACT_DIR that matches `pattern`,
    e.g. "best_model*.h5" or "label_to_idx*.json".
    """
    if not ARTIFACT_DIR.exists():
        return None
    matches: List[Path] = sorted(ARTIFACT_DIR.glob(pattern))
    return matches[0] if matches else None


# =============================================================================
# FEATURE PIPELINE
# =============================================================================
def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """
    Compute MFCC + Δ + Δ² features.

    Returns:
        feats: np.ndarray of shape (T, 3 * n_mfcc)
    """
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
    """
    Right-pad / truncate a (T, F) sequence to shape (target_frames, F),
    exactly like the training script.
    """
    T, F = seq.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = seq[:target_frames]
    else:
        out[-T:, :] = seq
    return out


# =============================================================================
# MODEL LOADING + GRAD-CAM
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_gradcam() -> Tuple[Optional[tf.keras.Model], Optional[tf.keras.Model], Optional[str], Optional[Path]]:
    """
    Load Keras model from best_model*.h5 and construct a Grad-CAM model
    using the last Conv1D layer.
    """
    model_path = resolve_artifact("best_model*.h5")
    if model_path is None:
        return None, None, None, None

    model = tf.keras.models.load_model(model_path)

    # Find the last Conv1D layer for Grad-CAM
    conv_layer_name: Optional[str] = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break

    grad_model: Optional[tf.keras.Model] = None
    if conv_layer_name is not None:
        conv_layer = model.get_layer(conv_layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )

    return model, grad_model, conv_layer_name, model_path


@st.cache_resource(show_spinner=False)
def load_label_map() -> Tuple[Dict[int, str], Dict[str, int], Optional[Path]]:
    """
    Load label_to_idx*.json and return both idx_to_label and label_to_idx.
    """
    json_path = resolve_artifact("label_to_idx*.json")
    if json_path is None:
        return {}, {}, None
    with open(json_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return idx_to_label, label_to_idx, json_path


def run_gradcam(
    grad_model: tf.keras.Model,
    sample: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM along the time axis for a single input sample
    of shape (1, T, F).

    Returns:
        cam_resized: np.ndarray of shape (T,)
        class_idx:   int, class index used for CAM
    """
    sample_tf = tf.convert_to_tensor(sample)

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(sample_tf)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outs)           # (1, T', C)
    weights = tf.reduce_mean(grads, axis=1)         # (1, C)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
    cam = tf.nn.relu(cam).numpy()[0]                # (T',)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    T_in = sample.shape[1]
    T_cam = cam.shape[0]
    cam_resized = np.interp(
        np.linspace(0, T_cam - 1, T_in), np.arange(T_cam), cam
    )
    return cam_resized, int(class_idx.numpy())


# =============================================================================
# STREAMLIT – WORLD-CLASS UI
# =============================================================================
st.set_page_config(
    page_title="VOCAPRA Audio Event Explorer",
    page_icon="🎧",
    layout="wide",
)

# ---- Custom CSS for product-grade feel ----
st.markdown(
    """
    <style>
    /* Global look */
    html, body, [class*="css"]  {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif !important;
    }

    .main {
        background: radial-gradient(circle at top left, #171821 0, #020308 40%, #020308 100%);
        color: #f4f4f4;
    }

    /* Center content and give it breathing room */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
        margin: 0 auto;
    }

    /* Header */
    .app-header {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #2c3e50 0%, #111827 60%, #0f766e 100%);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        color: #ecfdf5;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .app-header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }
    .app-header-subtitle {
        opacity: 0.82;
        font-size: 0.95rem;
    }
    .tag-pill {
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        background: rgba(15, 23, 42, 0.75);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    /* Cards */
    .card {
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }

    .sidebar-card {
        border-radius: 16px;
        padding: 0.9rem 0.75rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.25);
        font-size: 0.84rem;
    }

    .metric-main {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    .prob-badge {
        display: inline-flex;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(34,197,94,0.12);
        color: #bbf7d0;
        border: 1px solid rgba(52,211,153,0.35);
        margin-left: 0.4rem;
    }

    /* Waveform / Grad-CAM */
    .plot-card {
        border-radius: 18px;
        padding: 1.0rem 1.0rem 0.6rem 1.0rem;
        background: rgba(15,23,42,0.94);
        border: 1px solid rgba(148,163,184,0.3);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
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

# Sidebar – pipeline, artifacts, meta
with st.sidebar:
    st.markdown("### Pipeline")
    st.markdown(
        """
        <div class="sidebar-card">
        <b>Model</b><br/>
        • Conv1D → BN → MaxPool ×2<br/>
        • GlobalAveragePooling1D<br/>
        • Dense softmax head
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="height: 0.5rem;"></div>
        <div class="sidebar-card">
        <b>Features</b><br/>
        • 16 kHz mono<br/>
        • MFCC (13) + Δ + Δ² → 39 dims<br/>
        • 25 ms window, 10 ms hop<br/>
        • Fixed 80 frames via right padding
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="height: 0.5rem;"></div>
        <div class="sidebar-card">
        <b>Artifacts</b><br/>
        Model: <code>vocapra_project/best_model*.h5</code><br/>
        Labels: <code>vocapra_project/label_to_idx*.json</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Guard rails if artifacts missing
if model is None or not idx_to_label:
    st.markdown(
        """
        <div class="app-header">
          <div>
            <div class="app-header-title">🎧 VOCAPRA Audio Event Explorer</div>
            <div class="app-header-subtitle">
              Tiny Conv1D model for goat vocalisation events – deployment console.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        "Model or label map not found.\n\n"
        "Please ensure the following exist in your repository:\n\n"
        "• `vocapra_project/best_model*.h5`\n"
        "• `vocapra_project/label_to_idx*.json`"
    )
    st.stop()

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="app-header-title">🎧 VOCAPRA Audio Event Explorer</div>
            <div class="app-header-subtitle">
                Upload a goat vocalisation clip, let the tiny Conv1D model decode the context,
                and inspect which temporal regions drove the decision.
            </div>
        </div>
        <div>
            <div class="tag-pill">v1 · Tiny ConvNet · TFLite-Ready</div>
            <div style="height: 0.4rem;"></div>
            <div class="tag-pill">Sampling: 16 kHz · Frames: 80 · Features: 39D</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")  # spacing

# =============================================================================
# LAYOUT: UPLOAD + CLASS PANEL
# =============================================================================
top_left, top_right = st.columns([1.8, 1.3])

with top_left:
    st.markdown("#### 1 · Ingest audio")
    st.markdown(
        """
        <div class="card">
        <strong>Drop in a 16 kHz mono WAV file</strong><br/>
        If your file uses a different sample rate or is stereo, the app will
        transparently resample and fold it down to mono.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        " ",
        type=["wav"],
        label_visibility="collapsed",
        help="WAV only – internally resampled to 16 kHz mono.",
    )

with top_right:
    st.markdown("#### 2 · Model snapshot")
    with st.container():
        st.markdown(
            """
            <div class="card">
              <b>Classes</b><br/>
            """,
            unsafe_allow_html=True,
        )
        st.write(", ".join(idx_to_label[i] for i in sorted(idx_to_label.keys())))
        model_name_txt = model_path.name if model_path is not None else "best_model*.h5"
        json_name_txt = label_json_path.name if label_json_path is not None else "label_to_idx*.json"
        st.markdown(
            f"""
            <small>Loaded from <code>vocapra_project/{model_name_txt}</code><br/>
            Labels from <code>vocapra_project/{json_name_txt}</code></small>
            </div>
            """,
            unsafe_allow_html=True,
        )

if uploaded is None:
    st.info("👆 Upload a WAV file above to start the analysis.")
    st.stop()

# =============================================================================
# LOAD AUDIO
# =============================================================================
try:
    y, sr = librosa.load(uploaded, sr=SR, mono=True)
except Exception:
    uploaded.seek(0)
    data, sr_raw = sf.read(uploaded)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    y = librosa.resample(data, orig_sr=sr_raw, target_sr=SR)
    sr = SR

duration = len(y) / sr

audio_row_left, audio_row_right = st.columns([1.7, 1.3])

with audio_row_left:
    st.markdown("#### 3 · Signal preview")
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    t = np.linspace(0, duration, num=len(y))
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(t, y, linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.18)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with audio_row_right:
    st.markdown("#### 4 · Normalisation status")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**Duration:** {duration:.2f} s")
        st.write(f"**Sample rate (internal):** {sr} Hz")
        st.write(f"**Samples:** {len(y):,}")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FEATURE EXTRACTION + PREDICTION
# =============================================================================
feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0)

probs = model.predict(x_in, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_label = idx_to_label.get(pred_idx, str(pred_idx))

st.markdown("#### 5 · Model decision")

pred_cols = st.columns([1.4, 2.0])

with pred_cols[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <span style="font-size: 0.8rem; text-transform: uppercase; opacity: 0.75;">
        Primary hypothesis
        </span><br/>
        <span class="metric-main">{pred_label}</span>
        <span class="prob-badge">{probs[pred_idx]*100:.1f}% confidence</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with pred_cols[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**Probability distribution (top classes)**")

    sorted_indices = np.argsort(probs)[::-1]
    top_k = min(8, len(sorted_indices))
    top_indices = sorted_indices[:top_k]
    top_labels = [idx_to_label[i] for i in top_indices]
    top_values = probs[top_indices]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(range(top_k), top_values[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_labels[::-1])
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1.0)
    ax.set_title("Top-k class probabilities")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "<small>Raw probabilities:</small>",
        unsafe_allow_html=True,
    )
    st.json({idx_to_label[i]: float(probs[i]) for i in range(len(probs))})
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# GRAD-CAM SECTION
# =============================================================================
st.markdown("#### 6 · Model introspection (Grad-CAM)")

if grad_model is None:
    st.warning(
        "Grad-CAM disabled: no Conv1D layer detected in the model architecture."
    )
else:
    cam, cam_class_idx = run_gradcam(grad_model, x_in)
    cam_class_label = idx_to_label.get(cam_class_idx, str(cam_class_idx))

    gc_left, gc_right = st.columns([2.1, 1.1])

    with gc_left:
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.caption(
            f"Grad-CAM heatmap for class **{cam_class_label}** "
            f"(index {cam_class_idx}). Colours highlight time regions "
            "that were most influential for this decision."
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
        ax.set_ylabel("Feature bins (MFCC + Δ + Δ²)")
        ax.set_title("Grad-CAM overlay on feature map")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with gc_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**How to read this**")
        st.markdown(
            """
            • The horizontal axis is time (frame index).  
            • Hot colours (yellow / red) = frames that strongly supported the prediction.  
            • Cooler colours = frames with little impact.  
            • This is particularly useful for aligning events with husbandry logs
              or video footage.
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "VOCAPRA · Tiny Conv1D audio event model • Designed for TFLite and edge deployment."
)
