#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v1.O (Cyber-HUD Edition) — Full version with Evaluation Engine

Features:
- Robust model & label loading with demo fallback
- MFCC + delta + delta-delta feature extraction
- Fixed-frame padding for model input
- Grad-CAM visualisation (guarded)
- Neon HUD visuals + CSS
- Research-grade evaluation (per-class precision/recall/F1, confusion matrix, PR/ROC optional)
- Test dataset folder evaluation with downloadable CSV report
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

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
    n_fft = int(WIN_LEN * sr)
    hop_length = int(HOP_LEN * sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length)
    mf = mf.T  # shape (T, N_MFCC)
    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T
    feats = np.concatenate([mf, d1, d2], axis=1).astype(np.float32)  # (T, N_MFCC*3)
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
    """
    Returns: (model, grad_model, conv_layer_name, model_path)
    grad_model may be None if Conv1D layer not found or model missing.
    """
    model_path = resolve_artifact("best_model*.h5")
    if model_path is None:
        return None, None, None, None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None

    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D) or "conv" in layer.name.lower():
            conv_layer_name = layer.name
            break

    grad_model = None
    if conv_layer_name is not None:
        try:
            conv_out = model.get_layer(conv_layer_name).output
            logits_output = model.output
            grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_out, logits_output])
        except Exception:
            grad_model = None

    return model, grad_model, conv_layer_name, model_path

@st.cache_resource(show_spinner=False)
def load_label_map():
    json_path = resolve_artifact("label_to_idx*.json")
    if json_path is None:
        idx_to_label = {0: "UNKNOWN"}
        label_to_idx = {"UNKNOWN": 0}
        return idx_to_label, label_to_idx, None
    try:
        with open(json_path, "r") as f:
            label_to_idx = json.load(f)
        idx_to_label = {int(v): k for k, v in label_to_idx.items()}
        return idx_to_label, label_to_idx, json_path
    except Exception:
        idx_to_label = {0: "UNKNOWN"}
        label_to_idx = {"UNKNOWN": 0}
        return idx_to_label, label_to_idx, json_path

def run_gradcam(grad_model, sample):
    """
    Compute Grad-CAM for a single sample (shape: (1, T, F)).
    Returns:
        cam_resized: (T,) Grad-CAM weights over time, or None if cannot compute
        class_idx:   int, predicted class index (or 0 fallback)
    """
    if grad_model is None:
        return None, 0

    sample_tf = tf.convert_to_tensor(sample)  # (1, T, F)

    try:
        with tf.GradientTape() as tape:
            conv_outs, preds = grad_model(sample_tf)

            if isinstance(preds, (list, tuple)):
                preds_tensor = preds[-1]
            else:
                preds_tensor = preds  # shape (1, C)

            class_idx_tensor = tf.argmax(preds_tensor[0])
            class_idx = int(class_idx_tensor.numpy())

            loss = preds_tensor[:, class_idx]  # shape (1,)
            grads = tape.gradient(loss, conv_outs)  # (1, T', C)
            if grads is None:
                return None, class_idx

            weights = tf.reduce_mean(grads, axis=1)  # (1, C)
            cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
            cam = tf.nn.relu(cam).numpy()[0]  # (T',)

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

            T_in = int(sample.shape[1])
            T_cam = cam.shape[0]
            cam_resized = np.interp(
                np.linspace(0, T_cam - 1, T_in),
                np.arange(T_cam),
                cam,
            )

            return cam_resized, class_idx
    except Exception:
        return None, 0

# =============================================================================
# EVALUATION ENGINE
# =============================================================================
def evaluate_model_on_folder(model, idx_to_label, test_dir: str = "vocapra_project/test_audio"):
    """
    Expects folder structure:
    test_audio/
        CLASS_A/
            a1.wav
            a2.wav
        CLASS_B/
            ...
    Returns dict with metrics and confusion matrix.
    """
    label_order = sorted(list(set(idx_to_label.values())))
    true_labels = []
    pred_labels = []

    test_root = Path(test_dir)
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # iterate each class folder (use label_order for predictable ordering)
    for class_name in label_order:
        class_folder = test_root / class_name
        if not class_folder.exists():
            continue
        for wav_file in sorted(class_folder.glob("*.wav")):
            try:
                y, _ = librosa.load(wav_file, sr=SR, mono=True)
            except Exception:
                try:
                    data, sr_raw = sf.read(wav_file)
                    if data.ndim == 2:
                        data = np.mean(data, axis=1)
                    y = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr_raw, target_sr=SR)
                except Exception:
                    continue

            feats = compute_mfcc_with_deltas(y, sr=SR)
            fx = to_fixed_frames(feats)
            x_in = np.expand_dims(fx, 0)
            try:
                preds = model.predict(x_in, verbose=0)
                if isinstance(preds, (list, tuple)):
                    preds = preds[-1]
                pred_idx = int(np.argmax(preds[0]))
                pred_label = idx_to_label.get(pred_idx, "UNKNOWN")
            except Exception:
                pred_label = "UNKNOWN"

            true_labels.append(class_name)
            pred_labels.append(pred_label)

    if len(true_labels) == 0:
        raise ValueError("No test samples found or readable in the provided test directory.")

    # Compute metrics
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=label_order, zero_division=0
    )
    report = classification_report(
        true_labels, pred_labels, labels=label_order, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "report": report,
        "cm": cm,
        "labels": label_order,
        "true_labels": true_labels,
        "pred_labels": pred_labels
    }

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
# STREAMLIT UI CONFIG & CSS
# =============================================================================
st.set_page_config(page_title="VOCAPRA HUD", page_icon="💠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

    [data-testid="stAppViewContainer"] {
        background-color: #030508;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(0, 243, 255, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(188, 19, 254, 0.08), transparent 25%);
        color: #e0e0e0;
    }
    [data-testid="stHeader"] { background: transparent; }

    * { font-family: 'Space Grotesk', sans-serif !important; }
    code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

    .hud-card {
        background: rgba(10, 15, 25, 0.7);
        border: 1px solid rgba(0, 243, 255, 0.15);
        border-radius: 4px;
        padding: 1.5rem;
        position: relative;
        backdrop-filter: blur(8px);
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
        clip-path: polygon(
            0 0, 100% 0, 
            100% calc(100% - 15px), calc(100% - 15px) 100%, 
            0 100%
        );
    }

    .hud-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 3px; height: 100%;
        background: linear-gradient(to bottom, #00f3ff, transparent);
        opacity: 0.5;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #333;
        background: rgba(0,0,0,0.3);
        padding: 2rem;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00f3ff;
        background: rgba(0, 243, 255, 0.05);
    }
    [data-testid="stFileUploader"] section { background: transparent; }

    .hud-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fff, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        line-height: 1;
    }
    .hud-subtitle {
        color: #00f3ff;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 2rem;
    }

    .label-small {
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace !important;
        margin-bottom: 0.5rem;
    }

    .value-big {
        font-size: 2rem;
        font-weight: 300;
        color: #fff;
    }

    @keyframes blink { 50% { opacity: 0.3; } }
    .blink { animation: blink 2s infinite; }

    .glow-text {
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
    }

    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #030508;
        padding: 5px 0;
        z-index: 1000;
        border-top: 1px solid rgba(10, 15, 25, 0.7);
        pointer-events: none;
    }
    .fixed-footer * { pointer-events: auto; }

    ::-webkit-scrollbar { width: 8px; background: #050505; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00f3ff; }
    </style>
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
        > MODEL: {model_path.name if model_path else 'MISSING'}
        > SR: {SR} Hz
        > FRAMES: {TARGET_FRAMES}
        > STATUS: { 'ONLINE' if model_path else 'DEMO' }
        """, language="yaml"
    )
    st.markdown("---")
    st.markdown("<div class='label-small' style='color:#444'>ARCHITECTURE</div>", unsafe_allow_html=True)
    st.caption("Conv1D Stack / GlobalAvgPool / Softmax (approx.)")

if model is None or not idx_to_label:
    st.warning("Some artifacts are missing from `vocapra_project/`. App will run in demo mode with fallback outputs.")
    if not st.session_state.get('demo_mode', False):
        st.session_state.demo_mode = True

# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.markdown("<div class='hud-title'>VOCAPRA <span style='color:#00f3ff'>.AI</span></div>", unsafe_allow_html=True)
st.markdown("<div class='hud-subtitle'>// Acoustic Event Recognition System v4.0</div>", unsafe_allow_html=True)

c1, c2 = st.columns([1.5, 1])

with c1:
    st.markdown("<div class='label-small'>INPUT STREAM</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload WAV", type=["wav"], label_visibility="collapsed")

with c2:
    num_classes = len(idx_to_label) if idx_to_label else 0
    st.markdown(
        f"""
        <div class="hud-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="label-small">SYSTEM STATUS</div>
                    <div style="color:#00f3ff; font-weight:bold;">● {'OPERATIONAL' if model is not None else 'DEMO'}</div>
                </div>
                <div style="text-align:right;">
                    <div class="label-small">CLASSES</div>
                    <div class="mono" style="font-size:1.5rem;">{num_classes:02d}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded is None:
    st.stop()

# Playback
try:
    uploaded.seek(0)
    st.audio(uploaded.getvalue(), format='audio/wav')
except Exception:
    pass

# =============================================================================
# INFERENCE & RESULTS
# =============================================================================
try:
    uploaded.seek(0)
    y, sr = librosa.load(uploaded, sr=SR, mono=True)
except Exception:
    try:
        uploaded.seek(0)
        data, sr_raw = sf.read(uploaded)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        y = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr_raw, target_sr=SR)
        sr = SR
    except Exception as e:
        st.error(f"Failed to read uploaded audio: {e}")
        st.stop()

feats = compute_mfcc_with_deltas(y, sr=sr)
fixed = to_fixed_frames(feats, TARGET_FRAMES)
x_in = np.expand_dims(fixed, axis=0)

probs = None
pred_idx = 0
if model is None:
    num = len(idx_to_label) or 3
    probs = np.zeros(num, dtype=float)
    probs[0] = 1.0
    pred_idx = int(np.argmax(probs))
else:
    try:
        with st.spinner("Running inference..."):
            preds = model.predict(x_in, verbose=0)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            probs = np.array(preds[0], dtype=float)
            pred_idx = int(np.argmax(probs))
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        num = len(idx_to_label) or 3
        probs = np.zeros(num, dtype=float)
        probs[0] = 1.0
        pred_idx = 0

pred_label = idx_to_label.get(pred_idx, "UNKNOWN").upper()
conf = float(probs[pred_idx]) if probs is not None and len(probs) > pred_idx else 0.0

st.write("")
st.markdown(f"<div class='label-small blink'>Analyzing... COMPLETE</div>", unsafe_allow_html=True)

# Primary detection card
st.markdown(
    f"""
    <div class="hud-card" style="border-left: 4px solid #bc13fe;">
        <div class="label-small" style="color:#bc13fe;">PRIMARY DETECTION</div>
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <div class="hud-title" style="font-size: 4rem; background: linear-gradient(to right, #fff, #bc13fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {pred_label}
            </div>
            <div style="text-align:right;">
                <div class="label-small">CONFIDENCE INTERVAL</div>
                <div class="mono glow-text" style="font-size: 2rem; color:#fff;">{conf*100:05.2f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Visualization grid
g1, g2 = st.columns([1.8, 1.2])

with g1:
    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label-small'>SIGNAL OSCILLOSCOPE</div>", unsafe_allow_html=True)
    fig = make_neon_plot(np.linspace(0, len(y)/sr, len(y)), y)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label-small'>PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)

    if probs is None:
        probs = np.zeros(len(idx_to_label) or 1, dtype=float)
    top_k = min(5, len(probs))
    sorted_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = [idx_to_label.get(i, "UNKNOWN") for i in sorted_indices]
    top_vals = probs[sorted_indices]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color='#bc13fe', height=0.5)
    for i, rect in enumerate(bars):
        width = rect.get_width()
        ax.text(width + 0.05, rect.get_y() + rect.get_height() / 2.0,
                f'{width:.2f}', ha='left', va='center', color='#bc13fe', fontsize=8, family='monospace')

    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels([l.upper() for l in top_labels[::-1]], color='#e0e0e0', fontfamily='monospace', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)

    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Grad-CAM
if grad_model and model is not None:
    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)

    with st.spinner("Generating activation map..."):
        cam, _ = run_gradcam(grad_model, x_in)

    if cam is not None:
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        ax.imshow(fixed.T, origin="lower", aspect="auto", cmap='ocean', alpha=0.2)
        extent = [0, fixed.shape[0], 0, fixed.shape[1]]
        ax.imshow(np.tile(cam, (fixed.shape[1], 1)), origin="lower", aspect="auto",
                  alpha=0.8, cmap='inferno', extent=extent)

        ax.set_facecolor("none")
        ax.axis('off')
        ax.axhline(y=0, color='#333', linewidth=1)

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Grad-CAM not available for this model configuration.")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Grad-CAM not available (missing conv layer or model artifact).")

# =============================================================================
# RESEARCH-GRADE EVALUATION UI
# =============================================================================
st.markdown("---")
st.subheader("📊 Research-Grade Evaluation Metrics")

col_eval1, col_eval2 = st.columns([2, 1])
with col_eval1:
    st.markdown("**Test dataset folder** (structure: test_audio/CLASS/*.wav)")
    test_dir_input = st.text_input("Test dataset path", value="vocapra_project/test_audio")
with col_eval2:
    run_eval_btn = st.button("Run Full Evaluation on Test Dataset")

if run_eval_btn:
    if model is None:
        st.error("Model not loaded. Cannot run evaluation.")
    else:
        try:
            with st.spinner("Evaluating model on test set... This may take a while depending on dataset size."):
                eval_out = evaluate_model_on_folder(model, idx_to_label, test_dir=test_dir_input)

            st.success("Evaluation complete!")

            # Overall accuracy
            st.metric("Overall Accuracy", f"{eval_out['accuracy']*100:.2f}%")

            # Per-class report DataFrame
            report = eval_out["report"]
            df_report = pd.DataFrame(report).T
            # present common columns nicely if available
            if 'support' in df_report.columns:
                df_show = df_report[['precision', 'recall', 'f1-score', 'support']].fillna(0)
            else:
                df_show = df_report.fillna(0)
            st.markdown("### Per-Class Precision / Recall / F1-Score")
            st.dataframe(df_show.style.format({
                'precision': "{:.3f}",
                'recall': "{:.3f}",
                'f1-score': "{:.3f}",
                'support': "{:.0f}"
            }))

            # Confusion matrix heatmap
            st.markdown("### Confusion Matrix")
            labels = eval_out['labels']
            fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.6), max(4, len(labels)*0.4)))
            sns.heatmap(eval_out['cm'], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

            # Per-class F1 bar chart
            st.markdown("### Per-Class F1 Score")
            fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.6), 4))
            ax.bar(labels, eval_out['f1'], color='#00f3ff')
            ax.set_ylabel("F1 Score")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

            # Provide downloadable CSV of classification report
            csv_bytes = df_show.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Download per-class report (CSV)", data=csv_bytes, file_name="vocapra_class_report.csv", mime="text/csv")

        except FileNotFoundError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

# =============================================================================
# COPYRIGHT FOOTER
# =============================================================================
st.markdown(
    """
    <div class='fixed-footer'>
        <div class='mono' style='text-align: center; color: #666; font-size: 0.7rem; padding-bottom: 5px;'>
            &copy; 2025 Rights Reserved by BioQuantum Labs
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
