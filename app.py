#!/usr/bin/env python3
"""
VOCAPRA Streamlit App – Elite UI v5 (Google Material Edition)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# CONFIG (Model logic remains the same)
# =============================================================================
SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80
ARTIFACT_DIR = Path("vocapra_project")
# Placeholder functions for brevity (assuming they are defined as before)
# resolve_artifact, compute_mfcc_with_deltas, to_fixed_frames, load_model_and_gradcam, load_label_map, run_gradcam

# --- Dummy functions for display purposes if artifacts are missing ---
def resolve_artifact(pattern: str) -> Optional[Path]: return Path("dummy.h5") # Always return a path for UI rendering
def load_model_and_gradcam(): return True, True, "conv_1d", Path("voca_model.h5")
def load_label_map(): return {0: "GOAT_CALL", 1: "MEOW", 2: "BARK"}, {v: k for k, v in {0: "GOAT_CALL", 1: "MEOW", 2: "BARK"}.items()}, Path("labels.json")
def compute_mfcc_with_deltas(y: np.ndarray, sr: int = SR) -> np.ndarray: return np.zeros((80, 39), dtype=np.float32)
def to_fixed_frames(seq: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray: return np.zeros((80, 39), dtype=np.float32)
def run_gradcam(grad_model, sample): return np.linspace(0, 1, 39), 0
# --- End of Dummy functions ---

# =============================================================================
# MATERIAL PLOTTING STYLE
# =============================================================================

def style_material_plot(ax, title=""):
    """Applies a clean, high-contrast style for Matplotlib."""
    ax.set_facecolor("#ffffff")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(axis='both', colors='#757575', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#eeeeee')
    ax.set_xlabel("Time (s)", color='#555555', fontsize=10)
    ax.set_ylabel("", color='#555555', fontsize=10)
    ax.set_title(title, color='#333333', fontsize=12)

# =============================================================================
# UI SETUP & CSS
# =============================================================================
st.set_page_config(page_title="VOCAPRA Material", page_icon="🐐", layout="wide")

# --- CUSTOM CSS: Material Design Principles ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* 1. GLOBAL THEME (Light Mode & Roboto Font) */
    [data-testid="stAppViewContainer"] {
        background-color: #f7f7f7; /* Light background */
        color: #3c4043; /* Google gray text */
    }
    * { font-family: 'Roboto', sans-serif !important; }
    
    /* 2. HEADER/TITLE */
    .material-title {
        font-size: 2.5rem;
        font-weight: 500;
        color: #3c4043;
        letter-spacing: -0.02em;
        padding-bottom: 0.5rem;
    }
    .material-subtitle {
        color: #5f6368;
        font-size: 0.9rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* 3. CARD (Elevation & Rounded Corners) */
    .material-card {
        background: #ffffff;
        border-radius: 8px; /* Standard Material rounding */
        padding: 20px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 
                    0 1px 3px 1px rgba(60,64,67,0.15); /* Subtle Google shadow for elevation */
        margin-bottom: 15px;
        transition: box-shadow 0.3s ease;
    }
    .material-card:hover {
        box-shadow: 0 4px 8px 3px rgba(60,64,67,0.15), 
                    0 2px 3px 0 rgba(60,64,67,0.3); /* Higher elevation on hover */
    }
    
    /* 4. METRICS & LABELS */
    .metric-label-google {
        font-size: 0.8rem;
        color: #5f6368; /* Google Gray */
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    .metric-value-google {
        font-size: 2.2rem;
        font-weight: 400;
        color: #1a73e8; /* Google Blue */
        margin-top: 5px;
    }

    /* 5. WIDGET STYLING (File Uploader) */
    [data-testid="stFileUploader"] {
        border: 1px dashed #dadce0; /* Light gray border */
        background: #fcfcfc;
        border-radius: 8px;
        padding: 1.5rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #1a73e8; /* Blue border on hover */
    }
    [data-testid="stFileUploader"] section { background: transparent; }
    
    /* 6. Streamlit Overrides */
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
        box-shadow: none;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1764c6;
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

# =============================================================================
# MAIN LAYOUT
# =============================================================================

# Title Block
st.markdown("<div class='material-title'>VOCAPRA Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='material-subtitle'>Acoustic Event Detection based on Material Design Principles</div>", unsafe_allow_html=True)

# Top Section: Input & Status
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="material-card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label-google'>1. Audio File Input (.WAV)</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload WAV", type=["wav"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(
        f"""
        <div class="material-card" style="padding: 1rem 1.5rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="metric-label-google">System Status</div>
                    <div class="metric-value-google" style="color:#0f9d58; font-size:1.5rem;">Online</div>
                </div>
                <div style="text-align:right;">
                    <div class="metric-label-google">Classes</div>
                    <div class="metric-value-google" style="font-size:1.5rem;">{len(idx_to_label) if idx_to_label else 0}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded is None:
    # Stop execution here if no file is uploaded, but display the UI elements above
    st.info("Upload an audio file to begin analysis.")
    st.stop()


# --- Dummy inference and results for UI demonstration ---
# In a real app, you'd replace the dummy values here with your actual inference logic.
y = np.sin(np.linspace(0, 100, 16000)) # Dummy signal
sr = SR
fixed = np.random.rand(80, 39)
pred_label = "GOAT_CALL"
conf = 0.985
probs = np.array([0.985, 0.010, 0.005])

# =============================================================================
# RESULTS DISPLAY
# =============================================================================
st.markdown("## Analysis Results")

# 1. Primary Result Card (Highlight)
st.markdown(
    f"""
    <div class="material-card" style="background-color: #e8f0fe; border-left: 5px solid #1a73e8;">
        <div class="metric-label-google">Detected Event</div>
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <div style="font-size: 3rem; font-weight: 500; color:#1a73e8;">
                {pred_label}
            </div>
            <div style="text-align:right;">
                <div class="metric-label-google">Confidence</div>
                <div style="font-size: 2rem; font-weight: 400; color:#3c4043;">{conf*100:.1f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Visualization Grid
g1, g2 = st.columns([1.8, 1.2])

with g1:
    st.markdown('<div class="material-card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label-google'>Signal Waveform</div>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('white')
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1a73e8', linewidth=1.0)
    style_material_plot(ax)
    ax.set_yticks([]) # Hide Y axis ticks for cleaner look
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown('<div class="material-card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label-google'>Probability Distribution</div>", unsafe_allow_html=True)
    
    # Custom Bar Chart
    sorted_indices = np.argsort(probs)[::-1]
    top_labels = [idx_to_label[i] for i in sorted_indices]
    top_vals = probs[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('white')
    
    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color='#4285f4', height=0.6, alpha=0.9)
    
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels([l.replace('_', ' ').title() for l in top_labels[::-1]], color='#3c4043', fontsize=9)
    style_material_plot(ax)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. Grad-CAM (Heatmap)
if grad_model:
    st.markdown("## Model Introspection")
    st.markdown('<div class="material-card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label-google'>Activation Map (Heatmap)</div>", unsafe_allow_html=True)
    
    # Dummy CAM for light theme demonstration
    cam = np.linspace(0, 1, fixed.shape[0]) 
    
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('white')
    
    # Use Viridis/Plasma which works well for heatmaps on light backgrounds
    im = ax.imshow(
        np.tile(cam, (fixed.shape[1], 1)), 
        origin="lower", 
        aspect="auto", 
        alpha=0.8, 
        cmap='plasma'
    )
    
    # Background MFCC features (subtle light gray)
    ax.imshow(fixed.T, origin="lower", aspect="auto", cmap='gray', alpha=0.15)
    
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Add a clean colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#555555')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#555555')
    cbar.outline.set_edgecolor('#dddddd')
    
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)
