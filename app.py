# ============================
# VOCAPRA — INFERENCE + EVALUATION (No training, model already trained)
# ============================

from pathlib import Path
import json, io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.colab import files
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

# -----------------------
# GLOBAL PATHS
# -----------------------
OUTDIR = Path("vocapra_project")
OUTDIR.mkdir(exist_ok=True)

MODEL_PATH = OUTDIR / "best_model.h5"
LABEL_MAP_PATH = OUTDIR / "label_to_idx.json"

# -----------------------
# UPLOAD MODEL
# -----------------------
print("Upload your best_model.h5")
uploaded = files.upload()
for name, data in uploaded.items():
    if name.endswith(".h5"):
        with open(MODEL_PATH, "wb") as f:
            f.write(data)
        print("✅ Model saved:", MODEL_PATH)

# -----------------------
# UPLOAD LABEL MAP
# -----------------------
print("\nUpload your label_to_idx.json")
uploaded = files.upload()
for name, data in uploaded.items():
    if name.endswith(".json"):
        with open(LABEL_MAP_PATH, "wb") as f:
            f.write(data)
        print("✅ Label map saved:", LABEL_MAP_PATH)

# -----------------------
# LOAD MODEL + LABELS
# -----------------------
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH) as f:
    label_to_idx = json.load(f)
idx_to_label = {int(v): k for k, v in label_to_idx.items()}

print("\n✅ Model loaded. Classes:", list(idx_to_label.values()))

# -----------------------
# AUDIO LOADING & FEATURE EXTRACTION
# -----------------------
SR = 22050
N_MFCC = 40
MAX_LEN = 400

def load_audio(path, sr=SR):
    try:
        audio, orig_sr = sf.read(path)
        if orig_sr != sr:
            audio = librosa.resample(audio.astype(float), orig_sr=orig_sr, target_sr=sr)
    except:
        audio, _ = librosa.load(path, sr=sr)
    return audio

def extract_features(audio, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T
    if features.shape[0] < max_len:
        pad = max_len - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))
    else:
        features = features[:max_len]
    return features.astype(np.float32)

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_file(path):
    audio = load_audio(path)
    feats = extract_features(audio)
    feats = feats[np.newaxis, ..., np.newaxis]  # (1, T, F, 1)
    probs = model.predict(feats, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_label[pred_idx]
    confidence = float(probs[pred_idx]*100)
    return pred_label, confidence, probs

# -----------------------
# UPLOAD AUDIO FILE(S)
# -----------------------
print("\nUpload WAV files for prediction and evaluation")
uploaded = files.upload()
audio_files = list(uploaded.keys())

y_true = []
y_pred = []
probs_list = []

# -----------------------
# PREDICTION LOOP
# -----------------------
for f in audio_files:
    pred_label, conf, probs = predict_file(f)
    print(f"\nFile: {f}")
    print(f"Predicted: {pred_label} ({conf:.2f}%)")
    y_pred.append(pred_label)
    probs_list.append(probs)
    # Ask user for true label if evaluating multiple files
    true_label = input(f"Enter true label for {f} (or press Enter to skip evaluation): ")
    if true_label:
        y_true.append(true_label)
    else:
        y_true.append(pred_label)  # fallback if not provided

# -----------------------
# EVALUATION METRICS
# -----------------------
if y_true:
    y_true_idx = [label_to_idx[lbl] for lbl in y_true]
    y_pred_idx = [label_to_idx[lbl] for lbl in y_pred]
    probs_array = np.vstack(probs_list)
    acc = accuracy_score(y_true_idx, y_pred_idx)
    bal_acc = balanced_accuracy_score(y_true_idx, y_pred_idx)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='macro', zero_division=0)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    print("\nConfusion Matrix:\n", cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_idx, y_pred_idx, target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())], zero_division=0))

    # Optional: ROC-AUC if more than 2 classes
    if probs_array.shape[1] > 1:
        y_true_bin = label_binarize(y_true_idx, classes=sorted(idx_to_label.keys()))
        try:
            roc_macro = roc_auc_score(y_true_bin, probs_array, average='macro', multi_class='ovr')
            roc_micro = roc_auc_score(y_true_bin, probs_array, average='micro', multi_class='ovr')
            print(f"ROC-AUC Macro: {roc_macro:.4f}")
            print(f"ROC-AUC Micro: {roc_micro:.4f}")
        except:
            print("ROC-AUC could not be computed.")
