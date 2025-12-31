# ============================================================
# CNN + META-RF PIPELINE (INFERENCE ONLY - FLASK)
# ============================================================

import os
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import butter, filtfilt
import joblib
import matplotlib
matplotlib.use("Agg")   # <-- IMPORTANT

# ================= CONFIG =================
MODEL_PATH = r"D:\\A\\models\\best_final_hybrid.h5"
META_MODEL_PATH = r"D:\\A\\meta_rf\\meta_rf.pkl"


BEAT_LEN = 280
PRE_R = 90
POST_R = 190

CLASS_TO_IDX = {'N':0,'S':1,'V':2,'F':3,'Q':4}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}

# ================= LOAD MODELS =================
print("Loading beat-level CNN...")
beat_model = tf.keras.models.load_model(MODEL_PATH)
print("CNN loaded ✔")

print("Loading Meta RandomForest...")
meta_clf = joblib.load(META_MODEL_PATH)
print("Meta-RF loaded ✔")

# ================= ECG FILTER =================
def filter_ecg(sig, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5/nyq, 40/nyq], btype='band')
    sig = filtfilt(b, a, sig)
    return (sig - sig.mean()) / (sig.std() + 1e-8)

# ================= BEAT EXTRACTION =================
def extract_beats_from_record(rec_id, base_path):
    record = wfdb.rdrecord(os.path.join(base_path, rec_id))
    ann = wfdb.rdann(os.path.join(base_path, rec_id), 'atr')

    fs = record.fs
    sig = filter_ecg(record.p_signal[:,0], fs)

    beats = []
    rlocs = ann.sample

    for r in rlocs:
        beat = sig[max(0,r-PRE_R):min(len(sig),r+POST_R)]
        beat = np.pad(beat, (0, BEAT_LEN-len(beat)))[:BEAT_LEN]
        beats.append(beat)

    X = np.array(beats, dtype=np.float32)[...,None]
    return X, sig, rlocs, fs

# ================= RECORD FEATURES =================
def record_features(rec_id, base_path):
    X, _, _, _ = extract_beats_from_record(rec_id, base_path)
    probs = beat_model.predict(X, verbose=0)
    preds = [IDX_TO_CLASS[i] for i in np.argmax(probs, axis=1)]

    counts = Counter(preds)
    return np.array([[
        len(preds),
        counts['S']+counts['V']+counts['F']+counts['Q'],
        counts['N'], counts['S'], counts['V'],
        counts['F'], counts['Q'],
        np.max(probs[:,1:])
    ]])

# ================= FINAL INFERENCE =================
def predict_and_plot_record(rec_id, base_path, save_dir, seconds=10):

    X, sig, rlocs, fs = extract_beats_from_record(rec_id, base_path)
    feat = record_features(rec_id, base_path)

    idx = meta_clf.predict(feat)[0]
    prob = meta_clf.predict_proba(feat)[0][idx]
    label = IDX_TO_CLASS[idx]

    os.makedirs(save_dir, exist_ok=True)
    plot_name = f"{rec_id}_ecg.png"
    plot_path = os.path.join(save_dir, plot_name)

    n = int(seconds * fs)
    plt.figure(figsize=(15,4))
    plt.plot(sig[:n])
    plt.scatter(rlocs[rlocs<n], sig[rlocs[rlocs<n]], c='red', s=15)
    plt.title(f"ECG Record {rec_id} | Predicted: {label}")
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

    return label, prob, plot_name
