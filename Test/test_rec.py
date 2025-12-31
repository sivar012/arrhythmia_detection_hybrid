# CNN + META-RF PIPELINE

import os
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier

# CONFIG
MODEL_PATH = "D:\\Arrhythmia\\models\\best_final_hybrid.h5"   

BEAT_LEN = 280
PRE_R = 90
POST_R = 190

AAMI_CLASSES = ['N','S','V','F','Q']
CLASS_TO_IDX = {'N':0,'S':1,'V':2,'F':3,'Q':4}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}

# Record-level GT (used ONLY to train Meta-RF once)
record_major_class = {
    100:"S",101:"S",102:"Q",103:"S",104:"Q",105:"V",106:"V",107:"Q",
    108:"V",109:"V",111:"V",112:"S",113:"S",114:"V",115:"N",116:"V",
    117:"S",118:"S",119:"V",121:"S",122:"N",123:"V",124:"V"
}

# LOAD CNN
print("Loading beat-level CNN...")
beat_model = tf.keras.models.load_model(MODEL_PATH)
print("CNN loaded ✔")

# ECG FILTER
def filter_ecg(sig, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5/nyq, 40/nyq], btype='band')
    sig = filtfilt(b, a, sig)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return sig

# BEAT EXTRACTION (UPDATED)
def extract_beats_from_record(rec_id, base_path):
    record = wfdb.rdrecord(os.path.join(base_path, rec_id))
    ann = wfdb.rdann(os.path.join(base_path, rec_id), 'atr')

    fs = record.fs
    sig = filter_ecg(record.p_signal[:,0], fs)

    beats = []
    rlocs = ann.sample

    for r in rlocs:
        start = max(0, r - PRE_R)
        end = min(len(sig), r + POST_R)
        beat = sig[start:end]

        if len(beat) < BEAT_LEN:
            beat = np.pad(beat, (0, BEAT_LEN - len(beat)))
        else:
            beat = beat[:BEAT_LEN]

        beats.append(beat)

    X = np.array(beats, dtype=np.float32)[..., np.newaxis]
    return X, sig, rlocs, fs

# BEAT PREDICTION
def predict_beats(X):
    probs = beat_model.predict(X, verbose=0)
    idx = np.argmax(probs, axis=1)
    preds = [IDX_TO_CLASS[i] for i in idx]
    return preds, probs

# RECORD FEATURES
def record_features(rec_id, base_path):
    X, _, _, _ = extract_beats_from_record(rec_id, base_path)
    preds, probs = predict_beats(X)

    counts = Counter(preds)
    total = sum(counts.values())
    arr = counts['S'] + counts['V'] + counts['F'] + counts['Q']

    feat = [
        total, arr,
        counts['N'], counts['S'], counts['V'],
        counts['F'], counts['Q'],
        np.max(probs[:,1:])
    ]
    return np.array(feat).reshape(1, -1)

# TRAIN META RANDOM FOREST (ONCE)
print("Training Meta RandomForest...")
X_meta, y_meta = [], []

BASE_TRAIN_DIR = "mitbih_train"   # folder containing MIT-BIH training files

for rid, label in record_major_class.items():
    X_meta.append(record_features(str(rid), BASE_TRAIN_DIR).squeeze())
    y_meta.append(CLASS_TO_IDX[label])

meta_clf = RandomForestClassifier(n_estimators=300, random_state=42)
meta_clf.fit(np.array(X_meta), np.array(y_meta))

print("Meta-RF trained ✔")

# FINAL INFERENCE FUNCTION (FLASK USES THIS)
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
    plt.scatter(rlocs[rlocs < n], sig[rlocs[rlocs < n]], c='red', s=15)
    plt.title(f"ECG Record {rec_id} | Predicted: {label}")
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

    return label, prob, plot_name
