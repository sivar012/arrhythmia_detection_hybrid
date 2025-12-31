import os
import numpy as np
import wfdb
import tensorflow as tf
import joblib
from collections import Counter
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier

# ========= CONFIG =========
TRAIN_DB_DIR = r"D:\\Arrhythmia\\Dataset\\mitbih"   
MODEL_PATH = r"D:\\Arrhythmia\\models\\best_final_hybrid.h5"

BEAT_LEN = 280
PRE_R = 90
POST_R = 190

CLASS_TO_IDX = {'N':0,'S':1,'V':2,'F':3,'Q':4}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}

record_major_class = {
    100:"S",101:"S",102:"Q",103:"S",104:"Q",105:"V",106:"V",107:"Q",
    108:"V",109:"V",111:"V",112:"S",113:"S",114:"V",115:"N",116:"V",
    117:"S",118:"S",119:"V",121:"S",122:"N",123:"V",124:"V"
}

# ========= LOAD Hybrid Model =========
beat_model = tf.keras.models.load_model(MODEL_PATH)

def filter_ecg(sig, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5/nyq, 40/nyq], btype='band')
    sig = filtfilt(b, a, sig)
    return (sig - sig.mean()) / (sig.std() + 1e-8)

def record_features(rec_id):
    record = wfdb.rdrecord(os.path.join(TRAIN_DB_DIR, str(rec_id)))
    ann = wfdb.rdann(os.path.join(TRAIN_DB_DIR, str(rec_id)), 'atr')

    fs = record.fs
    sig = filter_ecg(record.p_signal[:,0], fs)

    beats = []
    for r in ann.sample:
        beat = sig[max(0,r-PRE_R):min(len(sig),r+POST_R)]
        beat = np.pad(beat, (0, BEAT_LEN-len(beat)))[:BEAT_LEN]
        beats.append(beat)

    X = np.array(beats)[...,None]
    probs = beat_model.predict(X, verbose=0)
    preds = [IDX_TO_CLASS[i] for i in np.argmax(probs, axis=1)]

    counts = Counter(preds)
    return [
        len(preds),
        counts['S']+counts['V']+counts['F']+counts['Q'],
        counts['N'], counts['S'], counts['V'],
        counts['F'], counts['Q'],
        np.max(probs[:,1:])
    ]

# ========= TRAIN META-RF =========
X_meta, y_meta = [], []
for rid, lab in record_major_class.items():
    X_meta.append(record_features(rid))
    y_meta.append(CLASS_TO_IDX[lab])

meta_clf = RandomForestClassifier(n_estimators=300, random_state=42)
meta_clf.fit(X_meta, y_meta)

joblib.dump(meta_clf, "meta_rf.pkl")
print(" meta_rf.pkl created successfully")
