import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import butter, filtfilt
import wfdb

# CONFIGURATION
LOCAL_DB_DIR = '/content/drive/MyDrive/Siva/data/mitbih'

BEAT_LEN = 280          # final beat length 
PRE_R  = 90             # samples before R-peak
POST_R = 190            # samples after R-peak
RANDOM_STATE = 42
TEST_SIZE = 0.2

# CORRECT AAMI MAPPING (MIT-BIH → AAMI EC57 5 classes)
AAMI_MAP = {
    # N class (Normal)
    'N':'N','L':'N','R':'N','e':'N','j':'N','E':'N',

    # S class (Supraventricular)
    'A':'S','a':'S','J':'S','S':'S',

    # V class (Ventricular)
    'V':'V','!':'V','x':'V','|':'V',

    # F class (Fusion)
    'F':'F','f':'F',

    # Q class (Unknown / Paced / Artifact)
    '/':'Q','Q':'Q','+':'Q','"':'Q','~':'Q','[':'Q',']':'Q'
}

AAMI_CLASSES = ['N','S','V','F','Q']
CLASS_TO_IDX = {'N':0,'S':1,'V':2,'F':3,'Q':4}


# ECG FILTERING
def filter_ecg(sig, fs):
    nyq = 0.5 * fs

    # Bandpass 0.5–40 Hz
    b1, a1 = butter(4, [0.5/nyq, 40/nyq], btype='band')
    sig = filtfilt(b1, a1, sig)

    # High-pass 0.5 Hz
    b2, a2 = butter(2, 0.5/nyq, btype='high')
    sig = filtfilt(b2, a2, sig)

    # Normalize (z-score)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    return sig


# BEAT EXTRACTION
def extract_beats_from_record(rec_name):
    rec_path = os.path.join(LOCAL_DB_DIR, rec_name)

    try:
        record = wfdb.rdrecord(rec_path)
        ann    = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f" Cannot read record {rec_name}: {e}")
        return [], []

    fs  = record.fs
    sig = record.p_signal[:, 0]  # MLII

    sig = filter_ecg(sig, fs)

    beats  = []
    labels = []

    for r, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI_MAP:
            continue

        start = max(0, r - PRE_R)
        end   = min(len(sig), r + POST_R)
        beat  = sig[start:end]

        # pad/trim to BEAT_LEN
        if len(beat) < BEAT_LEN:
            padded = np.zeros(BEAT_LEN)
            padded[:len(beat)] = beat
            beat = padded
        elif len(beat) > BEAT_LEN:
            beat = beat[:BEAT_LEN]

        beats.append(beat)
        labels.append(AAMI_MAP[sym])

    return beats, labels


# MAIN PROCESSING
print(" Processing MIT-BIH records...\n")

records = sorted([
    f.replace('.dat','')
    for f in os.listdir(LOCAL_DB_DIR)
    if f.endswith('.dat')
])

all_beats   = []
all_labels  = []
total_beats = 0

for i, rec in enumerate(records):
    beats, labels = extract_beats_from_record(rec)
    total_beats += len(beats)

    print(f"Record {rec}: {len(beats)} beats")

    all_beats.extend(beats)
    all_labels.extend(labels)

    if (i+1) % 10 == 0:
        print(f" Progress {i+1}/{len(records)} — {total_beats} total beats\n")

X = np.array(all_beats, dtype=np.float32)
y = np.array(all_labels)

print("\n FINAL EXTRACTION RESULTS")
print(f" Total beats: {len(X):,}")
print(f" Class counts: {Counter(y)}\n")

# BALANCE + SAVE DATASET
from sklearn.model_selection import train_test_split

if len(X) == 0:
    print(" ERROR: No beats extracted. Check directory path.")
else:
    counts = Counter(y)
    print("Before balancing:", counts)

    # Mild balancing: each class up to min_count * 2
    min_count = min(counts.values())
    rng = np.random.default_rng(RANDOM_STATE)
    selected_idx = []

    for cls in AAMI_CLASSES:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        n = min(len(idx), min_count * 2)
        selected_idx.extend(rng.choice(idx, n, replace=False))

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)

    X_bal = X[selected_idx]
    y_bal = y[selected_idx]

    print("After balancing:", Counter(y_bal))

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=TEST_SIZE,
        stratify=y_bal, random_state=RANDOM_STATE
    )

    y_train_num = np.array([CLASS_TO_IDX[c] for c in y_train])
    y_test_num  = np.array([CLASS_TO_IDX[c] for c in y_test])

    cols = [f"sample_{i}" for i in range(BEAT_LEN)]

    out_dir = "/content/drive/MyDrive/Major/Pre_processing"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "mitbih_beats_train.csv")
    test_path  = os.path.join(out_dir, "mitbih_beats_test.csv")

    pd.DataFrame(X_train, columns=cols).assign(label=y_train_num)\
        .to_csv(train_path, index=False)

    pd.DataFrame(X_test, columns=cols).assign(label=y_test_num)\
        .to_csv(test_path, index=False)

    print("\n SUCCESS!")
    print(f" Train beats: {X_train.shape[0]}")
    print(f" Test beats : {X_test.shape[0]}")
    print(f" Saved to: {out_dir}")
