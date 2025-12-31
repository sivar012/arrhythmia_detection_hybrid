# ================================
# LOAD TEST SET
# ================================
df_test = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_test.csv")

X_test = df_test.iloc[:, :280].values.astype(np.float32)
Y_test = df_test['label'].values.astype(int)

print("Test beats:", X_test.shape[0])

# ================================
# LOAD TRAIN SET
# ================================
df_train = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_train.csv")

X_train = df_train.iloc[:, :280].values.astype(np.float32)
Y_train = df_train['label'].values.astype(int)

print("Train beats:", X_train.shape[0])
class_map = {
    0: "N (Normal)",
    1: "S (Supraventricular)",
    2: "V (Ventricular)",
    3: "F (Fusion)",
    4: "Q (Unknown)"
}
import numpy as np
import matplotlib.pyplot as plt

def show_single_beat_prediction(beat_id, model, X, Y=None):
    """
    beat_id : index of beat from X (train/test)
    model   : trained ECG model
    X       : dataset (X_train or X_test)
    Y       : labels (optional)
    """

    if beat_id < 0 or beat_id >= len(X):
        print(f"Invalid beat_id {beat_id}. Valid range: 0 to {len(X)-1}")
        return

    beat = X[beat_id]          # shape (280,)
    beat_input = beat.reshape(1, 280, 1)

    # ---- Prediction ----
    probs = model.predict(beat_input, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    pred_label = class_map[pred_class]

    # ---- True label (if Y exists) ----
    if Y is not None:
        true_class = int(Y[beat_id])
        true_label = class_map[true_class]
    else:
        true_label = "Unknown"

    # ---- Print summary ----
    print("\n========== BEAT PREDICTION ==========")
    print(f"Beat ID           : {beat_id}")
    print(f"True Label        : {true_label}")
    print(f"Predicted Label   : {pred_label}")
    print("Class Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {class_map[i]} : {p:.4f}")

    # ---- Plot ECG waveform + probability bar chart ----
    plt.figure(figsize=(12, 7))

    # ECG waveform
    plt.subplot(2, 1, 1)
    plt.plot(beat, linewidth=1.5)
    plt.title(f"ECG Beat #{beat_id} | True: {true_label} | Predicted: {pred_label}")
    plt.xlabel("Sample Index (0-279)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # Probability bar chart
    plt.subplot(2, 1, 2)
    labels = ["N", "S", "V", "F", "Q"]
    bars = plt.bar(labels, probs)
    plt.ylim(0, 1.05)
    plt.ylabel("Probability")
    plt.title("Class Probabilities")

    for b, p in zip(bars, probs):
        plt.text(b.get_x() + b.get_width()/2, p + 0.02, f"{p:.2f}",
                 ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()







show_single_beat_prediction(1033, model, X_test, Y_test)