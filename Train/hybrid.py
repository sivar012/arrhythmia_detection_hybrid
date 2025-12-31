# ============================================================
# 0. Imports & setup
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

from tensorflow.keras import optimizers, losses, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization, Dropout, Dense,
    Activation, Add, Concatenate, Bidirectional, LSTM
)

# ============================================================
# 1. LOAD 280-SAMPLE BEAT DATA
# ============================================================
train_path = "/content/drive/MyDrive/Major/Om/mitbih_beats_train.csv"
test_path  = "/content/drive/MyDrive/Major/Om/mitbih_beats_test.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# label column is named 'label'
Y      = df_train["label"].values.astype(np.int8)
X      = df_train.drop("label", axis=1).values[..., np.newaxis]   # (N, 280, 1)
Y_test = df_test["label"].values.astype(np.int8)
X_test = df_test.drop("label", axis=1).values[..., np.newaxis]

class_names = ["N","S","V","F","Q"]
n_classes = 5

print("Train:", X.shape, "Test:", X_test.shape)

# ============================================================
# 2. BR-SquareNet residual block
# ============================================================
from tensorflow.keras.layers import Add

def square_block(x, filters, kernel_size=3, pool=True):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = Conv1D(filters, kernel_size, padding="same")(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)

    out = Conv1D(filters, kernel_size, padding="same")(out)
    out = BatchNormalization()(out)

    out = Add()([shortcut, out])
    out = Activation("relu")(out)

    if pool:
        out = MaxPool1D(2)(out)
    return out

# ============================================================
# 3. FINAL ARCHITECTURE
# ============================================================
def get_final_hybrid_model(input_len=280, n_classes=5):
    inp = Input(shape=(input_len, 1))

    # Branch A: plain CNN
    a = Conv1D(32, 7, padding="same", activation="relu")(inp)
    a = BatchNormalization()(a)
    a = Conv1D(32, 7, padding="same", activation="relu")(a)
    a = BatchNormalization()(a)
    a = MaxPool1D(2)(a)
    a = Dropout(0.2)(a)

    a = Conv1D(64, 5, padding="same", activation="relu")(a)
    a = BatchNormalization()(a)
    a = Conv1D(64, 5, padding="same", activation="relu")(a)
    a = BatchNormalization()(a)
    a = MaxPool1D(2)(a)
    a = Dropout(0.2)(a)

    a = Conv1D(128, 3, padding="same", activation="relu")(a)
    a = BatchNormalization()(a)
    a = Conv1D(128, 3, padding="same", activation="relu")(a)
    a = BatchNormalization()(a)
    a = MaxPool1D(2)(a)     # -> ~35 time steps
    a = Dropout(0.25)(a)

    # Branch B: BR-SquareNet
    b = Conv1D(32, 7, padding="same")(inp)
    b = BatchNormalization()(b)
    b = Activation("relu")(b)

    b = square_block(b, 32, kernel_size=7, pool=True)   # 280 -> 140
    b = square_block(b, 64, kernel_size=5, pool=True)   # 140 -> 70
    b = square_block(b, 128, kernel_size=3, pool=True)  # 70  -> 35
    b = square_block(b, 128, kernel_size=3, pool=False) # keep 35

    # Fusion
    x = Concatenate(axis=-1)([a, b])   # (None, ~35, 256)

    # BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # Dense classifier
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model

model = get_final_hybrid_model(input_len=X.shape[1], n_classes=n_classes)
model.summary()

# ============================================================
# 4. TRAINING
# ============================================================
file_path = "/content/drive/MyDrive/Major/Om/best_final_hybrid.h5"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

checkpoint = ModelCheckpoint(file_path, monitor="val_accuracy",
                             save_best_only=True, mode="max", verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=15,
                      restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                              patience=7, min_lr=1e-7, verbose=1)
callbacks_list = [checkpoint, early, reduce_lr]

print(" Training final CNN + BR-SquareNet + BiLSTM model (CPU)...")
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list,
    verbose=1
)

# ============================================================
# 5. BEAT-LEVEL TEST EVALUATION
# ============================================================
model.load_weights(file_path)

proba = model.predict(X_test)
pred  = np.argmax(proba, axis=1)

acc = accuracy_score(Y_test, pred)
f1_macro    = f1_score(Y_test, pred, average="macro")
f1_weighted = f1_score(Y_test, pred, average="weighted")

print("\n FINAL RESULTS (Beat-level):")
print(f"Test Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-macro:     {f1_macro:.4f}")
print(f"F1-weighted:  {f1_weighted:.4f}")

print("\nClassification report:")
print(classification_report(Y_test, pred, target_names=class_names))
# ============================================================
# 5. BEAT-LEVEL TEST EVALUATION + PLOTS (CM, CURVES, ROC)
# ============================================================
# Load best weights
model.load_weights(file_path)

# --- Predictions & basic metrics ---
proba = model.predict(X_test)                 # shape: (N_test, n_classes)
pred  = np.argmax(proba, axis=1)             # predicted class indices

acc = accuracy_score(Y_test, pred)
f1_macro    = f1_score(Y_test, pred, average="macro")
f1_weighted = f1_score(Y_test, pred, average="weighted")

print("\n FINAL RESULTS (Beat-level):")
print(f"Test Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-macro:     {f1_macro:.4f}")
print(f"F1-weighted:  {f1_weighted:.4f}")

print("\nClassification report:")
print(classification_report(Y_test, pred, target_names=class_names))


# ============================================================
# 6. CONFUSION MATRIX (COUNTS + NORMALIZED)
# ============================================================
cm = confusion_matrix(Y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Counts) - Beat Level")
plt.tight_layout()
plt.show()

# Normalized confusion matrix (per true class)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            vmin=0.0, vmax=1.0)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Normalized) - Beat Level")
plt.tight_layout()
plt.show()


# ============================================================
# 7. TRAINING CURVES (ACCURACY & LOSS)
# ============================================================
train_acc  = history.history.get("accuracy", [])
val_acc    = history.history.get("val_accuracy", [])
train_loss = history.history.get("loss", [])
val_loss   = history.history.get("val_loss", [])

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 8. ROC CURVES & AUC (ONE-VS-REST)
# ============================================================
# Binarize labels for one-vs-rest ROC
Y_test_bin = label_binarize(Y_test, classes=range(n_classes))  # 0..4

fpr = {}
tpr = {}
roc_auc = {}

# ROC for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
roc_auc["macro"] = auc(all_fpr, mean_tpr)

plt.figure(figsize=(7, 6))

# Per-class ROC
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.3f})")

# Macro-average ROC
plt.plot(all_fpr, mean_tpr,
         label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
         linestyle="--", linewidth=2)

# Chance line
plt.plot([0, 1], [0, 1], "k--", label="Chance")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Beat Level (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
