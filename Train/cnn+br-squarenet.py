# ==========================
# FORCE CPU (avoids TF 2.19 GPU MaxPool+Dropout bug)
# ==========================
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # <-- do this BEFORE importing tensorflow

import tensorflow as tf
tf.keras.backend.clear_session()

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import optimizers, losses, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization,
    Dropout, Dense, Activation, Add,
    GlobalMaxPool1D, GlobalAveragePooling1D,
    Concatenate
)

# ==========================
# 1. LOAD DATA
# ==========================
df_train = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_train.csv")
df_test  = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_test.csv")

# shuffle train
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

Y      = df_train["label"].values.astype(np.int8)
X      = df_train.drop("label", axis=1).values[..., np.newaxis]   # (n, 280, 1)
Y_test = df_test["label"].values.astype(np.int8)
X_test = df_test.drop("label", axis=1).values[..., np.newaxis]

class_names = ["N","S","V","F","Q"]
n_classes = 5

print("Train:", X.shape, "Test:", X_test.shape)

# ==========================
# 2. BR-SquareNet BLOCK
# ==========================
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

# ==========================
# 3. HYBRID PURE CNN + PURE BR-SquareNet
# ==========================
def get_cnn_br_hybrid():
    inp = Input(shape=(280, 1))

    # ---- Branch 1: pure CNN ----
    c = Conv1D(32, 7, padding="same", activation="relu")(inp)
    c = BatchNormalization()(c)
    c = Conv1D(32, 7, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    c = MaxPool1D(2)(c)
    c = Dropout(0.2)(c)

    c = Conv1D(64, 5, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    c = Conv1D(64, 5, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    c = MaxPool1D(2)(c)
    c = Dropout(0.2)(c)

    c = Conv1D(128, 3, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    c = Conv1D(128, 3, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    c = MaxPool1D(2)(c)
    c = Dropout(0.25)(c)

    c = GlobalMaxPool1D()(c)
    c = Dropout(0.3)(c)

    # ---- Branch 2: BR-SquareNet ----
    r = Conv1D(32, 7, padding="same")(inp)
    r = BatchNormalization()(r)
    r = Activation("relu")(r)

    r = square_block(r, 32, kernel_size=7, pool=True)
    r = square_block(r, 64, kernel_size=5, pool=True)
    r = square_block(r, 128, kernel_size=3, pool=True)
    r = square_block(r, 128, kernel_size=3, pool=False)

    r = GlobalAveragePooling1D()(r)
    r = Dropout(0.3)(r)

    # ---- Fuse ----
    x = Concatenate()([c, r])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model

model = get_cnn_br_hybrid()
model.summary()

file_path = "/content/drive/MyDrive/Major/CNN+BR-SquareNett/hybrid_pure_cnn_br_squarenet.h5"

checkpoint = ModelCheckpoint(file_path, monitor="val_accuracy",
                             save_best_only=True, mode="max", verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=15, mode="max", verbose=1,
                      restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                              patience=7, min_lr=1e-7, verbose=1)
callbacks_list = [checkpoint, early, reduce_lr]

print(" Training Hybrid (pure CNN + pure BR-SquareNet) on CPU...")
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list,
    verbose=1
)

# ==========================
# 4. EVAL ON TEST SET
# ==========================
model.load_weights(file_path)

proba = model.predict(X_test)
pred  = np.argmax(proba, axis=1)

f1 = f1_score(Y_test, pred, average="macro")
acc = accuracy_score(Y_test, pred)
print("Test F1 (macro):", f1)
print("Test accuracy   :", acc)

cm = confusion_matrix(Y_test, pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hybrid CNN+BR-SquareNet – Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(Y_test, pred, target_names=class_names))

# ==========================
# 5. TRAINING CURVES
# ==========================
plt.figure(figsize=(7,4))
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.legend(); plt.grid(True); plt.title("Accuracy"); plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.legend(); plt.grid(True); plt.title("Loss"); plt.tight_layout()
plt.show()


# ==========================
# 6. ROC CURVES (Multi-class)
# ==========================
n_classes = 5
# One‑vs‑rest binarization of true labels
Y_test_bin = label_binarize(Y_test, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
    )

plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Hybrid CNN + BR-SquareNet")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
