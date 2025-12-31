import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from keras import optimizers, losses, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import (Input, Bidirectional, LSTM, Dropout, Dense,
                          Conv1D, MaxPool1D, BatchNormalization,
                          Add, Activation, GlobalAveragePooling1D,
                          Concatenate)

# ==========================
# 1. LOAD DATA
# ==========================
df_train = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_train.csv")
df_test  = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_test.csv")

df_train = df_train.sample(frac=1).reset_index(drop=True)

Y      = df_train["label"].values.astype(np.int8)
X      = df_train.drop("label", axis=1).values[..., np.newaxis]    # (N,280,1)
Y_test = df_test["label"].values.astype(np.int8)
X_test = df_test.drop("label", axis=1).values[..., np.newaxis]

class_names = ["N","S","V","F","Q"]
n_classes = 5

print("Train:", X.shape, "Test:", X_test.shape)

# ==========================
# 2. BR-SquareNet BLOCK (pure CNN branch)
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
# 3. HYBRID MODEL: pure BiLSTM + pure BR-SquareNet
# ==========================
def get_hybrid_bilstm_br_model():
    inp = Input(shape=(280, 1))

    # ----- Branch 1: PURE BiLSTM (no Conv) -----
    b1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
    b1 = Dropout(0.3)(b1)
    b1 = Bidirectional(LSTM(64, return_sequences=False))(b1)
    b1 = Dropout(0.3)(b1)          # shape (None, 128)

    # ----- Branch 2: PURE BR-SquareNet CNN -----
    b2 = Conv1D(32, 7, padding="same")(inp)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b2 = square_block(b2, 32, kernel_size=7, pool=True)   # -> 140
    b2 = square_block(b2, 64, kernel_size=5, pool=True)   # -> 70
    b2 = square_block(b2, 128, kernel_size=3, pool=True)  # -> 35
    b2 = square_block(b2, 128, kernel_size=3, pool=False)
    b2 = GlobalAveragePooling1D()(b2)                     # -> (None,128)
    b2 = Dropout(0.3)(b2)

    # ----- Fuse both pure branches -----
    x = Concatenate()([b1, b2])      # -> (None, 256)

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

model = get_hybrid_bilstm_br_model()
model.summary()

file_path = "/content/drive/MyDrive/Major/Bi-LSTM+BR-SquareNet/hybrid_pure_bilstm_br_squarenet.h5"

checkpoint = ModelCheckpoint(file_path, monitor="val_accuracy",
                             save_best_only=True, mode="max", verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=15, mode="max", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                              patience=7, min_lr=1e-7, verbose=1)
callbacks_list = [checkpoint, early, reduce_lr]

print("Training hybrid (pure BiLSTM + pure BR-SquareNet)...")
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list,
    verbose=1
)

model.load_weights(file_path)
print("Loaded best hybrid weights!")

# ==========================
# 4. EVALUATION
# ==========================
pred_proba = model.predict(X_test)
pred_test  = np.argmax(pred_proba, axis=-1)

acc = accuracy_score(Y_test, pred_test)
f1_macro = f1_score(Y_test, pred_test, average="macro")
f1_weighted = f1_score(Y_test, pred_test, average="weighted")

print("\nRESULTS (pure BiLSTM + pure BR-SquareNet):")
print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-macro: {f1_macro:.4f}")
print(f"F1-weighted: {f1_weighted:.4f}")
# after training
pred_proba = model.predict(X_test)          # shape (N, 5)
pred_test  = np.argmax(pred_proba, axis=-1) # shape (N,)
# true labels
Y_test     = Y_test                         # shape (N,)
class_names = ['N','S','V','F','Q']


# 1. CONFUSION MATRIX
cm = confusion_matrix(Y_test, pred_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - Hybrid BiLSTM + BR-SquareNet')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

print("\nClassification report:")
print(classification_report(Y_test, pred_test,
                            target_names=class_names, digits=4))

# 2. TRAINING & VALIDATION CURVES
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. MULTICLASS ROC CURVES
n_classes = 5
Y_test_bin = label_binarize(Y_test, classes=range(n_classes))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Hybrid BiLSTM + BR-SquareNet')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()