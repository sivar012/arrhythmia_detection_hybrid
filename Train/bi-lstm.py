import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Bidirectional, LSTM, BatchNormalization

# ==========================
# 1. LOAD YOUR 280-SAMPLE DATASET
# ==========================
df_train = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/Major/Pre_processing/mitbih_beats_test.csv")

df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle

#  YOUR DATA FORMAT: 280 samples + label column (281 total)
Y = np.array(df_train['label'].values).astype(np.int8)
X = np.array(df_train.drop('label', axis=1).values)[..., np.newaxis]  # [N, 280, 1]

Y_test = np.array(df_test['label'].values).astype(np.int8)
X_test = np.array(df_test.drop('label', axis=1).values)[..., np.newaxis]

print(f" Train shape: {X.shape}, Labels: {np.unique(Y)}")
print(f" Test shape:  {X_test.shape}, Labels: {np.unique(Y_test)}")

class_names = ['N', 'S', 'V', 'F', 'Q']  # Your 5 AAMI classes

# ==========================
# 2. Bi-LSTM MODEL (280 samples)
# ==========================
def get_model():
    nclass = 5
    inp = Input(shape=(280, 1))  #  280-sample beats as sequence (timesteps=280, features=1)

    # ------- Bi-LSTM Block 1 -------
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # ------- Bi-LSTM Block 2 -------
    x = Bidirectional(LSTM(64, return_sequences=False))(x)  # last hidden state
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # ------- Dense classifier (same spirit as your CNN head) -------
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(nclass, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

# Create & save best model
model = get_model()
file_path = "/content/drive/MyDrive/Major/Bi-LSTM/mitbih_bilstm.h5"

# Callbacks
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy',
                             save_best_only=True, mode='max', verbose=1)
early = EarlyStopping(monitor='val_accuracy', patience=15, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                              patience=7, min_lr=1e-7, verbose=1)

callbacks_list = [checkpoint, early, reduce_lr]

# ==========================
# 3. TRAIN MODEL
# ==========================
print(" Starting Training (Bi-LSTM)...")
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list,
    verbose=1
)

# Load best weights
model.load_weights(file_path)
print(" Loaded best Bi-LSTM model weights!")

# ==========================
# 4. EVALUATE ON TEST SET
# ==========================
pred_proba = model.predict(X_test)
pred_test = np.argmax(pred_proba, axis=-1)

# Metrics
f1_macro = f1_score(Y_test, pred_test, average="macro")
f1_weighted = f1_score(Y_test, pred_test, average="weighted")
accuracy = accuracy_score(Y_test, pred_test)

print("\n FINAL RESULTS (Bi-LSTM):")
print(f"Test Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Test F1-macro:    {f1_macro:.4f}")
print(f"Test F1-weighted: {f1_weighted:.4f}")

# ==========================
# 5. CONFUSION MATRIX
# ==========================
cm = confusion_matrix(Y_test, pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MIT-BIH 5-Class Bi-LSTM\n(0=N,1=S,2=V,3=F,4=Q)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n CLASSIFICATION REPORT (Bi-LSTM):")
print(classification_report(Y_test, pred_test,
                            target_names=class_names, digits=4))

# ==========================
# 6. TRAINING HISTORY
# ==========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.title('Model Accuracy (Bi-LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss (Bi-LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================
# 7. ROC CURVES (Multi-class)
# ==========================
n_classes = 5
Y_test_bin = label_binarize(Y_test, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All 5 Classes (Bi-LSTM)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n TRAINING COMPLETE (Bi-LSTM)!")
print(" Model saved: best_mitbih_bilstm.h5")
