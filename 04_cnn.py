"""
04_cnn.py
---------
Step 4 of the MHEALTH Human Action Detection Pipeline.
Builds and trains a 1D CNN using TensorFlow/Keras for
time-series classification of physical activities.
Compares CNN results against ML baselines.

Run:
    python 04_cnn.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── 2. Config ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
PLOTS_DIR     = os.path.join("outputs", "plots")
RESULTS_PATH  = os.path.join("results", "results.txt")
TARGET_COL    = "Activity"

EPOCHS        = 30
BATCH_SIZE    = 64
RANDOM_STATE  = 42

ACTIVITY_LABELS = {
    1:  "Standing",
    2:  "Sitting",
    3:  "Lying Down",
    4:  "Walking",
    5:  "Stairs",
    6:  "Waist Bend",
    7:  "Arm Raise",
    8:  "Crouching",
    9:  "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump",
}

ACCENT = "#00e5ff"

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── 3. Load Processed Data ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — 1D CNN TRAINING & EVALUATION")
print("=" * 60)

train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

X_train = train_df.drop(columns=[TARGET_COL]).values
y_train = train_df[TARGET_COL].values
X_test  = test_df.drop(columns=[TARGET_COL]).values
y_test  = test_df[TARGET_COL].values

print(f"\n[LOAD] Train : {X_train.shape}  |  Test : {X_test.shape}")

# ── 4. Encode Labels (1-12 → 0-11) ───────────────────────────────────────────
# CNN needs labels starting from 0
classes       = sorted(np.unique(y_train))
n_classes     = len(classes)
class_to_idx  = {c: i for i, c in enumerate(classes)}
idx_to_class  = {i: c for c, i in class_to_idx.items()}

y_train_idx = np.array([class_to_idx[c] for c in y_train])
y_test_idx  = np.array([class_to_idx[c] for c in y_test])

y_train_cat = to_categorical(y_train_idx, num_classes=n_classes)
y_test_cat  = to_categorical(y_test_idx,  num_classes=n_classes)

print(f"[PREP] Classes        : {n_classes}")
print(f"[PREP] y_train shape  : {y_train_cat.shape}")

# ── 5. Reshape for CNN input (samples, timesteps, features) ──────────────────
# Each row = one sample with 12 sensor features
# We treat each feature as a timestep → shape: (samples, 12, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"[PREP] CNN input shape : {X_train_cnn.shape}  (samples, timesteps, channels)")

# ── 6. Build 1D CNN Model ─────────────────────────────────────────────────────
print("\n[CNN] Building model...")

model = Sequential([
    # Block 1 — extract low-level patterns
    Conv1D(filters=64, kernel_size=3, activation="relu",
           padding="same", input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, padding="same"),
    Dropout(0.3),

    # Block 2 — extract higher-level patterns
    Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, padding="same"),
    Dropout(0.3),

    # Block 3 — deep feature extraction
    Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
    BatchNormalization(),
    Dropout(0.3),

    # Flatten + Dense classifier head
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary(line_length=80)

# ── 7. Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── 8. Train ──────────────────────────────────────────────────────────────────
print(f"\n[CNN] Training for up to {EPOCHS} epochs (batch={BATCH_SIZE})...")

history = model.fit(
    X_train_cnn, y_train_cat,
    epochs=BATCH_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
print("\n[CNN] Evaluating on test set...")

y_pred_prob = model.predict(X_test_cnn, verbose=0)
y_pred_idx  = np.argmax(y_pred_prob, axis=1)
y_pred      = np.array([idx_to_class[i] for i in y_pred_idx])

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"\n[CNN] Accuracy  : {acc:.4f}")
print(f"[CNN] F1-Macro  : {f1:.4f}")

# ── 10. Save Model ────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
cnn_model_path = os.path.join(MODELS_DIR, "CNN_1D.keras")
model.save(cnn_model_path)
print(f"[SAVE] CNN model saved -> {cnn_model_path}")

# ── 11. Plot — Training History ───────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     color=ACCENT,    label="Train Acc", linewidth=2)
axes[0].plot(history.history["val_accuracy"], color="#ff6ec7", label="Val Acc",   linewidth=2)
axes[0].set_title("CNN — Accuracy", fontsize=13, color=ACCENT, pad=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     color=ACCENT,    label="Train Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], color="#ff6ec7", label="Val Loss",   linewidth=2)
axes[1].set_title("CNN — Loss", fontsize=13, color=ACCENT, pad=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
history_path = os.path.join(PLOTS_DIR, "cnn_training_history.png")
plt.savefig(history_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[PLOT] Saved -> {history_path}")

# ── 12. Plot — CNN Confusion Matrix ──────────────────────────────────────────
classes_sorted = sorted(np.unique(y_test))
cm     = confusion_matrix(y_test, y_pred, labels=classes_sorted, normalize="true")
labels = [ACTIVITY_LABELS.get(c, str(c)) for c in classes_sorted]

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    cm, annot=True, fmt=".2f", cmap="YlOrRd",
    xticklabels=labels, yticklabels=labels,
    linewidths=0.4, linecolor="#111",
    ax=ax, cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},
)
ax.set_title("Confusion Matrix - 1D CNN", fontsize=13, color=ACCENT, pad=14)
ax.set_xlabel("Predicted", labelpad=10)
ax.set_ylabel("True",      labelpad=10)
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "cm_cnn.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[PLOT] Saved -> {cm_path}")

# ── 13. Plot — CNN vs ML Model Comparison ────────────────────────────────────
print("\n[PLOT] Generating CNN vs ML comparison chart...")

# ML results from 03_models.py (hardcoded for comparison)
all_results = pd.DataFrame([
    {"Model": "Logistic Reg", "Accuracy": 0.6487, "F1_Macro": 0.6046},
    {"Model": "KNN",          "Accuracy": 0.9754, "F1_Macro": 0.9689},
    {"Model": "Decision Tree","Accuracy": 0.9342, "F1_Macro": 0.9247},
    {"Model": "Random Forest","Accuracy": 0.9796, "F1_Macro": 0.9752},
    {"Model": "1D CNN",       "Accuracy": round(acc, 4), "F1_Macro": round(f1, 4)},
])

x      = np.arange(len(all_results))
width  = 0.35
colors = ["#00e5ff", "#ff6ec7"]

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width / 2, all_results["Accuracy"], width,
            label="Accuracy", color=colors[0], edgecolor="#000", linewidth=0.5)
b2 = ax.bar(x + width / 2, all_results["F1_Macro"], width,
            label="F1-Macro", color=colors[1], edgecolor="#000", linewidth=0.5, alpha=0.85)

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=8, color="#ccc",
    )

ax.set_xticks(x)
ax.set_xticklabels(all_results["Model"], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("All Models Comparison - ML vs CNN", fontsize=13, color=ACCENT, pad=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
full_comparison_path = os.path.join(PLOTS_DIR, "full_model_comparison.png")
plt.savefig(full_comparison_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[PLOT] Saved -> {full_comparison_path}")

# ── 14. Append CNN Results to results.txt ─────────────────────────────────────
with open(RESULTS_PATH, "a") as f:
    f.write("\n" + "=" * 60 + "\n")
    f.write("  1D CNN RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"F1-Macro  : {f1:.4f}\n\n")
    f.write(report + "\n")

print(f"[SAVE] CNN results appended -> {RESULTS_PATH}")

# ── 15. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY — ALL MODELS")
print("=" * 60)
print(all_results.to_string(index=False))
best = all_results.loc[all_results["F1_Macro"].idxmax()]
print(f"\n  Best Model : {best['Model']}  (F1-macro = {best['F1_Macro']:.4f})")
print("\n[DONE] Full ML + CNN pipeline complete!")
print("=" * 60)