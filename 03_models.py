"""
03_models.py
------------
Step 3 of the MHEALTH Human Action Detection Pipeline.
Loads processed train/test data, trains 4 ML classifiers,
evaluates them, saves models + plots + results.

Models:
    - Logistic Regression
    - K-Nearest Neighbours (KNN)
    - Decision Tree
    - Random Forest

Run:
    python 03_models.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── 2. Config ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
PLOTS_DIR     = os.path.join("outputs", "plots")
RESULTS_PATH  = os.path.join("results", "results.txt")
TARGET_COL    = "Activity"

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

# ── 3. Load Processed Data ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — ML MODEL TRAINING & EVALUATION")
print("=" * 60)

train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

print(f"\n[LOAD] Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"[LOAD] Classes : {sorted(y_train.unique())}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs("results",  exist_ok=True)

# ── 4. Define Models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        n_jobs=-1,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),
}

# ── 5. Train, Evaluate, Save Each Model ───────────────────────────────────────
summary = []
reports = []

for name, model in models.items():
    print(f"\n[MODEL] Training : {name} ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"[MODEL] {name:25s} | Acc: {acc:.4f} | F1-macro: {f1:.4f} | Time: {train_time:.1f}s")

    summary.append({
        "Model":        name,
        "Accuracy":     round(acc,  4),
        "F1_Macro":     round(f1,   4),
        "Train_Time_s": round(train_time, 1),
    })
    reports.append((name, report))

    # Save model as .pkl
    model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"[SAVE] Model saved -> {model_path}")

    # ── Confusion Matrix plot ─────────────────────────────────────────────────
    classes = sorted(y_test.unique())
    cm      = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
    labels  = [ACTIVITY_LABELS.get(c, str(c)) for c in classes]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="#111",
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title(f"Confusion Matrix - {name}", fontsize=13, color=ACCENT, pad=14)
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("True",      labelpad=10)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fname = name.replace(" ", "_").lower()
    cm_path = os.path.join(PLOTS_DIR, f"cm_{fname}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {cm_path}")

# ── 6. Model Comparison Bar Chart ─────────────────────────────────────────────
print("\n[PLOT] Generating model comparison chart...")

summary_df = pd.DataFrame(summary)
x      = np.arange(len(summary_df))
width  = 0.35
colors = ["#00e5ff", "#ff6ec7"]

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width / 2, summary_df["Accuracy"], width,
            label="Accuracy", color=colors[0], edgecolor="#000", linewidth=0.5)
b2 = ax.bar(x + width / 2, summary_df["F1_Macro"], width,
            label="F1-Macro", color=colors[1], edgecolor="#000", linewidth=0.5, alpha=0.85)

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=9, color="#ccc",
    )

ax.set_xticks(x)
ax.set_xticklabels(summary_df["Model"], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison - Accuracy & F1-Macro", fontsize=13, color=ACCENT, pad=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
comparison_path = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[PLOT] Saved -> {comparison_path}")

# ── 7. Save Results to Text File ──────────────────────────────────────────────
with open(RESULTS_PATH, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  HUMAN ACTION DETECTION - ML MODEL RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    for name, report in reports:
        f.write(f"\n{'-' * 60}\n")
        f.write(f"  {name}\n")
        f.write(f"{'-' * 60}\n")
        f.write(report + "\n")

print(f"\n[SAVE] Results saved -> {RESULTS_PATH}")

# ── 8. Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))
best = summary_df.loc[summary_df["F1_Macro"].idxmax()]
print(f"\n  Best Model : {best['Model']}  (F1-macro = {best['F1_Macro']:.4f})")
print("\n[DONE] Models complete. Run 04_cnn.py next.")
print("=" * 60)