"""
02_eda.py
---------
Step 2 of the MHEALTH Human Action Detection Pipeline.
Loads raw data and generates exploratory analysis plots
saved to outputs/plots/.

Run:
    python 02_eda.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── 2. Config ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw", "mhealth_raw_data.csv")
PLOTS_DIR     = os.path.join("outputs", "plots")

ACTIVITY_LABELS = {
    0:  "Null",
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

ACCENT  = "#00e5ff"
PALETTE = sns.color_palette("cool", 13)

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

# ── 3. Load Data ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\n[LOAD] Reading: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH)
print(f"[LOAD] Shape : {df.shape}")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 4. Basic Info ─────────────────────────────────────────────────────────────
print(f"\n[INFO] Columns       : {list(df.columns)}")
print(f"\n[INFO] Dtypes:\n{df.dtypes}")
print(f"\n[INFO] Missing values:\n{df.isnull().sum()}")
print(f"\n[INFO] Descriptive stats:\n{df.describe().T.to_string()}")
print(f"\n[INFO] Activity value counts:\n{df['Activity'].value_counts().sort_index()}")

# ── 5. Plot 1 — Class Distribution ───────────────────────────────────────────
print("\n[EDA] Plotting class distribution...")

counts = df["Activity"].value_counts().sort_index()
labels = [ACTIVITY_LABELS.get(i, str(i)) for i in counts.index]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(labels, counts.values, color=PALETTE, edgecolor="#000", linewidth=0.5)
ax.set_title("Activity Class Distribution", fontsize=15, color=ACCENT, pad=14)
ax.set_xlabel("Activity", labelpad=10)
ax.set_ylabel("Sample Count", labelpad=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.4)
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 300,
        f"{int(bar.get_height()):,}",
        ha="center", va="bottom", fontsize=7, color="#ccc",
    )
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "class_distribution.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[EDA] Saved -> {path}")

# ── 6. Plot 2 — Missing Values Heatmap ───────────────────────────────────────
print("[EDA] Plotting missing values heatmap...")

sample = df.sample(min(5000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    sample.isnull(),
    cbar=False, yticklabels=False, ax=ax,
    cmap=["#1a1a1a", ACCENT],
)
ax.set_title("Missing Values Heatmap (5k-row sample)", fontsize=13, color=ACCENT, pad=12)
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "missing_values.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[EDA] Saved -> {path}")

# ── 7. Plot 3 — Correlation Heatmap ──────────────────────────────────────────
print("[EDA] Plotting correlation heatmap...")

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=False,
    cmap="coolwarm", center=0,
    linewidths=0.3, linecolor="#111",
    ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix", fontsize=14, color=ACCENT, pad=14)
plt.xticks(fontsize=7, rotation=45, ha="right")
plt.yticks(fontsize=7)
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[EDA] Saved -> {path}")

# ── 8. Plot 4 — Sensor Boxplots per Activity ─────────────────────────────────
print("[EDA] Plotting sensor boxplots...")

sensor_cols = ["alx", "aly", "alz"]
sample50k   = df.sample(min(50_000, len(df)), random_state=42)

for col in sensor_cols:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(16, 6))
    activity_order = sorted(sample50k["Activity"].unique())
    data_grouped   = [
        sample50k.loc[sample50k["Activity"] == act, col].values
        for act in activity_order
    ]
    bp = ax.boxplot(
        data_grouped,
        patch_artist=True,
        medianprops=dict(color=ACCENT, linewidth=2),
        whiskerprops=dict(color="#777"),
        capprops=dict(color="#777"),
        flierprops=dict(marker=".", color="#555", markersize=2),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    xlabels = [ACTIVITY_LABELS.get(a, str(a)) for a in activity_order]
    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Sensor: {col.upper()} — Distribution per Activity", fontsize=13, color=ACCENT, pad=12)
    ax.set_ylabel(col)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"boxplot_{col}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {path}")

# ── 9. Plot 5 — Subject x Activity Heatmap ───────────────────────────────────
print("[EDA] Plotting subject x activity heatmap...")

if "subject" in df.columns:
    pivot = df.groupby(["subject", "Activity"]).size().unstack(fill_value=0)
    pivot.columns = [ACTIVITY_LABELS.get(c, str(c)) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        pivot, annot=True, fmt=",d", cmap="YlOrRd",
        linewidths=0.5, linecolor="#111", ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Sample Count per Subject x Activity", fontsize=13, color=ACCENT, pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "subject_activity_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {path}")

# ── 10. Plot 6 — Feature Distributions (histogram grid) ──────────────────────
print("[EDA] Plotting feature distributions...")

feature_cols = [c for c in df.columns if c not in ["Activity", "subject"]]
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col].dropna(), bins=60, color=ACCENT, edgecolor="#000", alpha=0.8)
    axes[i].set_title(col, fontsize=10, color="#e0e0e0")
    axes[i].grid(axis="y", alpha=0.3)
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Feature Distributions", fontsize=15, color=ACCENT, y=1.01)
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "feature_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[EDA] Saved -> {path}")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")
print("[DONE] EDA complete. Run 03_models.py next.")
print("=" * 60)