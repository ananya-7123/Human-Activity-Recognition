"""
01_preprocessing.py
-------------------
Step 1 of the MHEALTH Human Action Detection Pipeline.
Loads raw data, cleans it, scales features, and saves
processed train/test splits + scaler to disk.

Run:
    python 01_preprocessing.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── 2. Config ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw", "mhealth_raw_data.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")

TARGET_COL    = "Activity"
DROP_COLS     = ["subject"]
NULL_ACTIVITY = 0
TEST_SIZE     = 0.20
SAMPLE_FRAC   = 0.3        # use 0.3 for fast runs, 1.0 for full dataset
RANDOM_STATE  = 42

# ── 3. Load Data ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — PREPROCESSING")
print("=" * 60)

print(f"\n[LOAD] Reading: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH)
print(f"[LOAD] Shape after load : {df.shape}")
print(f"[LOAD] Columns          : {list(df.columns)}")

# ── 4. Drop Null Activity rows (class 0 = no activity) ───────────────────────
before = len(df)
df = df[df[TARGET_COL] != NULL_ACTIVITY].reset_index(drop=True)
print(f"\n[PREP] Dropped null-activity rows : {before - len(df):,}  |  Remaining: {len(df):,}")

# ── 5. Drop non-feature columns ───────────────────────────────────────────────
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"[PREP] Dropped columns            : {cols_to_drop}")

# ── 6. Missing Value Check ────────────────────────────────────────────────────
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"[PREP] Missing values found: {missing} — imputing with median...")
    for col in df.columns[df.isnull().any()]:
        df[col].fillna(df[col].median(), inplace=True)
else:
    print("[PREP] No missing values found — skipping imputation.")

# ── 7. Stratified Downsample (optional, for faster dev runs) ─────────────────
if SAMPLE_FRAC < 1.0:
    df = df.groupby(TARGET_COL, group_keys=False).apply(
        lambda x: x.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    ).reset_index(drop=True)
    print(f"[PREP] Sampled {SAMPLE_FRAC * 100:.0f}% → {len(df):,} rows")

# ── 8. Feature / Target Split ─────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print(f"\n[PREP] Features : {X.shape[1]}  |  Classes : {y.nunique()}  |  Samples : {len(X):,}")
print(f"[PREP] Class distribution:\n{y.value_counts().sort_index()}")

# ── 9. Train / Test Split (stratified) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)
print(f"\n[PREP] Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

# ── 10. Standard Scaling ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)
print("[PREP] Features standardized — mean=0, std=1.")

# ── 11. Save Processed Data ───────────────────────────────────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)

train_df = X_train_scaled.copy()
train_df[TARGET_COL] = y_train.values
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)

test_df = X_test_scaled.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

joblib.dump(scaler, SCALER_PATH)

print(f"\n[SAVE] train.csv → {PROCESSED_DIR}/train.csv")
print(f"[SAVE] test.csv  → {PROCESSED_DIR}/test.csv")
print(f"[SAVE] scaler    → {SCALER_PATH}")

print("\n[DONE] Preprocessing complete. Run 02_eda.py next.")
print("=" * 60)