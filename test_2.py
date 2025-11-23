#!/usr/bin/env python3

import numpy as np
import pandas as pd
import joblib
import os

# ==========================================================
# 1. CONSTANT PATHS (edit only if needed)
# ==========================================================

CSV_PATH = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/cleaned_data.csv"
MODEL_PATH = "mouse_model_xgb_5.pkl"
OUT_PATH = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/cleaned_data_scored_5.csv"

# ==========================================================
# 2. 11-FEATURE SET for model _5
# ==========================================================

FEATURES_11 = [
    "jitter_index",
    "speed_mean",
    "speed_std",
    "speed_max",
    "acc_mean",
    "acc_std",
    "acc_max",
    "jerk_mean",
    "jerk_std",
    "jerk_max",
    "curvature",
]

# ==========================================================
# 3. Load model
# ==========================================================

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[INFO] Loading model from {model_path}")
    return joblib.load(model_path)

# ==========================================================
# 4. Load dataset
# ==========================================================

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"[INFO] Loading CSV from {csv_path}")
    return pd.read_csv(csv_path)

# ==========================================================
# 5. Score rows
# ==========================================================

def score_dataframe(df, model):
    # Ensure required features exist
    missing = [f for f in FEATURES_11 if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURES_11].astype(float).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scores = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["human_likeness_score"] = scores

    return df

# ==========================================================
# 6. MAIN
# ==========================================================

def main():
    print("\n=== TESTING MODEL XGB_5 ===\n")

    model = load_model(MODEL_PATH)
    df = load_data(CSV_PATH)

    df_scored = score_dataframe(df, model)

    df_scored.to_csv(OUT_PATH, index=False)
    print(f"\n[INFO] Saved results to {OUT_PATH}")

    # Print summary
    print("\n=== human_likeness_score Summary ===")
    print(df_scored["human_likeness_score"].describe())


if __name__ == "__main__":
    main()