import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

# ==========================================================
# 1. FINAL 11-FEATURE SET (no dt, no duration/path)
# ==========================================================

FEATURE_COLS = [
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

NON_FEATURE_COLS = ["label", "session_id", "phase", "subset"]


# ==========================================================
# 2. Feature extraction + loader for .txt traces
#    (same format as your main dataset + synthetic bots)
# ==========================================================

def extract_features_11(times, xs_norm, ys_norm):
    """
    Compute the 11 neuromotor features used in mouse_model_xgb_3/_4.

    times   : 1D array of timestamps (float)
    xs_norm : 1D array of x positions (normalised 0..1)
    ys_norm : 1D array of y positions (normalised 0..1)
    """
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    if len(xs) < 3 or len(times) < 3:
        return {k: 0.0 for k in FEATURE_COLS}

    # Δt
    dts = np.diff(times)
    dts[dts <= 0] = 1e-6  # avoid zero/negative dt

    # distance between points
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx * dx + dy * dy)

    # speed
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = dist / dts
    speeds[~np.isfinite(speeds)] = 0.0

    speed_mean = speeds.mean()
    speed_std = speeds.std()
    speed_max = speeds.max()

    # acceleration
    if len(speeds) > 1:
        acc = np.diff(speeds)
        acc[~np.isfinite(acc)] = 0.0
    else:
        acc = np.array([0.0])

    acc_mean = acc.mean()
    acc_std = acc.std()
    acc_max = acc.max()

    # jerk
    if len(acc) > 1:
        jerk = np.diff(acc)
        jerk[~np.isfinite(jerk)] = 0.0
    else:
        jerk = np.array([0.0])

    jerk_mean = jerk.mean()
    jerk_std = jerk.std()
    jerk_max = jerk.max()

    # jitter index (second differences magnitude)
    if len(xs) > 2:
        jitter_index = np.abs(np.diff(xs, 2)).sum() + np.abs(np.diff(ys, 2)).sum()
    else:
        jitter_index = 0.0

    # curvature (sum of turning angles)
    curvature = 0.0
    if len(xs) > 2:
        for i in range(1, len(xs) - 1):
            v1 = np.array([xs[i] - xs[i - 1], ys[i] - ys[i - 1]])
            v2 = np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                dot = np.dot(v1, v2) / (n1 * n2)
                dot = np.clip(dot, -1.0, 1.0)
                curvature += np.arccos(dot)

    return {
        "jitter_index": float(jitter_index),
        "speed_mean": float(speed_mean),
        "speed_std": float(speed_std),
        "speed_max": float(speed_max),
        "acc_mean": float(acc_mean),
        "acc_std": float(acc_std),
        "acc_max": float(acc_max),
        "jerk_mean": float(jerk_mean),
        "jerk_std": float(jerk_std),
        "jerk_max": float(jerk_max),
        "curvature": float(curvature),
    }


def load_mouse_trace(path):
    """
    Your standard .txt format:

    resolution:1920,1080
    t,Move,x,y
    t,Move,x,y
    ...
    """
    times = []
    xs = []
    ys = []
    vw = vh = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("resolution:"):
                _, r = line.split(":")
                vw, vh = map(float, r.split(","))
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue

            t, event, x, y = parts
            if event != "Move":
                continue

            times.append(float(t))
            xs.append(float(x))
            ys.append(float(y))

    if vw is None:
        raise ValueError(f"Missing resolution in file {path}")

    return np.array(times), np.array(xs), np.array(ys), vw, vh


# ==========================================================
# 3. Build synthetic bot DataFrame from synthetic_bot_full/
#    with skewed usage per type (not all equal)
# ==========================================================

def build_synthetic_bot_df(
    synth_root="synthetic_bot_full",
):
    """
    Walk synthetic_bot_full and build a DataFrame of synthetic bots
    in the same style as cleaned_data.csv: 11 features + label + ids.

    We *do not* use all 800 per type; we bias towards simpler bots.
    """

    if not os.path.isdir(synth_root):
        print(f"[INFO] Synthetic root '{synth_root}' not found; skipping synthetic bots.")
        return pd.DataFrame()

    # How many from each type (you can tweak these)
    # You currently have 800 of each generated; we use fewer 'noisy_humanish'.
    MAX_PER_TYPE = {
        "simple_linear": 800,
        "linear_with_pauses": 800,
        "polyline": 600,
        "curved_easing": 400,
        "noisy_humanish": 200,
    }

    # Collect files by type (type = immediate parent directory)
    files_by_type = {}
    for dirpath, _, filenames in os.walk(synth_root):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(dirpath, fname)
            bot_type = os.path.basename(os.path.dirname(full_path))
            files_by_type.setdefault(bot_type, []).append(full_path)

    rows = []

    for bot_type, paths in files_by_type.items():
        max_n = MAX_PER_TYPE.get(bot_type, len(paths))
        if len(paths) > max_n:
            random.shuffle(paths)
            use_paths = paths[:max_n]
        else:
            use_paths = paths

        print(f"[SYNTH] Using {len(use_paths)} of {len(paths)} for type '{bot_type}'")

        for i, path in enumerate(use_paths):
            try:
                times, xs, ys, vw, vh = load_mouse_trace(path)
                if len(times) < 3:
                    continue

                xs_norm = xs / vw
                ys_norm = ys / vh

                feats = extract_features_11(times, xs_norm, ys_norm)

                row = {f: feats[f] for f in FEATURE_COLS}
                row["label"] = 0  # bot
                # unique synthetic session id
                row["session_id"] = f"synth_{bot_type}_{i:04d}"
                row["phase"] = "synthetic"
                row["subset"] = "synthetic"

                rows.append(row)

            except Exception as e:
                print(f"[SYNTH ERROR] {path}: {e}")

    if not rows:
        print("[INFO] No valid synthetic bot traces found.")
        return pd.DataFrame()

    df_synth = pd.DataFrame(rows)
    print(f"[SYNTH] Built synthetic bot DF with {len(df_synth)} rows.")
    return df_synth


# ==========================================================
# 4. Main training: cleaned_data.csv + synthetic bots → _4
# ==========================================================

def main():
    # ---------------------------------------------
    # 4.1 Load original cleaned_data.csv (used for _3)
    # ---------------------------------------------
    df_real = pd.read_csv("cleaned_data.csv")

    # Sanity check: features must exist in cleaned_data
    missing = [c for c in FEATURE_COLS if c not in df_real.columns]
    if missing:
        raise ValueError(f"Missing features in cleaned_data.csv: {missing}")

    # Ensure identifier columns exist (if not, create defaults)
    for col in ["label", "session_id", "phase", "subset"]:
        if col not in df_real.columns:
            if col == "label":
                raise ValueError("cleaned_data.csv must contain 'label' column")
            # create dummy values
            if col == "session_id":
                df_real[col] = [f"real_{i}" for i in range(len(df_real))]
            else:
                df_real[col] = "real"

    # ---------------------------------------------
    # 4.2 Build synthetic bots and concatenate
    # ---------------------------------------------
    df_synth = build_synthetic_bot_df("synthetic_bot_full")

    if not df_synth.empty:
        df_all = pd.concat([df_real, df_synth], ignore_index=True)
    else:
        df_all = df_real.copy()

    print("\n=== Combined dataset shape ===")
    print(df_all.shape)

    # Label distribution
    y_all = df_all["label"].astype(int).values
    counts = np.bincount(y_all)
    print(f"Label distribution (0=bot, 1=human): {counts}")

    # ---------------------------------------------
    # 4.3 Define features and session-level split
    # ---------------------------------------------
    X_all = df_all[FEATURE_COLS].astype(float).values

    unique_sids = df_all["session_id"].unique()
    session_labels = df_all.groupby("session_id")["label"].first()

    train_sids, test_sids = train_test_split(
        unique_sids,
        test_size=0.2,
        random_state=42,
        stratify=session_labels[unique_sids],
    )

    train_df = df_all[df_all["session_id"].isin(train_sids)]
    test_df = df_all[df_all["session_id"].isin(test_sids)]

    X_train = train_df[FEATURE_COLS].astype(float).values
    y_train = train_df["label"].astype(int).values

    X_valid = test_df[FEATURE_COLS].astype(float).values
    y_valid = test_df["label"].astype(int).values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nTrain shape:", X_train.shape, "| Valid shape:", X_valid.shape)

    # ---------------------------------------------
    # 4.4 Class imbalance
    # ---------------------------------------------
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nscale_pos_weight = {scale_pos_weight:.3f} (neg={n_neg}, pos={n_pos})")

    # ---------------------------------------------
    # 4.5 Train XGBoost model for _4
    # ---------------------------------------------
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        min_child_weight=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
        early_stopping_rounds=50,
    )

    # ---------------------------------------------
    # 4.6 Feature importances (11 features)
    # ---------------------------------------------
    importances = xgb_model.feature_importances_
    sorted_imps = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])

    print("\n=== FEATURE IMPORTANCE (11 features) ===")
    for f, imp in sorted_imps:
        print(f"{f:15s}: {imp:.5f}")

    # ---------------------------------------------
    # 4.7 Evaluation
    # ---------------------------------------------
    y_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_valid_proba >= 0.5).astype(int)

    print("\n=== Metrics on validation set ===")
    print("ROC AUC:", roc_auc_score(y_valid, y_valid_proba))
    print("Accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("F1 score:", f1_score(y_valid, y_valid_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, y_valid_pred))

    print("\nClassification report:")
    print(classification_report(y_valid, y_valid_pred, digits=4))

    # ---------------------------------------------
    # 4.8 Save model as _4
    # ---------------------------------------------
    model_path = "mouse_model_xgb_4.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()