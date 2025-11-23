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
# 2. Generic 11-feature extractor (same as in _3/_4)
# ==========================================================

def extract_features_11(times, xs_norm, ys_norm):
    """
    Compute the 11 neuromotor features used in mouse_model_xgb_3/_4/_5.

    times   : 1D array of timestamps (float)
    xs_norm : 1D array of x positions (normalised 0..1)
    ys_norm : 1D array of y positions (normalised 0..1)
    """
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    if len(xs) < 3 or len(times) < 3:
        return {k: 0.0 for k in FEATURE_COLS}

    # dt
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


# ==========================================================
# 3. Loader for your original .txt traces (real/synthetic)
# ==========================================================

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
# 4. Synthetic bots (same idea as _4, skewed by type)
# ==========================================================

def build_synthetic_bot_df(
    synth_root="/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/synthetic_bot_full",
):
    """
    Walk synthetic_bot_full and build a DataFrame of synthetic bots
    in the same style as cleaned_data.csv: 11 features + label + ids.

    We bias towards simpler bot types (more likely attackers).
    """

    if not os.path.isdir(synth_root):
        print(f"[INFO] Synthetic root '{synth_root}' not found; skipping synthetic bots.")
        return pd.DataFrame()

    # tweakable: how many from each type to include
    MAX_PER_TYPE = {
        "simple_linear": 800,
        "linear_with_pauses": 800,
        "polyline": 600,
        "curved_easing": 400,
        "noisy_humanish": 200,
    }

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
# 5. Attentive human logs loader (2900 CSVs, one session/file)
# ==========================================================

def load_attentive_csv(path):
    """
    Attentive dataset CSV: columns like
      cursor timestamp xpos ypos event xpath attrs extras

    We:
      - read with whitespace separator
      - keep only movement events: mousemove / mouseover
      - enforce x>0 or y>0
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=0,
        )
    except Exception as e:
        print(f"[READ ERROR] {path}: {e}")
        return None

    required = {"timestamp", "xpos", "ypos", "event"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {path}: missing {required - set(df.columns)}")
        return None

    # Movement events only
    df = df[df["event"].isin(["mousemove", "mouseover"])]
    df = df[(df["xpos"] > 0) | (df["ypos"] > 0)]

    if df.empty:
        return None

    df = df.sort_values("timestamp")

    times = df["timestamp"].to_numpy(dtype=float)
    xs = df["xpos"].to_numpy(dtype=float)
    ys = df["ypos"].to_numpy(dtype=float)

    # per-file min-max normalisation (no true viewport)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

    xs_norm = (xs - x_min) / x_range
    ys_norm = (ys - y_min) / y_range

    return times, xs_norm, ys_norm


def build_attentive_human_df(log_root, min_points=3):
    """
    Walk the 'logs' folder from the Attentive Cursor dataset, e.g.:

      the-attentive-cursor-dataset-master-logs/logs

    and build a human-only DataFrame of 11 features + label + ids.
    """
    if not os.path.isdir(log_root):
        print(f"[INFO] Attentive logs root '{log_root}' not found; skipping.")
        return pd.DataFrame()

    rows = []
    idx = 0

    for dirpath, _, filenames in os.walk(log_root):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(dirpath, fname)
            data = load_attentive_csv(fpath)
            if data is None:
                continue

            times, xs_norm, ys_norm = data
            if len(times) < min_points:
                continue

            feats = extract_features_11(times, xs_norm, ys_norm)
            row = {f: feats[f] for f in FEATURE_COLS}
            row["label"] = 1  # human
            row["session_id"] = f"attentive_{idx:05d}"
            row["phase"] = "attentive"
            row["subset"] = "attentive"
            rows.append(row)

            idx += 1

    if not rows:
        print("[INFO] No valid Attentive human sessions found.")
        return pd.DataFrame()

    df_att = pd.DataFrame(rows)
    print(f"[ATTENTIVE] Built Attentive human DF with {len(df_att)} rows.")
    return df_att


# ==========================================================
# 6. Main: cleaned_data + synthetic bots + attentive humans → _5
# ==========================================================

def main():
    # ---------------------------------------------
    # 6.1 Load original cleaned_data.csv
    # ---------------------------------------------
    df_real = pd.read_csv("/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/cleaned_data.csv")

    missing = [c for c in FEATURE_COLS if c not in df_real.columns]
    if missing:
        raise ValueError(f"Missing features in cleaned_data.csv: {missing}")

    # Ensure identifier columns exist
    for col in ["label", "session_id", "phase", "subset"]:
        if col not in df_real.columns:
            if col == "label":
                raise ValueError("cleaned_data.csv must contain 'label' column")
            if col == "session_id":
                df_real[col] = [f"real_{i}" for i in range(len(df_real))]
            else:
                df_real[col] = "real"

    # ---------------------------------------------
    # 6.2 Synthetic bots
    # ---------------------------------------------
    df_synth = build_synthetic_bot_df("/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/synthetic_bot_full")

    # ---------------------------------------------
    # 6.3 Attentive human logs (SET THIS PATH)
    # ---------------------------------------------
    ATTENTIVE_LOG_ROOT = (
        "/Users/naman/sem_1_2025_26/scalable_computing/final_project/"
        "web_bot_detection_training/the-attentive-cursor-dataset-master-logs/logs"
    )
    df_att = build_attentive_human_df(ATTENTIVE_LOG_ROOT)

    # ---------------------------------------------
    # 6.4 Combine all sources
    # ---------------------------------------------
    dfs = [df_real]
    if not df_synth.empty:
        dfs.append(df_synth)
    if not df_att.empty:
        dfs.append(df_att)

    df_all = pd.concat(dfs, ignore_index=True)
    print("\n=== Combined dataset shape ===")
    print(df_all.shape)

    y_all = df_all["label"].astype(int).values
    counts = np.bincount(y_all)
    print(f"Label distribution (0=bot, 1=human): {counts}")

    # ---------------------------------------------
    # 6.5 Session-level split
    # ---------------------------------------------
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
    # 6.6 Class imbalance
    # ---------------------------------------------
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nscale_pos_weight = {scale_pos_weight:.3f} (neg={n_neg}, pos={n_pos})")

    # ---------------------------------------------
    # 6.7 Train XGBoost model for _5
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
    # 6.8 Feature importances (11 features)
    # ---------------------------------------------
    importances = xgb_model.feature_importances_
    sorted_imps = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])

    print("\n=== FEATURE IMPORTANCE (11 features) ===")
    for f, imp in sorted_imps:
        print(f"{f:15s}: {imp:.5f}")

    # ---------------------------------------------
    # 6.9 Evaluation
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
    # 6.10 Save model as _5
    # ---------------------------------------------
    model_path = "mouse_model_xgb_5.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
