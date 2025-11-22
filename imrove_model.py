import os
import json
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# -------------------------------------------------------------------
# 0. Feature extraction and trace loader (with NaN/Inf protection)
# -------------------------------------------------------------------

def extract_features(times, xs_norm, ys_norm):
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    n_points = len(xs)
    if n_points == 0 or len(times) == 0:
        # completely empty trace, all zeros
        return {
            "duration": 0.0,
            "path_length": 0.0,
            "displacement": 0.0,
            "straightness": 0.0,
            "jitter_index": 0.0,
            "speed_mean": 0.0,
            "speed_std": 0.0,
            "speed_max": 0.0,
            "acc_mean": 0.0,
            "acc_std": 0.0,
            "acc_max": 0.0,
            "jerk_mean": 0.0,
            "jerk_std": 0.0,
            "jerk_max": 0.0,
            "dt_mean": 0.0,
            "dt_std": 0.0,
            "dt_max": 0.0,
            "curvature": 0.0,
            "n_points": 0,
        }

    # basic stats
    duration = times[-1] - times[0] if n_points > 1 else 0.0

    # path length
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx * dx + dy * dy)
    path_length = dist.sum()

    # net displacement
    displacement = np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
    straightness = displacement / path_length if path_length > 0 else 0.0

    # dt features
    dts = np.diff(times)
    dt_mean = dts.mean() if len(dts) else 0.0
    dt_std = dts.std() if len(dts) else 0.0
    dt_max = dts.max() if len(dts) else 0.0

    # speeds (protect against division by zero -> Inf/NaN)
    if len(dist) and len(dts):
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = dist / dts
        speeds[~np.isfinite(speeds)] = 0.0
    else:
        speeds = np.array([0.0])

    speed_mean = speeds.mean()
    speed_std = speeds.std()
    speed_max = speeds.max()

    # acceleration
    acc = np.diff(speeds) if len(speeds) > 1 else np.array([0.0])
    acc[~np.isfinite(acc)] = 0.0
    acc_mean = acc.mean()
    acc_std = acc.std()
    acc_max = acc.max()

    # jerk
    jerk = np.diff(acc) if len(acc) > 1 else np.array([0.0])
    jerk[~np.isfinite(jerk)] = 0.0
    jerk_mean = jerk.mean()
    jerk_std = jerk.std()
    jerk_max = jerk.max()

    # jitter index
    if len(xs) > 2:
        jitter = np.abs(np.diff(xs, 2)).sum() + np.abs(np.diff(ys, 2)).sum()
    else:
        jitter = 0.0

    # curvature approximation
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
                angle = np.arccos(dot)
                curvature += angle

    return {
        "duration": float(duration),
        "path_length": float(path_length),
        "displacement": float(displacement),
        "straightness": float(straightness),
        "jitter_index": float(jitter),
        "speed_mean": float(speed_mean),
        "speed_std": float(speed_std),
        "speed_max": float(speed_max),
        "acc_mean": float(acc_mean),
        "acc_std": float(acc_std),
        "acc_max": float(acc_max),
        "jerk_mean": float(jerk_mean),
        "jerk_std": float(jerk_std),
        "jerk_max": float(jerk_max),
        "dt_mean": float(dt_mean),
        "dt_std": float(dt_std),
        "dt_max": float(dt_max),
        "curvature": float(curvature),
        "n_points": int(n_points),
    }


def load_mouse_trace(path):
    times = []
    xs = []
    ys = []
    viewport_w = viewport_h = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # resolution line
            if line.startswith("resolution:"):
                _, res = line.split(":")
                w, h = res.split(",")
                viewport_w = float(w)
                viewport_h = float(h)
                continue

            # event line: t,event,x,y
            parts = line.split(",")
            if len(parts) != 4:
                continue

            t, event, x, y = parts
            if event != "Move":   # ignore Pressed/Released
                continue

            times.append(float(t))
            xs.append(float(x))
            ys.append(float(y))

    if viewport_w is None:
        raise ValueError(f"Resolution line missing in {path}")

    return np.array(times), np.array(xs), np.array(ys), viewport_w, viewport_h


# -------------------------------------------------------------------
# 1. Dataset construction from sessions.json
# -------------------------------------------------------------------

feature_cols = [
    "duration", "path_length", "displacement", "straightness",
    "jitter_index", "speed_mean", "speed_std", "speed_max",
    "acc_mean", "acc_std", "acc_max",
    "jerk_mean", "jerk_std", "jerk_max",
    "dt_mean", "dt_std", "dt_max",
    "curvature", "n_points"
]


def resolve_path(base_dir, rel_or_abs):
    """
    Resolve paths listed in sessions.json.
    If relative, resolve relative to the JSON directory.
    """
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(base_dir, rel_or_abs))


def build_dataset_from_sessions(sessions_path):
    """
    Read sessions.json with structure something like:
        {
          "human": [... paths ...],
          "bot":   [... paths ...]
        }
    and build X (features) and y (labels).
    """
    with open(sessions_path, "r") as f:
        sessions = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(sessions_path))

    rows = []
    labels = []
    file_list = []

    # human → label 1
    for rel_path in sessions.get("human", []):
        full_path = resolve_path(base_dir, rel_path)
        try:
            times, xs, ys, vw, vh = load_mouse_trace(full_path)

            # skip traces with no or almost no movement
            if len(times) < 2 or len(xs) < 2:
                print(f"[SKIP HUMAN] {full_path} has insufficient movement data.")
                continue

            xs_norm = xs / vw
            ys_norm = ys / vh
            feats = extract_features(times, xs_norm, ys_norm)
            row = [feats[c] for c in feature_cols]
            rows.append(row)
            labels.append(1)
            file_list.append(full_path)
        except Exception as e:
            print(f"[HUMAN] Error processing {full_path}: {e}")

    # bot → label 0
    for rel_path in sessions.get("bot", []):
        full_path = resolve_path(base_dir, rel_path)
        try:
            times, xs, ys, vw, vh = load_mouse_trace(full_path)

            if len(times) < 2 or len(xs) < 2:
                print(f"[SKIP BOT] {full_path} has insufficient movement data.")
                continue

            xs_norm = xs / vw
            ys_norm = ys / vh
            feats = extract_features(times, xs_norm, ys_norm)
            row = [feats[c] for c in feature_cols]
            rows.append(row)
            labels.append(0)
            file_list.append(full_path)
        except Exception as e:
            print(f"[BOT] Error processing {full_path}: {e}")

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)

    # final safety: remove NaN/Inf from features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"Built dataset from {len(file_list)} traces "
        f"(humans={(y == 1).sum()}, bots={(y == 0).sum()})"
    )

    return X, y


# -------------------------------------------------------------------
# 2. Training logic (rebuilds / improves your XGB model)
# -------------------------------------------------------------------

def main():
    sessions_path = "sessions.json"   # adjust if needed

    print("Loading dataset from sessions.json...")
    X, y = build_dataset_from_sessions(sessions_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # Try to reuse hyperparameters from existing pickle model, if present
    existing_model_path = "mouse_model_xgb.pkl"
    params = None
    if os.path.exists(existing_model_path):
        try:
            old_model = joblib.load(existing_model_path)
            params = old_model.get_params()
            print("Loaded existing model; reusing its hyperparameters.")
        except Exception as e:
            print(f"Could not load existing model ({e}); using default params.")

    if params is None:
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",  # change to "gpu_hist" if you use GPU
            random_state=42,
        )
    else:
        # ensure some important defaults if missing
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")

    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    print("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
        early_stopping_rounds=50,  # ok to keep; only raises a warning
    )

    # Evaluation
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_prob)
    acc = accuracy_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)

    # Save improved model (overwrites old pickle)
    joblib.dump(model, existing_model_path)
    print(f"Saved updated model to {existing_model_path}")


if __name__ == "__main__":
    main()