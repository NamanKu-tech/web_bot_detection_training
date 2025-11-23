import os
import json
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# ================================================================
# 1. Feature Extraction (compute dt, but only use non-dt features)
# ================================================================

def extract_features(times, xs_norm, ys_norm):
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    if len(xs) < 2 or len(times) < 2:
        return {
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
        }

    # dt
    dts = np.diff(times)
    dts[dts <= 0] = 1e-6
    dt_mean = dts.mean()
    dt_std = dts.std()
    dt_max = dts.max()

    # distance
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

    # jitter
    if len(xs) > 2:
        jitter = np.abs(np.diff(xs, 2)).sum() + np.abs(np.diff(ys, 2)).sum()
    else:
        jitter = 0.0

    # curvature
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
    }


# ================================================================
# 2. Trace Loader
# ================================================================

def load_mouse_trace(path):
    times, xs, ys = [], [], []
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
        raise ValueError(f"Resolution missing in {path}")

    return np.array(times), np.array(xs), np.array(ys), vw, vh


# ================================================================
# 3. Dataset Builder (sessions.json → X, y)
# ================================================================

# FINAL 11 FEATURE SET (dt_* REMOVED)
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


def resolve_path(base_dir, rel_or_abs):
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(base_dir, rel_or_abs))


def build_dataset_from_sessions(sessions_path):
    with open(sessions_path, "r") as f:
        sessions = json.load(f)

    base_dir = os.path.dirname(sessions_path)
    rows, labels = [], []

    def process_file(fpath, label):
        try:
            times, xs, ys, vw, vh = load_mouse_trace(fpath)
            if len(times) < 3:
                print(f"[SKIP] {fpath}: too few points")
                return
            xs_norm = xs / vw
            ys_norm = ys / vh
            feats = extract_features(times, xs_norm, ys_norm)
            rows.append([feats[c] for c in FEATURE_COLS])
            labels.append(label)
        except Exception as e:
            print(f"[ERR] {fpath}: {e}")

    for f in sessions.get("human", []):
        process_file(resolve_path(base_dir, f), 1)
    for f in sessions.get("bot", []):
        process_file(resolve_path(base_dir, f), 0)

    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels)

    print(f"Dataset built: {len(y)} traces (humans={y.sum()}, bots={(y == 0).sum()})")
    return X, y


# ================================================================
# 4. Train XGBoost using 11-feature dataset
# ================================================================

def main():
    print("Loading dataset...")
    X, y = build_dataset_from_sessions("sessions.json")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.3f} (neg={n_neg}, pos={n_pos})")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    print("Training XGB model (no dt features)...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50,
    )

    # Evaluate
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_val, probs)
    acc = accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)

    print(f"Validation AUC = {auc:.4f}")
    print(f"Validation ACC = {acc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # Feature importance
    print("\n=== FEATURE IMPORTANCE (11 features, no dt) ===")
    for name, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"{name:15s}: {imp:.5f}")

    # Save model as _3
    out_path = "mouse_model_xgb_3.pkl"
    joblib.dump(model, out_path)
    print(f"Saved model → {out_path}")


if __name__ == "__main__":
    main()