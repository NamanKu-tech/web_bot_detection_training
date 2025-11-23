import os
import numpy as np
import pandas as pd
import joblib


# ==============================================================
# 1. EXACT 11-FEATURE SET (same as model_xgb_3)
# ==============================================================

FEATURE_COLS = [
    "jitter_index",
    "speed_mean", "speed_std", "speed_max",
    "acc_mean", "acc_std", "acc_max",
    "jerk_mean", "jerk_std", "jerk_max",
    "curvature"
]


# ==============================================================
# 2. Feature extraction (identical to your model)
# ==============================================================

def extract_features(times, xs_norm, ys_norm):
    times = np.array(times, dtype=float)
    xs    = np.array(xs_norm, dtype=float)
    ys    = np.array(ys_norm, dtype=float)

    if len(xs) < 3:
        # Not enough data points
        return {k: 0.0 for k in FEATURE_COLS}

    # dt
    dts = np.diff(times)
    dts[dts <= 0] = 1e-6

    # distance
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx*dx + dy*dy)

    speeds = dist / dts
    speeds[~np.isfinite(speeds)] = 0.0

    speed_mean = speeds.mean()
    speed_std  = speeds.std()
    speed_max  = speeds.max()

    acc = np.diff(speeds) if len(speeds) > 1 else np.array([0])
    acc[~np.isfinite(acc)] = 0.0

    acc_mean = acc.mean()
    acc_std  = acc.std()
    acc_max  = acc.max()

    jerk = np.diff(acc) if len(acc) > 1 else np.array([0])
    jerk[~np.isfinite(jerk)] = 0.0

    jerk_mean = jerk.mean()
    jerk_std  = jerk.std()
    jerk_max  = jerk.max()

    # jitter
    if len(xs) > 2:
        jitter_index = abs(np.diff(xs, 2)).sum() + abs(np.diff(ys, 2)).sum()
    else:
        jitter_index = 0.0

    # curvature
    curvature = 0.0
    for i in range(1, len(xs)-1):
        v1 = np.array([xs[i]-xs[i-1], ys[i]-ys[i-1]])
        v2 = np.array([xs[i+1]-xs[i], ys[i+1]-ys[i]])
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            dot = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            curvature += np.arccos(np.clip(dot, -1, 1))

    return {
        "jitter_index": jitter_index,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_max": speed_max,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_max": acc_max,
        "jerk_mean": jerk_mean,
        "jerk_std": jerk_std,
        "jerk_max": jerk_max,
        "curvature": curvature,
    }


# ==============================================================
# 3. FIXED Balabit loader (correct headers + safe parsing)
# ==============================================================

def load_balabit_session(path):
    try:
        df = pd.read_csv(
            path,
            sep=",",
            engine="python",
            header=0,
        )
    except Exception as e:
        print(f"[READ ERROR] {path}: {e}")
        return None

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Must contain: record_timestamp, client_timestamp, button, state, x, y
    required = ["record_timestamp", "client_timestamp", "button", "state", "x", "y"]

    for req in required:
        if req not in df.columns:
            print(f"[SKIP] {path}: missing column '{req}' → columns: {df.columns.tolist()}")
            return None

    # Keep only movement
    df = df[df["state"] == "Move"]

    if df.empty:
        print(f"[SKIP] {path}: no Move events")
        return None

    times = df["record_timestamp"].astype(float).to_numpy()
    xs    = df["x"].astype(float).to_numpy()
    ys    = df["y"].astype(float).to_numpy()

    # No viewport → assume typical
    vw, vh = 1920.0, 1080.0

    return times, xs, ys, vw, vh


# ==============================================================
# 4. Score all Balabit sessions with your model
# ==============================================================

def evaluate_balabit(root_dir, model_path="mouse_model_xgb_5.pkl"):
    model = joblib.load(model_path)

    results = []

    for user in os.listdir(root_dir):
        user_dir = os.path.join(root_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for fname in os.listdir(user_dir):
            if not fname.startswith("session_"):
                continue

            fpath = os.path.join(user_dir, fname)
            data = load_balabit_session(fpath)

            if data is None:
                continue

            times, xs, ys, vw, vh = data

            xs_norm = xs / vw
            ys_norm = ys / vh
            feats = extract_features(times, xs_norm, ys_norm)

            X = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
            X = np.nan_to_num(X)

            score = model.predict_proba(X)[0, 1]

            results.append((user, fname, score))
            print(f"{user}/{fname}: {score:.4f}")

    df = pd.DataFrame(results, columns=["user", "session", "score"])
    df.to_csv("balabit_scores.csv", index=False)
    print("\nSaved: balabit_scores.csv")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    ROOT = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/Mouse-Dynamics-Challenge/training_files"
    evaluate_balabit(ROOT)