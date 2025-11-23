import os
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# 1. Feature extraction — 11-feature version (NO dt)
# ==========================================================

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
            "curvature": 0.0,
        }

    # dt still needed internally
    dts = np.diff(times)
    dts[dts <= 0] = 1e-6

    # distance
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx * dx + dy * dy)

    # speed
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

    # jitter index
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
                dot = np.clip(dot, -1, 1)
                curvature += np.arccos(dot)

    return {
        "jitter_index": jitter,
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


# ==========================================================
# 2. Load Attentive Cursor CSV
# ==========================================================

def load_attentive_csv(path, assume_width=1920, assume_height=1080):
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

    df = df[df["event"].isin(["mousemove", "mouseover"])]
    df = df[(df["xpos"] > 0) | (df["ypos"] > 0)]

    if df.empty:
        return None

    time = df["timestamp"].to_numpy(dtype=float)
    xs = df["xpos"].to_numpy(dtype=float)
    ys = df["ypos"].to_numpy(dtype=float)

    return time, xs, ys, assume_width, assume_height


# ==========================================================
# 3. Run the model recursively
# ==========================================================

def run_folder(
    root_dir,
    model_path="/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/mouse_model_xgb_5.pkl",
):

    model = joblib.load(model_path)

    # MATCHES the 11-feature model EXACTLY
    feature_cols = [
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

    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(dirpath, fname)
            data = load_attentive_csv(fpath)
            if data is None:
                continue

            times, xs, ys, vw, vh = data

            if len(times) < 3:
                print(f"[SKIP] {fpath}: too few points")
                continue

            xs_norm = xs / vw
            ys_norm = ys / vh

            feats = extract_features(times, xs_norm, ys_norm)

            X = np.array([[feats[c] for c in feature_cols]], dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            score = model.predict_proba(X)[0, 1]
            results.append((fpath, score))

            print(f"{fpath}: {score:.4f}")

    df = pd.DataFrame(results, columns=["file", "human_likeness_score"])
    df.to_csv("attentive_scores.csv", index=False)
    print("\nSaved: attentive_scores.csv")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    folder = (
        "/Users/naman/sem_1_2025_26/scalable_computing/final_project/"
        "web_bot_detection_training/the-attentive-cursor-dataset-master-logs/logs"
    )
    run_folder(folder)