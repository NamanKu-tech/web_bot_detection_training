import os
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# 1. Feature Extraction
# ------------------------------

def extract_features(times, xs_norm, ys_norm):
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    # For very short traces → fallback
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

    # dt (needed internally but NOT used as final feature)
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


# ------------------------------
# 2. Trace Loader
# ------------------------------

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
        raise ValueError(f"Missing resolution in {path}")

    return np.array(times), np.array(xs), np.array(ys), vw, vh


# ------------------------------
# 3. Test Model
# ------------------------------

def main():
    model = joblib.load("mouse_model_xgb_5.pkl")

    root_dir = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/synthetic_bot_full"

    # FINAL 11 FEATURES (MATCH model_3)
    feature_cols = [
        "jitter_index", "speed_mean", "speed_std", "speed_max",
        "acc_mean", "acc_std", "acc_max",
        "jerk_mean", "jerk_std", "jerk_max",
        "curvature"
    ]

    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(dirpath, fname)

            try:
                times, xs, ys, vw, vh = load_mouse_trace(fpath)

                if len(times) < 3:
                    print(f"[SKIP] {fpath}: too few points")
                    continue

                xs_norm = xs / vw
                ys_norm = ys / vh

                feats = extract_features(times, xs_norm, ys_norm)

                X = np.array([[feats[c] for c in feature_cols]], dtype=float)
                X = np.nan_to_num(X, 0.0)

                score = model.predict_proba(X)[0, 1]
                results.append({"file": fpath, "human_likeness_score": score})

            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("test_set_scores.csv", index=False)
    print("Saved test_set_scores.csv")


if __name__ == "__main__":
    main()