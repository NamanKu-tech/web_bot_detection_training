import os
import numpy as np
import pandas as pd
import joblib
import re

# ------------------------------
# 1. Feature Extraction Function
# (must match EXACT function used during training)
# ------------------------------

def extract_features(times, xs_norm, ys_norm):
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    # basic stats
    n_points = len(xs)
    duration = times[-1] - times[0] if n_points > 1 else 0.0

    # path length
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx*dx + dy*dy)
    path_length = dist.sum()

    # net displacement
    displacement = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)

    straightness = displacement / path_length if path_length > 0 else 0

    # dt features
    dts = np.diff(times)
    dt_mean = dts.mean() if len(dts) else 0
    dt_std  = dts.std()  if len(dts) else 0
    dt_max  = dts.max()  if len(dts) else 0

    # speeds
    speeds = dist / dts if len(dist) else np.array([0])
    speed_mean = speeds.mean()
    speed_std  = speeds.std()
    speed_max  = speeds.max()

    # acceleration
    acc = np.diff(speeds) if len(speeds) > 1 else np.array([0])
    acc_mean = acc.mean()
    acc_std  = acc.std()
    acc_max  = acc.max()

    # jerk
    jerk = np.diff(acc) if len(acc) > 1 else np.array([0])
    jerk_mean = jerk.mean()
    jerk_std  = jerk.std()
    jerk_max  = jerk.max()

    # jitter index
    if len(xs) > 2:
        jitter = np.abs(np.diff(xs,2)).sum() + np.abs(np.diff(ys,2)).sum()
    else:
        jitter = 0.0

    # curvature approximation
    curvature = 0.0
    if len(xs) > 2:
        for i in range(1, len(xs)-1):
            v1 = np.array([xs[i]-xs[i-1], ys[i]-ys[i-1]])
            v2 = np.array([xs[i+1]-xs[i], ys[i+1]-ys[i]])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                dot = np.dot(v1, v2) / (n1*n2)
                dot = np.clip(dot, -1, 1)
                angle = np.arccos(dot)
                curvature += angle

    return {
        "duration": duration,
        "path_length": path_length,
        "displacement": displacement,
        "straightness": straightness,
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
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "dt_max": dt_max,
        "curvature": curvature,
        "n_points": n_points,
    }


# ------------------------------
# 2. Parse your dataset file
# ------------------------------

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


# ------------------------------
# 3. Main Testing Logic
# ------------------------------

def main():
    # Load model
    model = joblib.load("mouse_model_xgb.pkl")

    # Root folder containing all test traces (will be scanned recursively)
    root_dir = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/test_set"

    feature_cols = [
        "duration", "path_length", "displacement", "straightness",
        "jitter_index", "speed_mean", "speed_std", "speed_max",
        "acc_mean", "acc_std", "acc_max",
        "jerk_mean", "jerk_std", "jerk_max",
        "dt_mean", "dt_std", "dt_max",
        "curvature", "n_points"
    ]

    results = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(dirpath, fname)

            try:
                # Load raw movements
                times, xs, ys, vw, vh = load_mouse_trace(file_path)

                # Normalize
                xs_norm = xs / vw
                ys_norm = ys / vh

                # Extract features
                feats = extract_features(times, xs_norm, ys_norm)

                # Build feature vector in correct order
                X = np.array([[feats[c] for c in feature_cols]])

                # Predict
                score = model.predict_proba(X)[0, 1]

                results.append({
                    "file": file_path,
                    "human_likeness_score": score
                })

            except Exception as e:
                # Log errors but continue with other files
                print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame and save/print
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("test_set_scores.csv", index=False)
    print("\nSaved scores to test_set_scores.csv")


if __name__ == "__main__":
    main()