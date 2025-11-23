import os
import numpy as np
import pandas as pd
import joblib

# ==========================================================
# 1. EXACT 11-FEATURE SET (matches mouse_model_xgb_3)
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


def extract_features(times, xs_norm, ys_norm):
    """
    Compute the 11 neuromotor features used in mouse_model_xgb_3.

    times   : 1D array of timestamps (float)
    xs_norm : 1D array of x positions (normalised 0..1)
    ys_norm : 1D array of y positions (normalised 0..1)
    """
    times = np.array(times, dtype=float)
    xs = np.array(xs_norm, dtype=float)
    ys = np.array(ys_norm, dtype=float)

    if len(xs) < 3 or len(times) < 3:
        # Not enough points – return zeros
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
# 2. Loader for DFL CSV format
#    client timestamp,button,state,x,y
# ==========================================================

def load_dfl_session(path, min_points=3):
    """
    Load one DFL CSV file with columns:
      client timestamp,button,state,x,y

    Returns:
      times, xs_norm, ys_norm   OR   None if unusable
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[READ ERROR] {path}: {e}")
        return None

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"client_timestamp", "button", "state", "x", "y"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {path}: missing columns {required - set(df.columns)}")
        return None

    # Keep only movement events (state == 'Move', case-insensitive)
    df = df[df["state"].astype(str).str.lower() == "move"]

    if len(df) < min_points:
        print(f"[SKIP] {path}: not enough Move points ({len(df)})")
        return None

    # Sort by timestamp
    df = df.sort_values("client_timestamp")

    times = df["client_timestamp"].astype(float).to_numpy()
    xs = df["x"].astype(float).to_numpy()
    ys = df["y"].astype(float).to_numpy()

    # Per-file min–max normalisation (no global screen size)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

    xs_norm = (xs - x_min) / x_range
    ys_norm = (ys - y_min) / y_range

    return times, xs_norm, ys_norm


# ==========================================================
# 3. Run model over a DFL folder
# ==========================================================

def score_dfl_folder(
    root_dir,
    model_path="mouse_model_xgb_3.pkl",
    out_csv="dfl_scores.csv",
):
    """
    Recursively scan root_dir for *.csv, treat each file as one session,
    extract 11 features and score with mouse_model_xgb_3.
    """
    model = joblib.load(model_path)

    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(dirpath, fname)
            data = load_dfl_session(fpath)
            if data is None:
                continue

            times, xs_norm, ys_norm = data
            feats = extract_features(times, xs_norm, ys_norm)

            X = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            score = float(model.predict_proba(X)[0, 1])

            results.append(
                {
                    "file": fpath,
                    "n_points": len(times),
                    "human_likeness_score": score,
                }
            )
            print(f"{fpath}: {score:.4f} (n_points={len(times)})")

    if not results:
        print("No valid DFL sessions found.")
        return

    df_scores = pd.DataFrame(results)
    df_scores.to_csv(out_csv, index=False)

    print(f"\nSaved DFL scores to {out_csv}")
    print("\n=== DFL score summary ===")
    print(df_scores["human_likeness_score"].describe())


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    # Change this to your DFL root folder
    DFL_ROOT = "/Users/naman/sem_1_2025_26/scalable_computing/final_project/web_bot_detection_training/DFL"  # <-- set this

    MODEL_PKL = "mouse_model_xgb_5.pkl"
    OUT_CSV = "dfl_scores.csv"

    score_dfl_folder(DFL_ROOT, MODEL_PKL, OUT_CSV)