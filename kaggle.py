import numpy as np
import pandas as pd
import joblib


# ==========================================================
# 1. EXACT 11-FEATURE EXTRACTOR (matches mouse_model_xgb_3)
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
# 2. Loader for this Kaggle dataset (kaggle.csv)
# ==========================================================

def load_kaggle_sessions(csv_path, move_event_type=2, min_points=3):
    """
    Load Kaggle mouse data and yield per-session traces.

    Assumes columns:
      uid, session_id, timestamp, event_type, screen_x, screen_y

    We:
      - Filter to move events (event_type == move_event_type)
      - Sort per session by timestamp
      - Min-max normalise x,y per session to [0,1]
      - Skip sessions with < min_points points

    Returns:
      list of dicts with keys:
        'uid', 'session_id', 'times', 'xs_norm', 'ys_norm', 'n_points'
    """
    df = pd.read_csv(csv_path)

    required = {"uid", "session_id", "timestamp", "event_type", "screen_x", "screen_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing expected columns in {csv_path}: {required - set(df.columns)}")

    # Keep only MOVE events
    df = df[df["event_type"] == move_event_type].copy()

    if df.empty:
        print("No move events found in Kaggle file.")
        return []

    # Sort globally to make groupby diffs well-behaved if needed
    df = df.sort_values(["session_id", "timestamp"])

    sessions = []

    for (session_id), g in df.groupby("session_id"):
        g = g.sort_values("timestamp")

        if len(g) < min_points:
            continue

        times = g["timestamp"].astype(float).to_numpy()
        xs = g["screen_x"].astype(float).to_numpy()
        ys = g["screen_y"].astype(float).to_numpy()

        # Per-session min-max normalisation (no absolute screen resolution info)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

        xs_norm = (xs - x_min) / x_range
        ys_norm = (ys - y_min) / y_range

        uid = g["uid"].iloc[0]

        sessions.append(
            {
                "uid": uid,
                "session_id": session_id,
                "times": times,
                "xs_norm": xs_norm,
                "ys_norm": ys_norm,
                "n_points": len(times),
            }
        )

    print(f"Loaded {len(sessions)} Kaggle sessions with >= {min_points} move points.")
    return sessions


# ==========================================================
# 3. Scoring with your 11-feature XGBoost model
# ==========================================================

def score_kaggle(
    csv_path="kaggle.csv",
    model_path="mouse_model_xgb_3.pkl",
    out_csv="kaggle_scores.csv",
):
    # Load model
    model = joblib.load(model_path)

    # Load sessions
    sessions = load_kaggle_sessions(csv_path)

    rows = []

    for sess in sessions:
        feats = extract_features(sess["times"], sess["xs_norm"], sess["ys_norm"])

        X = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        score = float(model.predict_proba(X)[0, 1])

        rows.append(
            {
                "uid": sess["uid"],
                "session_id": sess["session_id"],
                "n_points": sess["n_points"],
                "human_likeness_score": score,
            }
        )

    if not rows:
        print("No sessions scored; output will be empty.")
        return

    df_scores = pd.DataFrame(rows)
    df_scores.to_csv(out_csv, index=False)

    print(f"\nSaved Kaggle scores to {out_csv}")
    print("\n=== Kaggle score summary ===")
    print(df_scores["human_likeness_score"].describe())
    print("\nTop 10 sessions by score:")
    print(
        df_scores.sort_values("human_likeness_score", ascending=False)
        .head(10)
        .to_string(index=False)
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    # Adjust these paths as needed for your environment
    KAGGLE_CSV = "kaggle.csv"              # path to the Kaggle file
    MODEL_PKL  = "mouse_model_xgb_5.pkl"   # your 11-feature model
    OUT_CSV    = "kaggle_scores.csv"

    score_kaggle(KAGGLE_CSV, MODEL_PKL, OUT_CSV)