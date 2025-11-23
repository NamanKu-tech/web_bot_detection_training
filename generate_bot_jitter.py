"""
Synthetic "bot with jitter" mouse generator.

- Produces traces in the SAME format as your dataset:
    resolution:1920,1080
    t,event,x,y

- Paths are smooth and human-like (curved, variable speed, pauses)
  but generated algorithmically, with configurable jitter.

- Use the generated files as "bot" samples and run your XGBoost model
  on them to see if it still separates them from real humans.
"""

import os
import math
import random
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# Global config
# -------------------------------------------------------

RES_W = 1920
RES_H = 1080

# Typical human-ish timing (in milliseconds)
DT_MEAN = 12.0   # average time between points
DT_STD  = 5.0    # variation in time between points
DT_MIN  = 3.0    # minimum dt to avoid zeros


# -------------------------------------------------------
# Utility: timing + jitter helpers
# -------------------------------------------------------

def sample_dts(num_points: int) -> np.ndarray:
    """Sample a sequence of dt's (ms) with some variation, > DT_MIN."""
    dts = np.random.normal(DT_MEAN, DT_STD, size=num_points)
    dts = np.clip(dts, DT_MIN, None)
    return dts


def add_jitter(xs, ys, jitter_px=3.0):
    """
    Add small random jitter to a path.
    jitter_px is in pixels; note the path is already in pixels.
    """
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    jx = np.random.normal(0.0, jitter_px, size=len(xs))
    jy = np.random.normal(0.0, jitter_px, size=len(ys))

    xs_j = xs + jx
    ys_j = ys + jy

    # Clamp to viewport
    xs_j = np.clip(xs_j, 0, RES_W - 1)
    ys_j = np.clip(ys_j, 0, RES_H - 1)

    return xs_j, ys_j


def insert_pauses(xs, ys, pause_prob=0.1, max_pause_points=5):
    """
    Insert small pauses: repeated coordinates with time advancing
    but position unchanged. This is quite human-like.
    """
    new_xs = []
    new_ys = []

    for x, y in zip(xs, ys):
        new_xs.append(x)
        new_ys.append(y)

        if random.random() < pause_prob:
            # Repeat current point a few times
            k = random.randint(1, max_pause_points)
            for _ in range(k):
                new_xs.append(x)
                new_ys.append(y)

    return np.array(new_xs), np.array(new_ys)


# -------------------------------------------------------
# Path generators (human-ish but deterministic patterns)
# -------------------------------------------------------

def generate_line_path(num_points=80):
    """
    Straight-ish path from random start to random end,
    with some slight curvature for realism.
    """
    x0 = random.uniform(0.1 * RES_W, 0.9 * RES_W)
    y0 = random.uniform(0.1 * RES_H, 0.9 * RES_H)
    x1 = random.uniform(0.1 * RES_W, 0.9 * RES_W)
    y1 = random.uniform(0.1 * RES_H, 0.9 * RES_H)

    t = np.linspace(0.0, 1.0, num_points)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t

    # Add mild curvature (quadratic bump)
    curve_amp = random.uniform(-40, 40)
    ys = ys + curve_amp * (t * (1 - t))

    return xs, ys


def generate_circle_path(num_points=120):
    """
    Circular / elliptical motion with random centre and radius.
    """
    cx = random.uniform(0.3 * RES_W, 0.7 * RES_W)
    cy = random.uniform(0.3 * RES_H, 0.7 * RES_H)
    rx = random.uniform(100, 350)
    ry = random.uniform(80, 250)

    theta = np.linspace(0, 2 * math.pi, num_points)
    xs = cx + rx * np.cos(theta)
    ys = cy + ry * np.sin(theta)

    return xs, ys


def generate_target_overshoot_path(num_points=90):
    """
    Move towards a target, overshoot a bit, then correct back.
    This is very "human-like".
    """
    x0 = random.uniform(0.1 * RES_W, 0.3 * RES_W)
    y0 = random.uniform(0.2 * RES_H, 0.8 * RES_H)
    xt = random.uniform(0.7 * RES_W, 0.9 * RES_W)
    yt = random.uniform(0.2 * RES_H, 0.8 * RES_H)

    # Main move
    t1 = np.linspace(0.0, 1.0, int(num_points * 0.7))
    xs1 = x0 + (xt - x0) * t1
    ys1 = y0 + (yt - y0) * t1

    # Overshoot
    overshoot = random.uniform(10, 60)
    t2 = np.linspace(0.0, 1.0, int(num_points * 0.15))
    xs2 = xt + overshoot * t2
    ys2 = yt + overshoot * (t2 - 0.5)  # small vertical wiggle

    # Correct back
    t3 = np.linspace(0.0, 1.0, int(num_points * 0.15))
    xs3 = xs2[-1] + (xt - xs2[-1]) * t3
    ys3 = ys2[-1] + (yt - ys2[-1]) * t3

    xs = np.concatenate([xs1, xs2, xs3])
    ys = np.concatenate([ys1, ys2, ys3])

    return xs, ys


def generate_bot_path(pattern: str, jitter_px: float):
    """
    High-level generator: choose base path type + apply jitter + pauses.
    """
    if pattern == "line":
        xs, ys = generate_line_path()
    elif pattern == "circle":
        xs, ys = generate_circle_path()
    elif pattern == "target":
        xs, ys = generate_target_overshoot_path()
    else:
        # random choice if unknown
        xs, ys = random.choice(
            [generate_line_path, generate_circle_path, generate_target_overshoot_path]
        )()

    # Insert pauses (human-like micro-stops)
    xs, ys = insert_pauses(xs, ys, pause_prob=0.15, max_pause_points=4)

    # Add jitter
    xs, ys = add_jitter(xs, ys, jitter_px=jitter_px)

    return xs, ys


# -------------------------------------------------------
# Writing traces
# -------------------------------------------------------

def write_trace(path: Path, xs, ys):
    """
    Write a single trace file to disk in your dataset format.
    t is in milliseconds, starting at 0.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # sample dt for each step; length-1 because there are N-1 intervals
    dts = sample_dts(len(xs) - 1) if len(xs) > 1 else np.array([])
    times = np.zeros(len(xs), dtype=float)
    if len(xs) > 1:
        times[1:] = np.cumsum(dts)

    with open(path, "w") as f:
        # resolution line
        f.write(f"resolution:{RES_W},{RES_H}\n")

        # You could optionally add Pressed/Released rows, but your parser ignores them.
        # We'll just emit Move events for simplicity.
        for t, x, y in zip(times, xs, ys):
            f.write(f"{t:.1f},Move,{x:.1f},{y:.1f}\n")


# -------------------------------------------------------
# Dataset generator
# -------------------------------------------------------

def generate_bot_dataset(
    out_dir: str,
    n_traces: int = 100,
    jitter_range=(1.0, 8.0),
):
    """
    Generate a folder of synthetic "bot but human-like" traces.

    - jitter_range controls how "shaky" vs smooth the paths are.
    - Patterns are mixed: line, circle, target overshoot.

    Resulting files can be fed to your XGBoost model as candidate bots.
    """
    out_dir = Path(out_dir)
    patterns = ["line", "circle", "target"]

    for i in range(n_traces):
        pattern = random.choice(patterns)
        jitter_px = random.uniform(*jitter_range)

        xs, ys = generate_bot_path(pattern, jitter_px=jitter_px)

        fname = f"bot_jitter_{pattern}_{i:04d}.txt"
        fpath = out_dir / fname

        write_trace(fpath, xs, ys)
        print(f"Wrote {fpath} (pattern={pattern}, jitter={jitter_px:.2f}px)")


# -------------------------------------------------------
# Optional: quick scoring against your trained model
# -------------------------------------------------------

def optional_demo_scoring(model_path: str, samples_dir: str):
    """
    OPTIONAL helper: if you want, you can plug your existing
    extract_features + load_mouse_trace + XGBoost model here
    to see scores for the generated bots.

    This function is just a placeholder; you already have a
    working scoring script in your project.
    """
    import joblib
    from glob import glob

    from pathlib import Path

    model = joblib.load(model_path)
    from imrove_model import extract_features, load_mouse_trace, feature_cols  # adjust import

    for txt in glob(os.path.join(samples_dir, "*.txt"))[:10]:
        times, xs, ys, vw, vh = load_mouse_trace(txt)
        xs_norm = xs / vw
        ys_norm = ys / vh
        feats = extract_features(times, xs_norm, ys_norm)
        X = np.array([[feats[c] for c in feature_cols]])
        score = model.predict_proba(X)[0, 1]
        print(f"{txt}: human-likeness score = {score:.4f}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    # Example: generate 200 synthetic bot traces
    # in a folder you can later treat as "bot" samples.
    generate_bot_dataset(
        out_dir="synthetic_bot_jitter",  # change path if you like
        n_traces=200,
        jitter_range=(2.0, 10.0),       # low to high jitter
    )

    # If you want to immediately test them with your model,
    # uncomment and adapt the following lines:
    #
    # optional_demo_scoring(
    #     model_path="mouse_model_xgb.pkl",
    #     samples_dir="synthetic_bot_jitter",
    # )