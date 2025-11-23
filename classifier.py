# classifier.py
import json
import math
from typing import List, Dict, Any, Optional

import numpy as np


FEATURE_NAMES = [
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


class XGBJsonModel:
    """
    Minimal XGBoost JSON model evaluator (binary:logistic) for use on the Pi.

    - Parses the 'model.json' exported by xgboost.Booster.save_model(...).
    - Uses the internal array-based tree representation:
        * split_indices
        * split_conditions
        * left_children
        * right_children
        * default_left
        * base_weights
    - No xgboost dependency.
    """

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            raw = json.load(f)

        learner = raw["learner"]
        gb = learner["gradient_booster"]
        model = gb["model"]

        # base_score as stored in the learner params (margin bias)
        learner_params = learner.get("learner_model_param", {})
        base_score_str = learner_params.get("base_score", "0.5")
        self.base_score = float(base_score_str)  # XGBoost will internally convert to margin

        # list of tree dicts
        self.trees: List[Dict[str, Any]] = self._extract_trees(model["trees"])

        # Number of features; we assume dense feature vectors
        self.num_feature = int(learner_params.get("num_feature", len(FEATURE_NAMES)))

    def _extract_trees(self, trees_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert each tree's arrays into a simpler structure we can traverse.
        """
        result = []
        for t in trees_json:
            # Each of these is a list indexed by node id
            split_indices = t.get("split_indices", [])
            split_conditions = t.get("split_conditions", [])
            left_children = t.get("left_children", [])
            right_children = t.get("right_children", [])
            default_left_raw = t.get("default_left", [])
            base_weights = t.get("base_weights", [])

            # Convert default_left from [0,1] ints to bool
            default_left = [bool(v) for v in default_left_raw]

            tree_struct = {
                "split_indices": split_indices,
                "split_conditions": split_conditions,
                "left_children": left_children,
                "right_children": right_children,
                "default_left": default_left,
                "base_weights": base_weights,
            }
            result.append(tree_struct)

        return result

    def _predict_row_margin(self, x: np.ndarray) -> float:
        """
        Compute the raw margin (before logistic) for a single row x.
        x must be a 1D numpy array of length num_feature with our chosen ordering.
        """
        margin = self.base_score

        for tree in self.trees:
            i = 0  # start at root
            while True:
                left = tree["left_children"][i]
                right = tree["right_children"][i]

                # Leaf node: both children == -1
                if left == -1 and right == -1:
                    margin += tree["base_weights"][i]
                    break

                feat_idx = tree["split_indices"][i]
                thresh = tree["split_conditions"][i]
                v = x[feat_idx]

                if np.isnan(v):
                    # follow default path for missing values
                    if tree["default_left"][i]:
                        i = left
                    else:
                        i = right
                else:
                    if v < thresh:
                        i = left
                    else:
                        i = right

        return margin

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (n_samples, n_features)
        Returns: shape (n_samples, 2) = [P(class=0), P(class=1)]
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        if n_features != self.num_feature:
            # We don't crash; just log in a comment. Make sure the caller uses
            # exactly the same feature ordering and count as during training.
            # In your setup you ensure 11 features, in the order FEATURE_NAMES.
            pass

        probs = np.zeros((n_samples, 2), dtype=float)

        for i in range(n_samples):
            margin = self._predict_row_margin(X[i])
            # logistic transform
            p1 = 1.0 / (1.0 + math.exp(-margin))
            probs[i, 1] = p1
            probs[i, 0] = 1.0 - p1

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Class predictions (0/1) using a probability threshold.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)