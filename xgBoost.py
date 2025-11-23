import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib


def main():
    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    df_all = pd.read_csv("cleaned_data.csv")

    # These cols are metadata, not features
    non_feature_cols = ["label", "session_id", "phase", "subset"]

    # FINAL 14 NEUROMOTOR FEATURES
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
        "dt_mean",
        "dt_std",
        "dt_max",
        "curvature",
    ]

    # Sanity check
    missing = [c for c in feature_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing features in cleaned_data.csv: {missing}")

    print("\nUsing EXACTLY these 14 features:")
    print(feature_cols)

    X_all = df_all[feature_cols].astype(float).values
    y_all = df_all["label"].astype(int).values

    print("\nDataset:", X_all.shape)
    print("Class counts:", np.bincount(y_all))

    # --------------------------------------------------
    # 2. Session-based split
    # --------------------------------------------------
    unique_sids = df_all["session_id"].unique()
    session_labels = df_all.groupby("session_id")["label"].first()

    train_sids, val_sids = train_test_split(
        unique_sids,
        test_size=0.2,
        random_state=42,
        stratify=session_labels[unique_sids],
    )

    train_df = df_all[df_all["session_id"].isin(train_sids)]
    val_df = df_all[df_all["session_id"].isin(val_sids)]

    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df["label"].astype(int).values

    X_val = val_df[feature_cols].astype(float).values
    y_val = val_df["label"].astype(int).values

    # Clean NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)

    print("\nTrain:", X_train.shape, "| Val:", X_val.shape)

    # --------------------------------------------------
    # 3. Class imbalance handling
    # --------------------------------------------------
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(1, n_pos)

    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={n_neg}, pos={n_pos})")

    # --------------------------------------------------
    # 4. XGBoost model
    # --------------------------------------------------
    xgb_model = XGBClassifier(
        # model capacity
        n_estimators=450,
        max_depth=5,
        min_child_weight=5,

        # learning
        learning_rate=0.045,
        subsample=0.8,
        colsample_bytree=0.85,

        # regularisation
        reg_lambda=1.2,
        reg_alpha=0.0,

        # objective
        objective="binary:logistic",
        eval_metric="logloss",

        # class imbalance
        scale_pos_weight=scale_pos_weight,

        # system
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining XGBoost...")
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=40,
    )

    # --------------------------------------------------
    # 5. Feature importances
    # --------------------------------------------------
    print("\n=== FEATURE IMPORTANCE (14 NEUROMOTOR FEATURES) ===")
    importances = xgb_model.feature_importances_
    sorted_imps = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    for f, imp in sorted_imps:
        print(f"{f:15s}: {imp:.6f}")

    # --------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    print("\nValidation Metrics:")
    print("ROC-AUC:", roc_auc_score(y_val, y_val_proba))
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("F1:", f1_score(y_val, y_val_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred, digits=4))

    # --------------------------------------------------
    # 7. Save model
    # --------------------------------------------------
    out_path = "mouse_model_xgb_2.pkl"
    joblib.dump(xgb_model, out_path)

    print(f"\nSaved model → {out_path}")


if __name__ == "__main__":
    main()