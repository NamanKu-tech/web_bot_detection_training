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

    # Columns that are NOT features
    non_feature_cols = ["label", "session_id", "phase", "subset"]

    feature_cols = [c for c in df_all.columns if c not in non_feature_cols]
    print("Number of features:", len(feature_cols))
    print("Feature columns:", feature_cols)

    X_all = df_all[feature_cols].astype(float).values
    y_all = df_all["label"].astype(int).values

    print("X_all shape:", X_all.shape)
    print("y_all shape:", y_all.shape, "| class distribution:", np.bincount(y_all))

    # --------------------------------------------------
    # 2. Session-level train/valid split
    # --------------------------------------------------
    unique_sids = df_all["session_id"].unique()

    # stratify by session-level label
    session_labels = df_all.groupby("session_id")["label"].first()

    train_sids, test_sids = train_test_split(
        unique_sids,
        test_size=0.2,
        random_state=42,
        stratify=session_labels[unique_sids],
    )

    train_df = df_all[df_all["session_id"].isin(train_sids)]
    test_df = df_all[df_all["session_id"].isin(test_sids)]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_valid = test_df[feature_cols].values
    y_valid = test_df["label"].values

    print("Train shape:", X_train.shape, "| Valid shape:", X_valid.shape)

    # --------------------------------------------------
    # 3. Train XGBoost model
    # --------------------------------------------------
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )

    xgb_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 4. Feature importances
    # --------------------------------------------------
    print("\nTop 15 feature importances:")
    importances = xgb_model.feature_importances_
    for f, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]:
        print(f"{f}: {imp:.6f}")

    # --------------------------------------------------
    # 5. Evaluation on validation set
    # --------------------------------------------------
    y_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_valid_proba >= 0.5).astype(int)

    print("\nMetrics on validation set:")
    print("ROC AUC:", roc_auc_score(y_valid, y_valid_proba))
    print("Accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("F1 score:", f1_score(y_valid, y_valid_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, y_valid_pred))

    print("\nClassification report:")
    print(classification_report(y_valid, y_valid_pred, digits=4))

    # --------------------------------------------------
    # 6. Save model
    # --------------------------------------------------
    model_path = "mouse_model_xgb.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()