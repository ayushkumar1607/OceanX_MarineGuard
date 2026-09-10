"""
MarineGuard — Train AIS Anomaly Model
Random Forest classifier that flags suspicious vessel behavior.

Usage: python train_ais_anomaly.py
Output: models/ais_anomaly_model.pkl, reports/ais_anomaly_report.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ml_data", "ais_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ais_anomaly_model.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORT_PATH = os.path.join(REPORTS_DIR, "ais_anomaly_report.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURES = [
    "avg_speed_knots", "speed_variance", "max_speed_knots", "min_speed_knots",
    "course_changes", "loitering", "num_ais_gaps", "max_gap_duration_min",
    "total_gap_minutes", "heading_alignment",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found.")
        print("Run: python data/generate_ml_training_data.py")
        return

    print("Training AIS Anomaly Model (Random Forest)...")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df["is_suspicious"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 8,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": {
            f: float(v) for f, v in sorted(
                zip(FEATURES, model.feature_importances_),
                key=lambda x: -x[1],
            )
        },
    }

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Model saved   -> {MODEL_PATH}")
    print(f"[OK] Report saved  -> {REPORT_PATH}")
    print(f"     Accuracy  : {metrics['accuracy']:.4f}")
    print(f"     Precision : {metrics['precision']:.4f}")
    print(f"     Recall    : {metrics['recall']:.4f}")
    print(f"     F1        : {metrics['f1']:.4f}")
    print(f"     ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"     Top 5 features:")
    for f, imp in list(metrics["feature_importances"].items())[:5]:
        print(f"       {f:<25} {imp:.4f}")


if __name__ == "__main__":
    main()