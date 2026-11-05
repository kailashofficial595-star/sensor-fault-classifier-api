import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.features import FEATURES, MODEL_COLUMNS, add_features

DATA_PATH = Path("data/ai4i2020.csv")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")

RENAME = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}


def make_model():
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )


def choose_threshold(X_train_fe, y_train) -> float:
    """Pick the probability cutoff with the best F1, using ONLY training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(make_model(), X_train_fe, y_train, cv=cv,
                              method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def main():
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME)
    X, y = df[FEATURES], df["machine_failure"]  # no ID or failure-type columns: no leakage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_fe = add_features(X_train)[MODEL_COLUMNS]
    X_test_fe = add_features(X_test)[MODEL_COLUMNS]

    threshold = choose_threshold(X_train_fe, y_train)
    model = make_model().fit(X_train_fe, y_train)

    # The test set is used exactly once, at the end.
    proba = model.predict_proba(X_test_fe)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, pred, digits=3))
    print(confusion_matrix(y_test, pred))

    metrics = {
        "threshold": round(threshold, 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        "n_train": len(X_train), "n_test": len(X_test),
        "n_test_failures": int(y_test.sum()),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "train_feature_means": X_train.mean().round(4).to_dict(),
        "train_feature_stds": X_train.std().round(4).to_dict(),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "columns": MODEL_COLUMNS}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Saved", MODEL_PATH, "and", METRICS_PATH)
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("test_") or k == "threshold"}, indent=2))


if __name__ == "__main__":
    main()