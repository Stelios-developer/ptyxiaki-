from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.5_Feature_Selection"
RESULTS_DIR = ROOT / "results" / "5.3.1_Logistic_Regression"
MODELS_DIR = ROOT / "models" / "5.3.1_Logistic_Regression"
RANDOM_STATE = 42
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def build_model(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def train_and_evaluate(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_model(1.0),
        param_grid={"C": C_VALUES},
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=False,
        return_train_score=False,
    )
    tuning_start = time.perf_counter()
    search.fit(x_train, y_train)
    tuning_seconds = time.perf_counter() - tuning_start
    best_c = float(search.best_params_["C"])
    model = build_model(best_c)
    training_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - training_start
    prediction_start = time.perf_counter()
    predicted_labels = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    prediction_seconds = time.perf_counter() - prediction_start
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)
    matrix = confusion_matrix(y_test, predicted_labels)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    cv_scores = {
        str(parameter["C"]): float(score)
        for parameter, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
    }
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "precision": float(precision_score(y_test, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_test, predicted_labels, zero_division=0)),
        "f1_score": float(f1_score(y_test, predicted_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "false_positive_rate": float(false_positive / (false_positive + true_negative)),
    }
    result = {
        "dataset": name,
        "model": "Logistic Regression",
        "training_rows": int(x_train.shape[0]),
        "testing_rows": int(x_test.shape[0]),
        "input_features": int(x_train.shape[1]),
        "cross_validation": {
            "folds": 3,
            "scoring": "f1",
            "candidate_C_values": C_VALUES,
            "mean_f1_scores": cv_scores,
            "selected_C": best_c,
            "tuning_seconds": tuning_seconds,
        },
        "metrics": metrics,
        "decision_threshold": 0.5,
        "confusion_matrix": matrix.tolist(),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        RESULTS_DIR / f"{name}_roc_curve.npz",
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        thresholds=thresholds,
    )
    joblib.dump(model, MODELS_DIR / f"{name}_logistic_regression.joblib")
    return result


def main() -> None:
    all_results = {}
    for name in ("CICIDS2017", "UNSW_NB15"):
        print(f"\n{name}: Logistic Regression")
        all_results[name] = train_and_evaluate(name)
        result = all_results[name]
        print(f"Selected C: {result['cross_validation']['selected_C']}")
        print(f"Accuracy: {result['metrics']['accuracy']:.6f}")
        print(f"Precision: {result['metrics']['precision']:.6f}")
        print(f"Recall: {result['metrics']['recall']:.6f}")
        print(f"F1-score: {result['metrics']['f1_score']:.6f}")
        print(f"ROC-AUC: {result['metrics']['roc_auc']:.6f}")
        print(f"Average Precision: {result['metrics']['average_precision']:.6f}")
        print(f"False Positive Rate: {result['metrics']['false_positive_rate']:.6f}")
        print(f"Confusion matrix: {result['confusion_matrix']}")
        print(f"Training time: {result['training_seconds']:.2f} seconds")
        print(f"Prediction time: {result['prediction_seconds']:.2f} seconds")
    (RESULTS_DIR / "logistic_regression_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults: {RESULTS_DIR}")
    print(f"Models: {MODELS_DIR}")


if __name__ == "__main__":
    main()
