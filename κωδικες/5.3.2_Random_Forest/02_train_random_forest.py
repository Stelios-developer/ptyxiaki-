from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.5_Feature_Selection"
RESULTS_DIR = ROOT / "results" / "5.3.2_Random_Forest"
MODELS_DIR = ROOT / "models" / "5.3.2_Random_Forest"
RANDOM_STATE = 42
MAX_TUNING_ROWS = 80_000
CANDIDATE_PARAMETERS = [
    {"n_estimators": [150], "max_depth": [20], "min_samples_split": [2], "min_samples_leaf": [1], "max_features": ["sqrt"]},
    {"n_estimators": [200], "max_depth": [20], "min_samples_split": [5], "min_samples_leaf": [1], "max_features": ["sqrt"]},
    {"n_estimators": [200], "max_depth": [None], "min_samples_split": [2], "min_samples_leaf": [1], "max_features": ["sqrt"]},
    {"n_estimators": [250], "max_depth": [None], "min_samples_split": [5], "min_samples_leaf": [1], "max_features": ["sqrt"]},
    {"n_estimators": [200], "max_depth": [30], "min_samples_split": [2], "min_samples_leaf": [2], "max_features": [0.7]},
    {"n_estimators": [250], "max_depth": [30], "min_samples_split": [5], "min_samples_leaf": [2], "max_features": [0.7]},
]


def stratified_sample_indices(labels: np.ndarray, maximum_rows: int) -> np.ndarray:
    if len(labels) <= maximum_rows:
        return np.arange(len(labels))
    generator = np.random.default_rng(RANDOM_STATE)
    classes, counts = np.unique(labels, return_counts=True)
    target_counts = np.floor(counts / counts.sum() * maximum_rows).astype(int)
    target_counts[np.argmax(target_counts)] += maximum_rows - target_counts.sum()
    indices = []
    for label, target_count in zip(classes, target_counts):
        class_indices = np.flatnonzero(labels == label)
        indices.append(generator.choice(class_indices, size=int(target_count), replace=False))
    result = np.concatenate(indices)
    generator.shuffle(result)
    return result


def build_model(parameters: dict, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        min_samples_split=parameters["min_samples_split"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def train_and_evaluate(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    selected_report = json.loads((INPUT_DIR / f"{name}_selected_features.json").read_text(encoding="utf-8"))
    feature_names = [item["feature"] for item in selected_report["selected_features"]]
    tuning_indices = stratified_sample_indices(y_train, MAX_TUNING_ROWS)
    x_tuning = x_train[tuning_indices]
    y_tuning = y_train[tuning_indices]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_grid=CANDIDATE_PARAMETERS,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
    )
    tuning_start = time.perf_counter()
    search.fit(x_tuning, y_tuning)
    tuning_seconds = time.perf_counter() - tuning_start
    best_parameters = {
        "n_estimators": int(search.best_params_["n_estimators"]),
        "max_depth": None if search.best_params_["max_depth"] is None else int(search.best_params_["max_depth"]),
        "min_samples_split": int(search.best_params_["min_samples_split"]),
        "min_samples_leaf": int(search.best_params_["min_samples_leaf"]),
        "max_features": search.best_params_["max_features"],
    }
    model = build_model(best_parameters, n_jobs=-1)
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
    candidate_scores = []
    for parameters, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        candidate_scores.append({"parameters": parameters, "mean_f1_score": float(score)})
    importance_order = np.argsort(model.feature_importances_)[::-1]
    feature_importances = [
        {"feature": feature_names[int(index)], "importance": float(model.feature_importances_[int(index)])}
        for index in importance_order
    ]
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
        "model": "Random Forest",
        "training_rows": int(x_train.shape[0]),
        "testing_rows": int(x_test.shape[0]),
        "input_features": int(x_train.shape[1]),
        "cross_validation": {
            "folds": 3,
            "scoring": "f1",
            "tuning_rows": int(len(tuning_indices)),
            "candidate_count": len(CANDIDATE_PARAMETERS),
            "candidate_scores": candidate_scores,
            "selected_parameters": best_parameters,
            "tuning_seconds": tuning_seconds,
        },
        "metrics": metrics,
        "decision_threshold": 0.5,
        "confusion_matrix": matrix.tolist(),
        "feature_importances": feature_importances,
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
    joblib.dump(model, MODELS_DIR / f"{name}_random_forest.joblib")
    return result


def main() -> None:
    all_results = {}
    for name in ("CICIDS2017", "UNSW_NB15"):
        print(f"\n{name}: Random Forest")
        all_results[name] = train_and_evaluate(name)
        result = all_results[name]
        print(f"Selected parameters: {result['cross_validation']['selected_parameters']}")
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
        print("Top 5 features:")
        for item in result["feature_importances"][:5]:
            print(f"{item['feature']}: {item['importance']:.6f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "random_forest_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults: {RESULTS_DIR}")
    print(f"Models: {MODELS_DIR}")


if __name__ == "__main__":
    main()
