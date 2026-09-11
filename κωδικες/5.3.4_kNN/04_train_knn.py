from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.5_Feature_Selection"
RESULTS_DIR = ROOT / "results" / "5.3.4_kNN"
MODELS_DIR = ROOT / "models" / "5.3.4_kNN"
RANDOM_STATE = 42
MAX_TUNING_ROWS = 50_000
BATCH_SIZE = 10_000
K_VALUES = [3, 5, 7, 9, 11, 15, 21]
WEIGHT_VALUES = ["uniform", "distance"]


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


def build_model(neighbors: int, weights: str, n_jobs: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=neighbors,
        weights=weights,
        metric="minkowski",
        p=2,
        algorithm="kd_tree",
        leaf_size=40,
        n_jobs=n_jobs,
    )


def predict_in_batches(model: KNeighborsClassifier, x_test: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.empty(len(x_test), dtype=np.float64)
    predicted_labels = np.empty(len(x_test), dtype=np.int64)
    total_batches = (len(x_test) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_number, start in enumerate(range(0, len(x_test), BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, len(x_test))
        batch_probabilities = model.predict_proba(x_test[start:end])
        probabilities[start:end] = batch_probabilities[:, 1]
        predicted_labels[start:end] = model.classes_[np.argmax(batch_probabilities, axis=1)]
        if batch_number % 10 == 0 or batch_number == total_batches:
            print(f"{name}: prediction batch {batch_number}/{total_batches}")
    return predicted_labels, probabilities


def train_and_evaluate(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    tuning_indices = stratified_sample_indices(y_train, MAX_TUNING_ROWS)
    x_tuning = x_train[tuning_indices]
    y_tuning = y_train[tuning_indices]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_model(5, "uniform", n_jobs=1),
        param_grid={"n_neighbors": K_VALUES, "weights": WEIGHT_VALUES},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
    )
    tuning_start = time.perf_counter()
    search.fit(x_tuning, y_tuning)
    tuning_seconds = time.perf_counter() - tuning_start
    selected_neighbors = int(search.best_params_["n_neighbors"])
    selected_weights = str(search.best_params_["weights"])
    model = build_model(selected_neighbors, selected_weights, n_jobs=-1)
    training_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - training_start
    prediction_start = time.perf_counter()
    predicted_labels, probabilities = predict_in_batches(model, x_test, name)
    prediction_seconds = time.perf_counter() - prediction_start
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)
    matrix = confusion_matrix(y_test, predicted_labels)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    candidate_scores = []
    for parameters, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        candidate_scores.append({"parameters": parameters, "mean_f1_score": float(score)})
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
        "model": "k-Nearest Neighbors",
        "training_rows": int(x_train.shape[0]),
        "testing_rows": int(x_test.shape[0]),
        "input_features": int(x_train.shape[1]),
        "cross_validation": {
            "folds": 3,
            "scoring": "f1",
            "tuning_rows": int(len(tuning_indices)),
            "candidate_count": len(K_VALUES) * len(WEIGHT_VALUES),
            "candidate_scores": candidate_scores,
            "selected_neighbors": selected_neighbors,
            "selected_weights": selected_weights,
            "tuning_seconds": tuning_seconds,
        },
        "metrics": metrics,
        "decision_rule": "argmax_probability",
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
    joblib.dump(model, MODELS_DIR / f"{name}_knn.joblib")
    return result


def main() -> None:
    all_results = {}
    for name in ("CICIDS2017", "UNSW_NB15"):
        print(f"\n{name}: k-Nearest Neighbors")
        all_results[name] = train_and_evaluate(name)
        result = all_results[name]
        print(f"Selected neighbors: {result['cross_validation']['selected_neighbors']}")
        print(f"Selected weights: {result['cross_validation']['selected_weights']}")
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "knn_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults: {RESULTS_DIR}")
    print(f"Models: {MODELS_DIR}")


if __name__ == "__main__":
    main()
