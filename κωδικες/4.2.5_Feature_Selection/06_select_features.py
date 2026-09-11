from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.4_Class_Balancing"
FEATURE_NAMES_DIR = ROOT / "data" / "4.2.2_Categorical_Encoding"
OUTPUT_DIR = ROOT / "data" / "4.2.5_Feature_Selection"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
RANDOM_STATE = 42
MAX_SELECTION_ROWS = 80_000
MAX_VALIDATION_ROWS = 10_000
FOLDS = 5
REPEATS = 3
TREES = 100
SELECTED_FEATURES = 20


def stratified_sample_indices(labels: np.ndarray, size: int, random_state: int) -> np.ndarray:
    if len(labels) <= size:
        return np.arange(len(labels))
    generator = np.random.default_rng(random_state)
    classes, counts = np.unique(labels, return_counts=True)
    target_counts = np.floor(counts / counts.sum() * size).astype(int)
    target_counts[np.argmax(target_counts)] += size - target_counts.sum()
    parts = []
    for label, target_count in zip(classes, target_counts):
        candidates = np.flatnonzero(labels == label)
        parts.append(generator.choice(candidates, size=int(target_count), replace=False))
    indices = np.concatenate(parts)
    generator.shuffle(indices)
    return indices


def select_features(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    feature_names = json.loads((FEATURE_NAMES_DIR / f"{name}_feature_names.json").read_text(encoding="utf-8"))
    analysis_indices = stratified_sample_indices(y_train, MAX_SELECTION_ROWS, RANDOM_STATE)
    x_analysis = x_train[analysis_indices]
    y_analysis = y_train[analysis_indices]
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = []
    validation_sizes = []
    top_twenty_counts = np.zeros(x_train.shape[1], dtype=int)
    for fold_number, (fit_indices, validation_indices) in enumerate(splitter.split(x_analysis, y_analysis), start=1):
        x_fit = x_analysis[fit_indices]
        y_fit = y_analysis[fit_indices]
        x_validation = x_analysis[validation_indices]
        y_validation = y_analysis[validation_indices]
        evaluation_indices = stratified_sample_indices(y_validation, MAX_VALIDATION_ROWS, RANDOM_STATE + fold_number)
        x_evaluation = x_validation[evaluation_indices]
        y_evaluation = y_validation[evaluation_indices]
        model = RandomForestClassifier(
            n_estimators=TREES,
            random_state=RANDOM_STATE + fold_number,
            n_jobs=-1,
        )
        model.fit(x_fit, y_fit)
        importance = permutation_importance(
            model,
            x_evaluation,
            y_evaluation,
            scoring="f1",
            n_repeats=REPEATS,
            random_state=RANDOM_STATE + fold_number,
            n_jobs=-1,
        )
        values = importance.importances_mean
        fold_importances.append(values)
        validation_sizes.append(int(len(evaluation_indices)))
        top_twenty_counts[np.argsort(values)[::-1][:SELECTED_FEATURES]] += 1
        print(f"{name}: completed fold {fold_number}/{FOLDS}")
    importance_matrix = np.vstack(fold_importances)
    mean_importance = importance_matrix.mean(axis=0)
    standard_deviation = importance_matrix.std(axis=0)
    ranking = sorted(
        range(x_train.shape[1]),
        key=lambda index: (-mean_importance[index], standard_deviation[index], -top_twenty_counts[index]),
    )
    selected_indices = np.asarray(ranking[:SELECTED_FEATURES], dtype=int)
    x_train_selected = x_train[:, selected_indices]
    x_test_selected = x_test[:, selected_indices]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{name}_X_train.npy", x_train_selected)
    np.save(OUTPUT_DIR / f"{name}_X_test.npy", x_test_selected)
    np.save(OUTPUT_DIR / f"{name}_y_train.npy", y_train)
    np.save(OUTPUT_DIR / f"{name}_y_test.npy", y_test)
    selected_features = [
        {
            "rank": rank,
            "feature": feature_names[int(index)],
            "mean_f1_decrease": float(mean_importance[index]),
            "standard_deviation": float(standard_deviation[index]),
            "top_20_frequency": int(top_twenty_counts[index]),
        }
        for rank, index in enumerate(selected_indices, start=1)
    ]
    report = {
        "method": "Stratified 5-fold cross-validation with permutation importance scored by F1",
        "selection_training_rows": int(len(analysis_indices)),
        "validation_rows_per_fold": validation_sizes,
        "folds": FOLDS,
        "permutation_repeats": REPEATS,
        "candidate_features": int(x_train.shape[1]),
        "selected_feature_count": SELECTED_FEATURES,
        "training_shape": [int(x_train_selected.shape[0]), int(x_train_selected.shape[1])],
        "testing_shape": [int(x_test_selected.shape[0]), int(x_test_selected.shape[1])],
        "selected_features": selected_features,
    }
    joblib.dump(
        {
            "selected_indices": selected_indices,
            "input_feature_names": feature_names,
            "selected_feature_names": [item["feature"] for item in selected_features],
        },
        ARTIFACTS_DIR / f"{name}_feature_selector.joblib",
    )
    report["feature_selector_artifact"] = str(ARTIFACTS_DIR / f"{name}_feature_selector.joblib")
    (OUTPUT_DIR / f"{name}_selected_features.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    results = {}
    for name in ("CICIDS2017", "UNSW_NB15"):
        print(f"\n{name}: starting feature selection")
        results[name] = select_features(name)
        print(f"Training shape: {tuple(results[name]['training_shape'])}")
        print(f"Testing shape: {tuple(results[name]['testing_shape'])}")
        print("Selected features:")
        for item in results[name]["selected_features"]:
            print(f"{item['rank']}. {item['feature']}: {item['mean_f1_decrease']:.6f}")
    report_path = OUTPUT_DIR / "feature_selection_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
