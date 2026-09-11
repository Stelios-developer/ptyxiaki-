from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.3_Feature_Normalization"
OUTPUT_DIR = ROOT / "data" / "4.2.4_Class_Balancing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def counts(values: np.ndarray) -> dict[str, int]:
    labels, frequencies = np.unique(values, return_counts=True)
    return {str(int(label)): int(frequency) for label, frequency in zip(labels, frequencies)}


def process_dataset(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    labels, frequencies = np.unique(y_train, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"Το {name} δεν έχει δύο κλάσεις.")
    target_size = int(frequencies.min())
    generator = np.random.default_rng(RANDOM_STATE)
    selected = []
    for label in labels:
        indices = np.flatnonzero(y_train == label)
        selected.append(generator.choice(indices, size=target_size, replace=False))
    selected_indices = np.concatenate(selected)
    generator.shuffle(selected_indices)
    x_train_balanced = x_train[selected_indices]
    y_train_balanced = y_train[selected_indices]
    np.save(OUTPUT_DIR / f"{name}_X_train.npy", x_train_balanced)
    np.save(OUTPUT_DIR / f"{name}_y_train.npy", y_train_balanced)
    np.save(OUTPUT_DIR / f"{name}_X_test.npy", x_test)
    np.save(OUTPUT_DIR / f"{name}_y_test.npy", y_test)
    return {
        "before": counts(y_train),
        "after": counts(y_train_balanced),
        "balanced_training_shape": [int(x_train_balanced.shape[0]), int(x_train_balanced.shape[1])],
        "testing_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "testing_counts": counts(y_test),
    }


def main() -> None:
    report = {"CICIDS2017": process_dataset("CICIDS2017"), "UNSW_NB15": process_dataset("UNSW_NB15")}
    report_path = OUTPUT_DIR / "balancing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report.items():
        print(f"\n{name}")
        print(f"Before: {result['before']}")
        print(f"After: {result['after']}")
        print(f"Balanced training shape: {tuple(result['balanced_training_shape'])}")
        print(f"Testing shape: {tuple(result['testing_shape'])}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
