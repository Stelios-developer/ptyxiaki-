from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.2_Categorical_Encoding"
OUTPUT_DIR = ROOT / "data" / "4.2.3_Feature_Normalization"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def process_dataset(name: str) -> dict:
    x_train = np.load(INPUT_DIR / f"{name}_X_train.npy")
    x_test = np.load(INPUT_DIR / f"{name}_X_test.npy")
    y_train = np.load(INPUT_DIR / f"{name}_y_train.npy")
    y_test = np.load(INPUT_DIR / f"{name}_y_test.npy")
    scaler = MinMaxScaler()
    x_train_normalized = scaler.fit_transform(x_train).astype(np.float32)
    x_test_normalized = scaler.transform(x_test).astype(np.float32)
    np.save(OUTPUT_DIR / f"{name}_X_train.npy", x_train_normalized)
    np.save(OUTPUT_DIR / f"{name}_X_test.npy", x_test_normalized)
    np.save(OUTPUT_DIR / f"{name}_y_train.npy", y_train)
    np.save(OUTPUT_DIR / f"{name}_y_test.npy", y_test)
    joblib.dump(scaler, ARTIFACTS_DIR / f"{name}_minmax_scaler.joblib")
    return {
        "training_shape": [int(x_train_normalized.shape[0]), int(x_train_normalized.shape[1])],
        "testing_shape": [int(x_test_normalized.shape[0]), int(x_test_normalized.shape[1])],
        "training_minimum": float(x_train_normalized.min()),
        "training_maximum": float(x_train_normalized.max()),
        "testing_minimum": float(x_test_normalized.min()),
        "testing_maximum": float(x_test_normalized.max()),
        "scaler_artifact": str(ARTIFACTS_DIR / f"{name}_minmax_scaler.joblib"),
    }


def main() -> None:
    report = {"CICIDS2017": process_dataset("CICIDS2017"), "UNSW_NB15": process_dataset("UNSW_NB15")}
    report_path = OUTPUT_DIR / "normalization_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report.items():
        print(f"\n{name}")
        print(f"Training shape: {tuple(result['training_shape'])}")
        print(f"Testing shape: {tuple(result['testing_shape'])}")
        print(f"Training range: [{result['training_minimum']:.6f}, {result['training_maximum']:.6f}]")
        print(f"Testing range: [{result['testing_minimum']:.6f}, {result['testing_maximum']:.6f}]")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
