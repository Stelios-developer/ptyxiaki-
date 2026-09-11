from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2_Preprocessing"
OUTPUT_DIR = ROOT / "data" / "4.2.2_Categorical_Encoding"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def create_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def process_dataset(name: str, train_file: str, test_file: str) -> dict:
    train = pd.read_csv(INPUT_DIR / train_file, low_memory=False)
    test = pd.read_csv(INPUT_DIR / test_file, low_memory=False)
    x_train = train.drop(columns="label")
    y_train = train["label"].to_numpy(dtype=np.int8)
    x_test = test.drop(columns="label")
    y_test = test["label"].to_numpy(dtype=np.int8)
    categorical_columns = x_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    transformers = [("numeric", "passthrough", numeric_columns)]
    if categorical_columns:
        transformers.insert(0, ("categorical", create_encoder(), categorical_columns))
    transformer = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    x_train_encoded = transformer.fit_transform(x_train)
    x_test_encoded = transformer.transform(x_test)
    feature_names = transformer.get_feature_names_out().tolist()

    x_train_encoded = np.asarray(x_train_encoded, dtype=np.float32)
    x_test_encoded = np.asarray(x_test_encoded, dtype=np.float32)
    np.save(OUTPUT_DIR / f"{name}_X_train.npy", x_train_encoded)
    np.save(OUTPUT_DIR / f"{name}_X_test.npy", x_test_encoded)
    np.save(OUTPUT_DIR / f"{name}_y_train.npy", y_train)
    np.save(OUTPUT_DIR / f"{name}_y_test.npy", y_test)
    (OUTPUT_DIR / f"{name}_feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(transformer, ARTIFACTS_DIR / f"{name}_encoder.joblib")
    (ARTIFACTS_DIR / f"{name}_raw_feature_names.json").write_text(
        json.dumps(x_train.columns.tolist(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "categorical_columns": categorical_columns,
        "input_features": len(x_train.columns),
        "output_features": int(x_train_encoded.shape[1]),
        "training_shape": [int(x_train_encoded.shape[0]), int(x_train_encoded.shape[1])],
        "testing_shape": [int(x_test_encoded.shape[0]), int(x_test_encoded.shape[1])],
        "encoder_artifact": str(ARTIFACTS_DIR / f"{name}_encoder.joblib"),
    }


def main() -> None:
    report = {
        "CICIDS2017": process_dataset("CICIDS2017", "CICIDS2017_train.csv", "CICIDS2017_test.csv"),
        "UNSW_NB15": process_dataset("UNSW_NB15", "UNSW_NB15_train.csv", "UNSW_NB15_test.csv"),
    }
    report_path = OUTPUT_DIR / "encoding_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report.items():
        print(f"\n{name}")
        print(f"Categorical columns: {result['categorical_columns']}")
        print(f"Input features: {result['input_features']}")
        print(f"Output features: {result['output_features']}")
        print(f"Training shape: {tuple(result['training_shape'])}")
        print(f"Testing shape: {tuple(result['testing_shape'])}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
