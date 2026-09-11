from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
INPUT_DIR = ROOT / "data" / "4.2.1_Data_Cleaning"
OUTPUT_DIR = ROOT / "data" / "4.2_Preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def class_counts(data: pd.DataFrame) -> dict[str, int]:
    return {str(label): int(count) for label, count in data["label"].value_counts().sort_index().items()}


def split_cicids() -> dict:
    data = pd.read_csv(INPUT_DIR / "CIC_IDS2017_cleaned_deduplicated.csv", low_memory=False)
    data = data.rename(columns={"Label": "label"})
    data["label"] = (data["label"].astype(str).str.strip() != "BENIGN").astype(int)
    train, test = train_test_split(
        data,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_path = OUTPUT_DIR / "CICIDS2017_train.csv"
    test_path = OUTPUT_DIR / "CICIDS2017_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {
        "total_rows": len(data),
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_class_counts": class_counts(train),
        "testing_class_counts": class_counts(test),
        "training_file": str(train_path),
        "testing_file": str(test_path),
    }


def split_unsw() -> dict:
    data = pd.read_csv(INPUT_DIR / "UNSW_NB15_cleaned_deduplicated.csv", low_memory=False)
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    train, test = train_test_split(
        data,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_path = OUTPUT_DIR / "UNSW_NB15_train.csv"
    test_path = OUTPUT_DIR / "UNSW_NB15_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {
        "total_rows": len(data),
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_class_counts": class_counts(train),
        "testing_class_counts": class_counts(test),
        "training_file": str(train_path),
        "testing_file": str(test_path),
    }


def main() -> None:
    report = {"CICIDS2017": split_cicids(), "UNSW_NB15": split_unsw()}
    report_path = OUTPUT_DIR / "split_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report.items():
        print(f"\n{name}")
        print(f"Total rows: {result['total_rows']:,}")
        print(f"Training rows: {result['training_rows']:,}")
        print(f"Testing rows: {result['testing_rows']:,}")
        print(f"Training classes: {result['training_class_counts']}")
        print(f"Testing classes: {result['testing_class_counts']}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
