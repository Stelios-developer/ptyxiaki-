from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\stelios\Desktop\PTYX")
OUTPUT_DIR = ROOT / "data" / "4.2.1_Data_Cleaning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()
    return data


def clean_invalid_values(data: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    nan_rows = int(data.isna().any(axis=1).sum())
    numeric = data.select_dtypes(include=[np.number])
    infinity_rows = int(np.isinf(numeric.to_numpy()).any(axis=1).sum())

    data = data.replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna().copy()
    removed_union = before - len(data)
    return data, nan_rows, infinity_rows, removed_union


def remove_duplicates_and_conflicts(
    data: pd.DataFrame,
    label_column: str,
    excluded_columns: list[str],
) -> tuple[pd.DataFrame, int, int]:
    feature_columns = [
        column
        for column in data.columns
        if column != label_column and column not in excluded_columns
    ]
    if not feature_columns:
        raise ValueError("Δεν βρέθηκαν χαρακτηριστικά για τον έλεγχο διπλοτύπων.")

    feature_hash = pd.util.hash_pandas_object(data[feature_columns], index=False)
    label_count = pd.DataFrame({"hash": feature_hash, "label": data[label_column]}).groupby(
        "hash", sort=False
    )["label"].nunique()
    conflicting_hashes = label_count[label_count > 1].index
    conflicting_mask = feature_hash.isin(conflicting_hashes)
    conflicting_rows = int(conflicting_mask.sum())
    data = data.loc[~conflicting_mask].copy()

    duplicate_mask = data.duplicated(subset=feature_columns, keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].copy()

    removable = [column for column in excluded_columns if column in data.columns]
    return data.drop(columns=removable), duplicate_rows, conflicting_rows


def remove_redundant_feature_columns(
    data: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    feature_columns = [column for column in data.columns if column != label_column]
    constant_columns = [
        column for column in feature_columns if data[column].nunique(dropna=False) <= 1
    ]
    candidate_columns = [column for column in feature_columns if column not in constant_columns]
    signatures: dict[str, list[str]] = {}
    duplicate_columns: dict[str, str] = {}

    for column in candidate_columns:
        values = pd.util.hash_pandas_object(data[column], index=False).to_numpy()
        signature = hashlib.sha256(values.tobytes()).hexdigest()
        matched_column = None
        for previous_column in signatures.get(signature, []):
            if data[column].equals(data[previous_column]):
                matched_column = previous_column
                break
        if matched_column is None:
            signatures.setdefault(signature, []).append(column)
        else:
            duplicate_columns[column] = matched_column

    columns_to_remove = constant_columns + list(duplicate_columns)
    return data.drop(columns=columns_to_remove), constant_columns, duplicate_columns


def process_cicids() -> dict:
    source = ROOT / "CIC_IDS2017_full.csv"
    print("\nCICIDS2017: φόρτωση αρχικού αρχείου...", flush=True)
    data = normalize_columns(pd.read_csv(source, low_memory=False))
    if "Label" not in data.columns:
        raise ValueError("Δεν βρέθηκε η στήλη Label στο CICIDS2017.")
    data["Label"] = data["Label"].astype(str).str.strip()
    initial_rows = len(data)
    data, nan_rows, infinity_rows, invalid_removed = clean_invalid_values(data)
    data, duplicate_rows, conflicting_rows = remove_duplicates_and_conflicts(
        data=data,
        label_column="Label",
        excluded_columns=["Flow ID", "Timestamp"],
    )
    data, constant_columns, duplicate_columns = remove_redundant_feature_columns(
        data=data,
        label_column="Label",
    )
    output = OUTPUT_DIR / "CIC_IDS2017_cleaned_deduplicated.csv"
    data.to_csv(output, index=False)
    return {
        "source": source.name,
        "initial_rows": initial_rows,
        "rows_with_nan": nan_rows,
        "rows_with_infinity": infinity_rows,
        "rows_removed_invalid_union": invalid_removed,
        "rows_removed_duplicate_same_label": duplicate_rows,
        "rows_removed_conflicting_labels": conflicting_rows,
        "constant_feature_columns_removed": constant_columns,
        "duplicate_feature_columns_removed": duplicate_columns,
        "final_rows": len(data),
        "final_columns": len(data.columns),
        "output": str(output),
    }


def process_unsw() -> dict:
    folder = ROOT / "ALL UNSW_NB15"
    sources = [folder / "UNSW_NB15_training-set.csv", folder / "UNSW_NB15_testing-set.csv"]
    print("\nUNSW-NB15: φόρτωση αρχικών αρχείων...", flush=True)
    data = pd.concat([normalize_columns(pd.read_csv(source, low_memory=False)) for source in sources], ignore_index=True)
    if "label" not in data.columns:
        raise ValueError("Δεν βρέθηκε η στήλη label στο UNSW-NB15.")
    initial_rows = len(data)
    data, nan_rows, infinity_rows, invalid_removed = clean_invalid_values(data)
    data, duplicate_rows, conflicting_rows = remove_duplicates_and_conflicts(
        data=data,
        label_column="label",
        excluded_columns=["id", "attack_cat"],
    )
    data, constant_columns, duplicate_columns = remove_redundant_feature_columns(
        data=data,
        label_column="label",
    )
    output = OUTPUT_DIR / "UNSW_NB15_cleaned_deduplicated.csv"
    data.to_csv(output, index=False)
    return {
        "source": [source.name for source in sources],
        "initial_rows": initial_rows,
        "rows_with_nan": nan_rows,
        "rows_with_infinity": infinity_rows,
        "rows_removed_invalid_union": invalid_removed,
        "rows_removed_duplicate_same_label": duplicate_rows,
        "rows_removed_conflicting_labels": conflicting_rows,
        "constant_feature_columns_removed": constant_columns,
        "duplicate_feature_columns_removed": duplicate_columns,
        "final_rows": len(data),
        "final_columns": len(data.columns),
        "output": str(output),
    }


def main() -> None:
    report = {"CICIDS2017": process_cicids(), "UNSW_NB15": process_unsw()}
    report_path = OUTPUT_DIR / "report_4.2.1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nΟ καθαρισμός ολοκληρώθηκε.")
    for name, result in report.items():
        print(f"\n{name}")
        print(f"  Αρχικές εγγραφές: {result['initial_rows']:,}")
        print(f"  Αφαιρέθηκαν μη έγκυρες: {result['rows_removed_invalid_union']:,}")
        print(f"  Αφαιρέθηκαν αντιφατικές: {result['rows_removed_conflicting_labels']:,}")
        print(f"  Αφαιρέθηκαν διπλότυπες: {result['rows_removed_duplicate_same_label']:,}")
        print(f"  Αφαιρέθηκαν σταθερά χαρακτηριστικά: {len(result['constant_feature_columns_removed'])}")
        print(f"  Αφαιρέθηκαν ταυτόσημα χαρακτηριστικά: {len(result['duplicate_feature_columns_removed'])}")
        print(f"  Τελικές εγγραφές: {result['final_rows']:,}")
        print(f"  Στήλες μετά τον καθαρισμό: {result['final_columns']}")
    print(f"\nΑναφορά: {report_path}")


if __name__ == "__main__":
    main()
