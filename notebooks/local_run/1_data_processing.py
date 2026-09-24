"""Prepare the four modeling inputs used by 2_modeling.py.

Place this script in the same folder as SoCal.csv, market_score_reference.csv,
and data_dictionary.csv. It writes only modeling_data.csv, has_2025_data.csv,
full_period_data.csv, and model_feature_list.json to that folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ZIP_COLUMN = "Zip Code"
TARGET_COLUMN = "Market_Score"
REFERENCE_SCORE_COLUMN = "Market Score"
FLAG_COLUMNS = {ZIP_COLUMN, "has_2025", "latest_period"}


def normalize_zip(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    socal = pd.read_csv(BASE_DIR / "SoCal.csv", dtype={ZIP_COLUMN: "string"})
    reference = pd.read_csv(
        BASE_DIR / "market_score_reference.csv", dtype={ZIP_COLUMN: "string"}
    )
    dictionary = pd.read_csv(BASE_DIR / "data_dictionary.csv")
    return socal, reference, dictionary


def validate_inputs(
    socal: pd.DataFrame,
    reference: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> None:
    required = {
        "SoCal.csv": {ZIP_COLUMN, "PERIOD_BEGIN", "PERIOD_END", "City"},
        "market_score_reference.csv": {ZIP_COLUMN, REFERENCE_SCORE_COLUMN},
        "data_dictionary.csv": {"Variable", "Column_type"},
    }
    frames = {
        "SoCal.csv": socal,
        "market_score_reference.csv": reference,
        "data_dictionary.csv": dictionary,
    }
    missing = {
        name: sorted(columns - set(frames[name].columns))
        for name, columns in required.items()
        if columns - set(frames[name].columns)
    }
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")


def prepare_reference(reference: pd.DataFrame) -> pd.Series:
    prepared = reference.copy()
    prepared[ZIP_COLUMN] = normalize_zip(prepared[ZIP_COLUMN])
    prepared[TARGET_COLUMN] = pd.to_numeric(
        prepared[REFERENCE_SCORE_COLUMN], errors="coerce"
    )
    prepared = prepared.dropna(subset=[ZIP_COLUMN, TARGET_COLUMN])
    conflicts = prepared.groupby(ZIP_COLUMN)[TARGET_COLUMN].nunique().gt(1)
    if conflicts.any():
        raise ValueError("The reference file contains conflicting scores for a ZIP.")
    return prepared.drop_duplicates(ZIP_COLUMN).set_index(ZIP_COLUMN)[TARGET_COLUMN]


def identify_features(
    data: pd.DataFrame, dictionary: pd.DataFrame
) -> tuple[list[str], list[str]]:
    dictionary_types = dictionary.set_index("Variable")["Column_type"]
    numerical_columns = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column != TARGET_COLUMN
        and column not in FLAG_COLUMNS
        and dictionary_types.get(column) == "Numerical"
    ]
    categorical_columns = [
        column
        for column in data.columns
        if column not in FLAG_COLUMNS
        and dictionary_types.get(column) == "Categorical"
    ]
    return numerical_columns, categorical_columns


def process_data(
    socal: pd.DataFrame,
    reference: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    validate_inputs(socal, reference, dictionary)
    score_map = prepare_reference(reference)

    prepared = socal.copy()
    prepared[ZIP_COLUMN] = normalize_zip(prepared[ZIP_COLUMN])
    prepared["PERIOD_BEGIN"] = pd.to_datetime(
        prepared["PERIOD_BEGIN"], errors="coerce"
    )
    prepared["PERIOD_END"] = pd.to_datetime(prepared["PERIOD_END"], errors="coerce")
    if prepared[ZIP_COLUMN].isna().any() or prepared["PERIOD_END"].isna().any():
        raise ValueError("SoCal.csv contains missing ZIP codes or invalid PERIOD_END values.")

    is_2025 = prepared["PERIOD_END"].dt.year.eq(2025)
    prepared["has_2025"] = is_2025.groupby(prepared[ZIP_COLUMN]).transform("any")
    latest_2025_end = prepared.loc[is_2025].groupby(ZIP_COLUMN)["PERIOD_END"].max()
    prepared["latest_period"] = is_2025 & prepared["PERIOD_END"].eq(
        prepared[ZIP_COLUMN].map(latest_2025_end)
    )
    prepared[TARGET_COLUMN] = prepared[ZIP_COLUMN].map(score_map)

    unique_counts = prepared.nunique(dropna=False)
    cleaned = prepared.drop(
        columns=unique_counts[unique_counts <= 1].index.tolist()
    ).copy()
    if "City" not in cleaned.columns:
        raise ValueError("City became unavailable during uniform-column removal.")
    cleaned = cleaned.drop(columns="City")

    numerical_columns, categorical_columns = identify_features(cleaned, dictionary)
    feature_list = {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }

    modeling_data = (
        cleaned.loc[cleaned["latest_period"] & cleaned[TARGET_COLUMN].notna()]
        .sort_values([ZIP_COLUMN, "PERIOD_END"])
        .reset_index(drop=True)
    )
    has_2025_data = (
        cleaned.loc[cleaned["has_2025"]]
        .sort_values([ZIP_COLUMN, "PERIOD_END"])
        .reset_index(drop=True)
    )
    full_period_data = cleaned.sort_values(
        [ZIP_COLUMN, "PERIOD_END"]
    ).reset_index(drop=True)

    if not modeling_data[ZIP_COLUMN].is_unique:
        raise ValueError("modeling_data must contain one latest-2025 row per ZIP.")
    if modeling_data.isna().any().any():
        missing = modeling_data.columns[modeling_data.isna().any()].tolist()
        raise ValueError(f"modeling_data unexpectedly contains missing values: {missing}")
    if list(modeling_data.columns) != list(full_period_data.columns):
        raise ValueError("The three output datasets must share one schema.")
    return modeling_data, has_2025_data, full_period_data, feature_list


def main() -> None:
    modeling_data, has_2025_data, full_period_data, feature_list = process_data(
        *load_inputs()
    )
    modeling_data.to_csv(BASE_DIR / "modeling_data.csv", index=False)
    has_2025_data.to_csv(BASE_DIR / "has_2025_data.csv", index=False)
    full_period_data.to_csv(BASE_DIR / "full_period_data.csv", index=False)
    with (BASE_DIR / "model_feature_list.json").open("w", encoding="utf-8") as file:
        json.dump(feature_list, file, indent=2)

    for filename in (
        "modeling_data.csv",
        "has_2025_data.csv",
        "full_period_data.csv",
        "model_feature_list.json",
    ):
        print(f"Created {filename}")


if __name__ == "__main__":
    main()
