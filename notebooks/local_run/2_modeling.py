"""Fit the market-score model and create four shareable scoring outputs.

Place this script in the same folder as SoCal.csv and the four files produced
by 1_data_processing.py. It writes only:
    scores_latest_select_zips_eval.csv
    scores_latest_all_zips.csv
    scores_all_periods.csv
    scores_all_periods_updated.csv

No plots, diagnostic tables, model artifacts, or machine-specific paths are
created or exposed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ZIP_COLUMN = "Zip Code"
TARGET_COLUMN = "Market_Score"
RANDOM_SEED = 42
ZERO_TOLERANCE = 1e-8
LASSO_ALPHAS = np.logspace(-4, 2, 200)

INCOMPATIBLE_GROUPS = [
    ["MEDIAN_SALE_PRICE", "MEDIAN_SALE_PRICE_MOM", "MEDIAN_SALE_PRICE_YOY"],
    ["MEDIAN_LIST_PRICE", "MEDIAN_LIST_PRICE_MOM", "MEDIAN_LIST_PRICE_YOY"],
    ["MEDIAN_PPSF", "MEDIAN_PPSF_MOM", "MEDIAN_PPSF_YOY"],
    ["MEDIAN_LIST_PPSF", "MEDIAN_LIST_PPSF_MOM", "MEDIAN_LIST_PPSF_YOY"],
    ["HOMES_SOLD", "HOMES_SOLD_MOM", "HOMES_SOLD_YOY"],
    ["PENDING_SALES", "PENDING_SALES_MOM", "PENDING_SALES_YOY"],
    ["NEW_LISTINGS", "NEW_LISTINGS_MOM", "NEW_LISTINGS_YOY"],
    ["INVENTORY", "INVENTORY_MOM", "INVENTORY_YOY"],
    ["MEDIAN_DOM", "MEDIAN_DOM_MOM", "MEDIAN_DOM_YOY"],
    ["AVG_SALE_TO_LIST", "AVG_SALE_TO_LIST_MOM", "AVG_SALE_TO_LIST_YOY"],
    ["SOLD_ABOVE_LIST", "SOLD_ABOVE_LIST_MOM", "SOLD_ABOVE_LIST_YOY"],
    [
        "OFF_MARKET_IN_TWO_WEEKS",
        "OFF_MARKET_IN_TWO_WEEKS_MOM",
        "OFF_MARKET_IN_TWO_WEEKS_YOY",
    ],
    ["County", "PARENT_METRO_REGION"],
]


def normalize_zip(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / filename, dtype={ZIP_COLUMN: "string"})


def load_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], pd.Series
]:
    modeling_data = read_csv("modeling_data.csv")
    has_2025_data = read_csv("has_2025_data.csv")
    full_period_data = read_csv("full_period_data.csv")
    for frame in (modeling_data, has_2025_data, full_period_data):
        frame[ZIP_COLUMN] = normalize_zip(frame[ZIP_COLUMN])

    with (BASE_DIR / "model_feature_list.json").open(encoding="utf-8") as file:
        feature_list = json.load(file)
    numerical = feature_list["numerical_columns"]
    categorical = feature_list["categorical_columns"]

    socal = pd.read_csv(BASE_DIR / "SoCal.csv", dtype={ZIP_COLUMN: "string"})
    socal[ZIP_COLUMN] = normalize_zip(socal[ZIP_COLUMN])
    city_lookup = (
        socal.dropna(subset=["City"])
        .groupby(ZIP_COLUMN)["City"]
        .agg(lambda values: values.mode().iloc[0])
    )
    return (
        modeling_data,
        has_2025_data,
        full_period_data,
        numerical,
        categorical,
        city_lookup,
    )


def expected_sign(variable: str, categorical_columns: list[str]) -> str:
    if variable in categorical_columns:
        return "not_applicable"
    if variable.startswith(("MEDIAN_DOM", "INVENTORY")):
        return "negative"
    if variable.startswith(
        (
            "AVG_SALE_TO_LIST",
            "SOLD_ABOVE_LIST",
            "OFF_MARKET_IN_TWO_WEEKS",
            "HOMES_SOLD",
            "PENDING_SALES",
        )
    ):
        return "positive"
    if variable.endswith(("_MOM", "_YOY")) and variable.startswith(
        (
            "MEDIAN_SALE_PRICE",
            "MEDIAN_LIST_PRICE",
            "MEDIAN_PPSF",
            "MEDIAN_LIST_PPSF",
        )
    ):
        return "positive"
    return "unconstrained"


def compatibility_strength(
    data: pd.DataFrame, variable: str, categorical_columns: list[str]
) -> float:
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if variable in categorical_columns:
        fitted = data.groupby(variable, dropna=False)[TARGET_COLUMN].transform("mean")
        valid = target.notna() & fitted.notna()
        return r2_score(target.loc[valid], fitted.loc[valid]) if valid.sum() > 1 else -np.inf
    values = pd.to_numeric(data[variable], errors="coerce")
    valid = target.notna() & values.notna()
    if valid.sum() < 3 or values.loc[valid].nunique() < 2:
        return -np.inf
    correlation = spearmanr(target.loc[valid], values.loc[valid])[0]
    return abs(float(correlation)) if np.isfinite(correlation) else -np.inf


def select_compatible_features(
    data: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> tuple[list[str], list[str]]:
    selected = list(numerical_columns) + list(categorical_columns)
    for group in INCOMPATIBLE_GROUPS:
        available = [variable for variable in group if variable in selected]
        if len(available) <= 1:
            continue
        retained = max(
            available,
            key=lambda variable: (
                compatibility_strength(data, variable, categorical_columns),
                variable,
            ),
        )
        selected = [
            variable
            for variable in selected
            if variable not in available or variable == retained
        ]
    return (
        [variable for variable in numerical_columns if variable in selected],
        [variable for variable in categorical_columns if variable in selected],
    )


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            drop=None, handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(drop=None, handle_unknown="ignore", sparse=False)


def make_preprocessor(
    numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    transformers = []
    if numerical_columns:
        transformers.append(
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_cv_splits(
    data: pd.DataFrame, target: pd.Series
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.Series]:
    groups = data[ZIP_COLUMN].astype(str)
    repeated_zips = groups.duplicated().any()
    n_splits = min(5, groups.nunique() if repeated_zips else len(data))
    if n_splits < 2:
        raise ValueError("At least two observations or ZIP groups are required.")
    if repeated_zips:
        splits = list(GroupKFold(n_splits=n_splits).split(data, target, groups))
    else:
        splits = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED).split(
                data, target
            )
        )
    return splits, groups


def make_pipeline(
    numerical_columns: list[str],
    categorical_columns: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numerical_columns, categorical_columns)),
            (
                "model",
                LassoCV(
                    alphas=LASSO_ALPHAS,
                    cv=cv_splits,
                    max_iter=100000,
                    tol=1e-6,
                    selection="cyclic",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def encoded_original_variables(
    preprocessor: ColumnTransformer,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    original_variables = list(numerical_columns)
    if categorical_columns:
        encoder = preprocessor.named_transformers_["categorical"].named_steps[
            "encoder"
        ]
        for variable, categories in zip(categorical_columns, encoder.categories_):
            original_variables.extend([variable] * len(categories))
    return original_variables


def selected_variables(
    pipeline: Pipeline,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> tuple[list[str], list[tuple[str, float]]]:
    coefficients = pipeline.named_steps["model"].coef_
    originals = encoded_original_variables(
        pipeline.named_steps["preprocessor"], numerical_columns, categorical_columns
    )
    if len(coefficients) != len(originals):
        raise ValueError("Encoded feature metadata does not match model coefficients.")
    selected_pairs = [
        (variable, float(coefficient))
        for variable, coefficient in zip(originals, coefficients)
        if abs(coefficient) > ZERO_TOLERANCE
    ]
    selected = list(dict.fromkeys(variable for variable, _ in selected_pairs))
    return selected, selected_pairs


def fit_champion(
    modeling_data: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> tuple[Pipeline, list[str], list[str]]:
    excluded = {TARGET_COLUMN, ZIP_COLUMN, "has_2025", "latest_period"}
    numerical_candidates = [
        column
        for column in numerical_columns
        if column in modeling_data.columns and column not in excluded
    ]
    categorical_candidates = [
        column
        for column in categorical_columns
        if column in modeling_data.columns and column not in excluded
    ]
    compatible_numerical, compatible_categorical = select_compatible_features(
        modeling_data, numerical_candidates, categorical_candidates
    )
    compatible_features = compatible_numerical + compatible_categorical
    target = pd.to_numeric(modeling_data[TARGET_COLUMN], errors="raise").astype(float)
    cv_splits, _ = make_cv_splits(modeling_data, target)

    initial_pipeline = make_pipeline(
        compatible_numerical, compatible_categorical, cv_splits
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        initial_pipeline.fit(modeling_data[compatible_features], target)

    _, initial_selected_pairs = selected_variables(
        initial_pipeline, compatible_numerical, compatible_categorical
    )
    sign_failures = set()
    for variable, coefficient in initial_selected_pairs:
        expectation = expected_sign(variable, compatible_categorical)
        if expectation == "positive" and coefficient <= 0:
            sign_failures.add(variable)
        elif expectation == "negative" and coefficient >= 0:
            sign_failures.add(variable)

    refit_numerical = [
        variable for variable in compatible_numerical if variable not in sign_failures
    ]
    refit_categorical = [
        variable for variable in compatible_categorical if variable not in sign_failures
    ]
    refit_features = refit_numerical + refit_categorical
    if not refit_features:
        raise ValueError("No features remain after the business-sign check.")

    refit_splits, _ = make_cv_splits(modeling_data, target)
    champion = make_pipeline(refit_numerical, refit_categorical, refit_splits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        champion.fit(modeling_data[refit_features], target)
    champion_selected, _ = selected_variables(
        champion, refit_numerical, refit_categorical
    )
    return champion, refit_features, champion_selected


def constrained_scores(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(predictions), 0, 100).astype(int)


def add_period_and_city(data: pd.DataFrame, city_lookup: pd.Series) -> pd.DataFrame:
    prepared = data.copy()
    prepared["PERIOD_END"] = pd.to_datetime(prepared["PERIOD_END"], errors="raise")
    prepared["PERIOD"] = prepared["PERIOD_END"].dt.to_period("M").astype(str)
    prepared["City"] = prepared[ZIP_COLUMN].map(city_lookup)
    if prepared["City"].isna().any():
        raise ValueError("City is missing for one or more output rows.")
    return prepared


def score_frame(
    data: pd.DataFrame,
    pipeline: Pipeline,
    features: list[str],
    city_lookup: pd.Series,
) -> pd.DataFrame:
    scored = add_period_and_city(data, city_lookup)
    scored["MarketScore_hat"] = constrained_scores(
        pipeline.predict(scored[features])
    )
    return scored


def validate_unique(data: pd.DataFrame, keys: list[str], score_column: str) -> None:
    if data.duplicated(keys).any():
        raise ValueError(f"Output contains duplicate keys: {keys}")
    if not data[score_column].between(0, 100).all():
        raise ValueError(f"{score_column} falls outside 0-100.")
    if data[score_column].dtype.kind not in "iu":
        raise TypeError(f"{score_column} must contain integers.")


def build_outputs(
    modeling_data: pd.DataFrame,
    has_2025_data: pd.DataFrame,
    full_period_data: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    city_lookup: pd.Series,
) -> dict[str, pd.DataFrame]:
    champion, features, selected = fit_champion(
        modeling_data, numerical_columns, categorical_columns
    )

    latest_eval = score_frame(modeling_data, champion, features, city_lookup)
    latest_eval["y_true"] = np.rint(
        pd.to_numeric(latest_eval[TARGET_COLUMN], errors="raise")
    ).astype(int)
    latest_eval["y_pred"] = latest_eval["MarketScore_hat"]
    scores_latest_select = latest_eval[
        [ZIP_COLUMN, "City", "County", "PERIOD", "y_true", "y_pred"]
    ].copy()
    validate_unique(scores_latest_select, [ZIP_COLUMN], "y_pred")

    latest_all = has_2025_data.loc[
        has_2025_data["has_2025"].eq(True)
        & has_2025_data["latest_period"].eq(True)
    ].copy()
    latest_all = score_frame(latest_all, champion, features, city_lookup)
    scores_latest_all = latest_all[
        [ZIP_COLUMN, "City", "County", "PERIOD", "MarketScore_hat"]
    ].copy()
    validate_unique(scores_latest_all, [ZIP_COLUMN], "MarketScore_hat")

    full_scored = score_frame(full_period_data, champion, features, city_lookup)
    scores_all = (
        full_scored[[ZIP_COLUMN, "City", "County", "PERIOD", "MarketScore_hat"]]
        .sort_values([ZIP_COLUMN, "PERIOD"])
        .reset_index(drop=True)
    )
    validate_unique(scores_all, [ZIP_COLUMN, "PERIOD"], "MarketScore_hat")

    adjusted_source = full_period_data.copy()
    selected_numerical = [
        variable
        for variable in selected
        if variable in numerical_columns and variable in adjusted_source.columns
    ]
    for variable in selected_numerical:
        reference_values = pd.to_numeric(
            modeling_data[variable], errors="coerce"
        ).dropna()
        q1, q3 = reference_values.quantile([0.25, 0.75])
        iqr = q3 - q1
        adjusted_source[variable] = pd.to_numeric(
            adjusted_source[variable], errors="coerce"
        ).clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    adjusted_scored = score_frame(adjusted_source, champion, features, city_lookup)
    pre_mapping_scores = adjusted_scored["MarketScore_hat"].astype(float)
    reference_predictions = constrained_scores(
        champion.predict(modeling_data[features])
    ).astype(float)
    reference_mean = float(np.mean(reference_predictions))
    reference_std = float(np.std(reference_predictions, ddof=1))
    if not np.isfinite(reference_std) or reference_std <= 0:
        raise ValueError("The fitted reference score distribution is invalid.")
    ranks = pre_mapping_scores.rank(method="average").to_numpy()
    percentiles = (ranks - 0.5) / len(pre_mapping_scores)
    mapped = reference_mean + reference_std * norm.ppf(percentiles)
    adjusted_scored["MarketScore_hat"] = constrained_scores(mapped)
    scores_updated = (
        adjusted_scored[
            [ZIP_COLUMN, "City", "County", "PERIOD", "MarketScore_hat"]
        ]
        .sort_values([ZIP_COLUMN, "PERIOD"])
        .reset_index(drop=True)
    )
    validate_unique(scores_updated, [ZIP_COLUMN, "PERIOD"], "MarketScore_hat")

    return {
        "scores_latest_select_zips_eval.csv": scores_latest_select,
        "scores_latest_all_zips.csv": scores_latest_all,
        "scores_all_periods.csv": scores_all,
        "scores_all_periods_updated.csv": scores_updated,
    }


def main() -> None:
    inputs = load_inputs()
    outputs = build_outputs(*inputs)
    for filename, data in outputs.items():
        data.to_csv(BASE_DIR / filename, index=False)
        print(f"Created {filename}")


if __name__ == "__main__":
    main()
