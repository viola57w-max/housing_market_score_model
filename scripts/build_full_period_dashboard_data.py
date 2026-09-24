"""Build the compact JavaScript payload used by modeling-summary.html."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "outputs" / "scores_all_periods.csv"
FEATURES_PATH = ROOT / "data" / "full_period_data.csv"
OUTPUT_PATH = ROOT / "dashboards" / "assets" / "full-period-scoring-data.js"
REFERENCE_FEATURES_PATH = ROOT / "data" / "modeling_data.csv"
REFERENCE_SCORES_PATH = ROOT / "outputs" / "scores_latest_select_zips_eval.csv"

FEATURES = [
    "MEDIAN_SALE_PRICE_MOM",
    "MEDIAN_LIST_PRICE_MOM",
    "MEDIAN_LIST_PPSF_MOM",
    "HOMES_SOLD",
    "NEW_LISTINGS_YOY",
    "MEDIAN_DOM",
    "AVG_SALE_TO_LIST",
    "SOLD_ABOVE_LIST",
    "OFF_MARKET_IN_TWO_WEEKS",
]


def optional_number(value):
    if pd.isna(value):
        return None
    return round(float(value), 6)


scores = pd.read_csv(SCORES_PATH)
features = pd.read_csv(
    FEATURES_PATH, usecols=["Zip Code", "PERIOD_END", *FEATURES]
)
features["PERIOD"] = (
    pd.to_datetime(features["PERIOD_END"], errors="raise")
    .dt.to_period("M")
    .astype(str)
)

dashboard_data = scores.merge(
    features.drop(columns="PERIOD_END"),
    on=["Zip Code", "PERIOD"],
    how="left",
    validate="one_to_one",
)
if len(dashboard_data) != len(scores):
    raise ValueError("Dashboard feature merge changed the score-row count.")

reference_features = pd.read_csv(
    REFERENCE_FEATURES_PATH, usecols=["Zip Code", *FEATURES]
)
reference_scores = pd.read_csv(
    REFERENCE_SCORES_PATH, usecols=["Zip Code", "PERIOD", "y_pred"]
)
reference_data = reference_scores.merge(
    reference_features,
    on="Zip Code",
    how="left",
    validate="one_to_one",
)

payload = {
    "features": FEATURES,
    "referenceRows": [
        [str(int(row["Zip Code"])), row["PERIOD"], int(row["y_pred"])]
        + [optional_number(row[feature]) for feature in FEATURES]
        for _, row in reference_data.sort_values("Zip Code").iterrows()
    ],
    "zips": {},
}
for zip_code, group in dashboard_data.groupby("Zip Code", sort=True):
    group = group.sort_values("PERIOD")
    payload["zips"][str(zip_code)] = {
        "city": group["City"].iloc[0],
        "rows": [
            [row["PERIOD"], int(row["MarketScore_hat"])]
            + [optional_number(row[feature]) for feature in FEATURES]
            for _, row in group.iterrows()
        ],
    }

OUTPUT_PATH.write_text(
    "window.fullPeriodScoringData="
    + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    + ";\n",
    encoding="utf-8",
)
print(
    f"Saved {len(dashboard_data):,} rows across {len(payload['zips']):,} ZIPs "
    f"to {OUTPUT_PATH}"
)
