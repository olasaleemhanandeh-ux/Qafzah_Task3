"""Feature engineering for Qafzah Task 2 inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TIME_FEATURES = [
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
]

NUMERICAL_FEATURES = [
    "total_items",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "total_payments",
    "payment_count",
    "payment_types",
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "geolocation_lat",
    "geolocation_lng",
]

CATEGORICAL_FEATURES = [
    "customer_state",
]

TARGET_COLUMN = "late"

DROP_COLUMNS = [
    "late",
    "order_id",
    "customer_id",
    "customer_unique_id",
    "customer_city",
    "customer_zip_code_prefix",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_status",
]


def add_geolocation_features(
    df: pd.DataFrame,
    geo_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge geographic coordinates into an orders dataframe.

    This follows the merge used in Notebook 05:
    customer_zip_code_prefix ->
    geolocation_zip_code_prefix.
    """
    result = df.merge(
        geo_features,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    result = result.drop(
        columns=["geolocation_zip_code_prefix"]
    )

    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the time features used by the original notebook."""
    result = df.copy()

    result["order_purchase_timestamp"] = pd.to_datetime(
        result["order_purchase_timestamp"]
    )

    result["purchase_year"] = (
        result["order_purchase_timestamp"].dt.year
    )

    result["purchase_month"] = (
        result["order_purchase_timestamp"].dt.month
    )

    result["purchase_dayofweek"] = (
        result["order_purchase_timestamp"].dt.dayofweek
    )

    result["purchase_hour"] = (
        result["order_purchase_timestamp"].dt.hour
    )

    return result


def prepare_features(
    df: pd.DataFrame,
    geo_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the feature table expected by the saved preprocessor."""
    result = add_geolocation_features(
        df,
        geo_features,
    )

    result = create_time_features(result)

    missing_columns = [
        column
        for column in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    result = result.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    expected_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    return result[expected_columns]


def load_geo_features(
    path: str | Path,
) -> pd.DataFrame:
    """Load the geographic feature table."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Geo features file not found: {path}"
        )

    return pd.read_csv(path)
