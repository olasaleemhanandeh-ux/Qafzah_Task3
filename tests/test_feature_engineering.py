import pandas as pd
import pytest

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    add_geolocation_features,
    create_time_features,
    load_geo_features,
    prepare_features,
)


def test_create_time_features():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2016-09-15 12:16:38"
            ]
        }
    )

    result = create_time_features(df)

    assert result.loc[0, "purchase_year"] == 2016
    assert result.loc[0, "purchase_month"] == 9
    assert result.loc[0, "purchase_dayofweek"] == 3
    assert result.loc[0, "purchase_hour"] == 12


def test_geolocation_merge_does_not_duplicate_rows():
    orders = pd.DataFrame(
        {
            "customer_zip_code_prefix": [1000, 2000, 3000],
        }
    )

    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 2000, 3000],
            "geolocation_lat": [-10.0, -20.0, -30.0],
            "geolocation_lng": [-40.0, -50.0, -60.0],
        }
    )

    result = add_geolocation_features(orders, geo)

    assert len(result) == len(orders)
    assert "geolocation_zip_code_prefix" not in result.columns
    assert "geolocation_lat" in result.columns
    assert "geolocation_lng" in result.columns


def test_prepare_features_has_expected_schema():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2016-09-15 12:16:38"
            ],
            "customer_zip_code_prefix": [14600],
            "customer_state": ["SP"],
            "total_items": [3.0],
            "total_price": [134.97],
            "total_freight": [8.49],
            "unique_products": [1.0],
            "unique_sellers": [1.0],
            "total_payments": [None],
            "payment_count": [None],
            "payment_types": [None],
        }
    )

    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [14600],
            "geolocation_lat": [-20.0],
            "geolocation_lng": [-47.0],
        }
    )

    result = prepare_features(df, geo)

    expected_columns = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    assert list(result.columns) == expected_columns
    assert result.shape == (1, 15)


def test_prepare_features_raises_for_missing_columns():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2016-09-15 12:16:38"
            ],
            "customer_zip_code_prefix": [14600],
        }
    )

    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [14600],
            "geolocation_lat": [-20.0],
            "geolocation_lng": [-47.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_features(df, geo)


def test_load_geo_features_missing_file():
    with pytest.raises(FileNotFoundError):
        load_geo_features("does-not-exist.csv")
