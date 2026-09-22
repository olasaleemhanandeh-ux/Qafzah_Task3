from pathlib import Path

import great_expectations as gx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "artifacts" / "train.csv"


REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "total_items",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "total_payments",
    "payment_count",
    "payment_types",
    "late",
]

REQUIRED_NON_NULL_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "total_items",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "late",
]

NUMERICAL_COLUMNS = [
    "customer_zip_code_prefix",
    "total_items",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "total_payments",
    "payment_count",
    "payment_types",
]

EXPECTED_STATES = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]

EXPECTED_ORDER_STATUS = [
    "delivered",
    "canceled",
]

EXPECTED_LATE_VALUES = [0, 1]


def load_training_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Load the reference training data used for validation."""
    return pd.read_csv(path)


def build_expectations(df: pd.DataFrame):
    """Build Great Expectations for the model input dataset."""
    context = gx.get_context()

    datasource = context.sources.add_pandas(
        name="olist_training_data"
    )

    data_asset = datasource.add_dataframe_asset(
        name="train_dataframe"
    )

    batch_request = data_asset.build_batch_request(
        dataframe=df
    )

    validator = context.get_validator(
        batch_request=batch_request
    )

    # Schema: required columns
    validator.expect_table_columns_to_match_ordered_list(
        column_list=REQUIRED_COLUMNS
    )

    # Missingness: columns required by the feature pipeline
    for column in REQUIRED_NON_NULL_COLUMNS:
        validator.expect_column_values_to_not_be_null(
            column=column
        )

    # Numeric columns
    for column in NUMERICAL_COLUMNS:
        validator.expect_column_values_to_be_of_type(
            column=column,
            type_="float64" if column != "customer_zip_code_prefix" else "int64",
        )

    # Categories
    validator.expect_column_values_to_be_in_set(
        column="order_status",
        value_set=EXPECTED_ORDER_STATUS,
    )

    validator.expect_column_values_to_be_in_set(
        column="customer_state",
        value_set=EXPECTED_STATES,
    )

    validator.expect_column_values_to_be_in_set(
        column="late",
        value_set=EXPECTED_LATE_VALUES,
    )

    # Logical ranges
    validator.expect_column_values_to_be_between(
        column="total_items",
        min_value=1,
        max_value=100,
    )

    validator.expect_column_values_to_be_between(
        column="total_price",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        column="total_freight",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        column="unique_products",
        min_value=1,
    )

    validator.expect_column_values_to_be_between(
        column="unique_sellers",
        min_value=1,
    )

    validator.expect_column_values_to_be_between(
        column="payment_count",
        min_value=1,
    )

    validator.expect_column_values_to_be_between(
        column="payment_types",
        min_value=1,
    )

    validator.expect_column_values_to_be_between(
        column="late",
        min_value=0,
        max_value=1,
    )

    return validator


def validate_training_data(
    df: pd.DataFrame | None = None,
) -> bool:
    """Validate model input data and return True when all expectations pass."""
    if df is None:
        df = load_training_data()

    validator = build_expectations(df)
    result = validator.validate()

    if not result.success:
        print("Data validation FAILED.")

        for result_item in result.results:
            if not result_item.success:
                print(
                    f"- {result_item.expectation_config.expectation_type}"
                )

        return False

    print("Data validation PASSED.")
    return True


if __name__ == "__main__":
    success = validate_training_data()

    if not success:
        raise SystemExit(1)
