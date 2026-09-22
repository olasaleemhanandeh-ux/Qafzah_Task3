import pandas as pd

from src.predict import (
    DEFAULT_THRESHOLD,
    load_artifacts,
    predict,
)


ARTIFACT_DIR = "data/artifacts"


def test_load_artifacts():
    preprocessor, model, geo_features = load_artifacts(
        ARTIFACT_DIR
    )

    assert preprocessor is not None
    assert model is not None
    assert geo_features is not None

    assert len(geo_features) == 19015


def test_predict_returns_expected_columns():
    df = pd.read_csv(
        f"{ARTIFACT_DIR}/train.csv"
    ).head(1)

    result = predict(
        df,
        artifact_dir=ARTIFACT_DIR,
    )

    assert list(result.columns) == [
        "late_probability",
        "prediction",
        "prediction_label",
    ]


def test_predict_returns_one_result_per_input_row():
    df = pd.read_csv(
        f"{ARTIFACT_DIR}/train.csv"
    ).head(10)

    result = predict(
        df,
        artifact_dir=ARTIFACT_DIR,
    )

    assert len(result) == len(df)


def test_prediction_values_are_valid():
    df = pd.read_csv(
        f"{ARTIFACT_DIR}/train.csv"
    ).head(20)

    result = predict(
        df,
        artifact_dir=ARTIFACT_DIR,
    )

    assert result["late_probability"].between(
        0.0, 1.0
    ).all()

    assert result["prediction"].isin(
        [0, 1]
    ).all()

    assert result["prediction_label"].isin(
        ["Late", "On-time"]
    ).all()


def test_threshold_is_respected():
    df = pd.read_csv(
        f"{ARTIFACT_DIR}/train.csv"
    ).head(20)

    result = predict(
        df,
        artifact_dir=ARTIFACT_DIR,
        threshold=DEFAULT_THRESHOLD,
    )

    expected_predictions = (
        result["late_probability"] >= DEFAULT_THRESHOLD
    ).astype(int)

    assert (
        result["prediction"].to_numpy()
        == expected_predictions.to_numpy()
    ).all()
