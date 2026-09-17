"""Inference utilities for Qafzah Task 2."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import load_geo_features, prepare_features


DEFAULT_ARTIFACT_DIR = Path("data/artifacts")
DEFAULT_THRESHOLD = 0.51


def load_artifacts(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
):
    """Load the saved preprocessor, model, and geographic features."""
    artifact_dir = Path(artifact_dir)

    preprocessor = joblib.load(
        artifact_dir / "preprocessor.joblib"
    )

    model = joblib.load(
        artifact_dir / "final_model.joblib"
    )

    geo_features = load_geo_features(
        artifact_dir / "geo_features.csv"
    )

    return preprocessor, model, geo_features


def predict(
    df: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Generate late-delivery predictions for raw order data."""

    preprocessor, model, geo_features = load_artifacts(
        artifact_dir
    )

    features = prepare_features(
        df,
        geo_features,
    )

    processed_features = preprocessor.transform(
        features
    )

    probabilities = model.predict_proba(
        processed_features
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame(
        {
            "late_probability": probabilities,
            "prediction": predictions,
            "prediction_label": [
                "Late" if value == 1 else "On-time"
                for value in predictions
            ],
        },
        index=df.index,
    )

    return result


if __name__ == "__main__":
    data_path = Path("data/artifacts/train.csv")

    data = pd.read_csv(data_path)

    predictions = predict(data)

    print("Inference completed successfully.")
    print("Rows:", len(predictions))

    print("\nPredictions:")
    print(predictions.head())

    print("\nPrediction distribution:")
    print(predictions["prediction_label"].value_counts())

    if "late" in data.columns:
        print("\nActual target distribution:")
        print(data["late"].value_counts())

        accuracy = (
            predictions["prediction"].to_numpy()
            == data["late"].to_numpy()
        ).mean()

        print(f"\nAccuracy on provided data: {accuracy:.4f}")
