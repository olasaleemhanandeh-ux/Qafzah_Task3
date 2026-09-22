"""FastAPI application for Qafzah Task 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict


app = FastAPI(
    title="Qafzah Task 2 API",
    version="1.0.0",
    description="Late delivery prediction API.",
)


class PredictionRequest(BaseModel):
    order: dict[str, Any]


class PredictionResponse(BaseModel):
    late_probability: float
    prediction: int
    prediction_label: str


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "qafzah_task2",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_order(
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict whether an order will be late."""

    try:
        df = pd.DataFrame([request.order])

        result = predict(df)

        row = result.iloc[0]

        return PredictionResponse(
            late_probability=float(row["late_probability"]),
            prediction=int(row["prediction"]),
            prediction_label=str(row["prediction_label"]),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
