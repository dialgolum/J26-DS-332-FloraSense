"""FastAPI application for order-level C4 supply-demand forecasting."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import get_metadata, predict
from .schemas import ForecastRequest, ForecastResponse

app = FastAPI(
    title="FloraSense C4 Supply-Demand Forecasting API",
    version="1.0.0",
    description="Order-aware dispatch forecasts using the trained C4 XGBoost model.",
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "FLORASENSE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"data": {"status": "ok", "service": "c4-forecasting"}, "error": None}


@app.get("/api/v1/forecast/metadata", tags=["C4 forecasting"])
def forecast_metadata() -> dict[str, list[str]]:
    """Return distinct category options from the C4 training data."""
    try:
        return get_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/api/v1/forecast/predict",
    response_model=ForecastResponse,
    tags=["C4 forecasting"],
)
def forecast_dispatch(request: ForecastRequest) -> ForecastResponse:
    """Estimate dispatch supply after the order quantity is committed."""
    try:
        return predict(request)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
