"""Load the trained C4 artifacts and prepare one order-level forecast."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from .schemas import ForecastRequest, ForecastResponse

ROOT = Path(__file__).resolve().parents[3]
MASTER_PATH = ROOT / "data" / "c4_time_series" / "processed" / "c4_master_training_data.csv"
WEATHER_PATH = ROOT / "data" / "c4_time_series" / "interim" / "cleaned_weather_weekly.csv"
FEATURE_META_PATH = ROOT / "data" / "c4_time_series" / "ml_ready" / "feature_meta.json"
SCALER_PATH = ROOT / "data" / "c4_time_series" / "ml_ready" / "scaler.pkl"
MODEL_PATH = ROOT / "models" / "c4_time_series" / "checkpoints" / "xgboost_model.pkl"

MODEL_LABEL = "XGBoost Regressor (Test MAE=92.26)"
CAT_COLUMN_TO_REQUEST = {
    "Customer": "customer",
    "Plant_Variety": "plant_variety",
    "Size": "size",
    "Mode": "mode",
}


@dataclass(frozen=True)
class ForecastArtifacts:
    model: Any
    scaler: Any
    feature_cols: list[str]
    numeric_cols: list[str]
    category_values: dict[str, list[str]]
    imputation_medians: dict[str, float]
    weather_by_week: dict[tuple[int, int], dict[str, float]]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "C4 forecast artifacts are missing. Run the C4 cleaning, preprocessing, "
            "and model-training scripts first. Missing: " + ", ".join(missing)
        )


@lru_cache(maxsize=1)
def load_artifacts() -> ForecastArtifacts:
    _require_files([MASTER_PATH, WEATHER_PATH, FEATURE_META_PATH, SCALER_PATH, MODEL_PATH])
    with FEATURE_META_PATH.open(encoding="utf-8") as handle:
        feature_meta = json.load(handle)

    feature_cols = list(feature_meta["feature_cols"])
    numeric_cols = list(feature_meta["num_cols"])
    category_cols = list(feature_meta["cat_cols"])
    if set(CAT_COLUMN_TO_REQUEST) != set(category_cols):
        raise RuntimeError(f"Unexpected C4 categorical feature schema: {category_cols}")

    master = pd.read_csv(MASTER_PATH)
    numeric_train = master.loc[master["YEAR"] < 2025, numeric_cols]
    medians = numeric_train.median(numeric_only=True)
    if medians.reindex(numeric_cols).isna().any():
        raise RuntimeError("Cannot build the forecast imputer: a training feature has no median.")

    category_values = {
        column: sorted(master[column].dropna().astype(str).unique().tolist())
        for column in category_cols
    }
    weather = pd.read_csv(WEATHER_PATH)
    weather_by_week: dict[tuple[int, int], dict[str, float]] = {}
    for row in weather.to_dict(orient="records"):
        key = (int(row["YEAR"]), int(row["WEEK"]))
        weather_by_week[key] = {
            "Avg_Max_Temp_C": _finite_or_nan(row.get("Avg_Max_Temp_C")),
            "Weekly_Rainfall_mm": _finite_or_nan(row.get("Weekly_Rainfall_mm")),
            "Growing_Degree_Days_Weekly": _finite_or_nan(row.get("Growing_Degree_Days_Weekly")),
        }

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    if getattr(scaler, "n_features_in_", len(numeric_cols)) != len(numeric_cols):
        raise RuntimeError("The saved scaler does not match feature_meta.json.")
    if getattr(model, "n_features_in_", len(feature_cols)) != len(feature_cols):
        raise RuntimeError("The saved XGBoost model does not match feature_meta.json.")

    return ForecastArtifacts(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        category_values=category_values,
        imputation_medians={key: float(medians[key]) for key in numeric_cols},
        weather_by_week=weather_by_week,
    )


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def get_metadata() -> dict[str, list[str]]:
    artifacts = load_artifacts()
    response = {
        "customers": artifacts.category_values["Customer"],
        "plant_varieties": artifacts.category_values["Plant_Variety"],
        "sizes": artifacts.category_values["Size"],
        "modes": artifacts.category_values["Mode"],
    }
    return response


def _lagged_weather(
    artifacts: ForecastArtifacts,
    year: int,
    week: int,
    lag_weeks: int,
) -> dict[str, float]:
    # C4 training uses a continuous 52-week nursery calendar anchored in 2023.
    absolute_week = (year - 2023) * 52 + week - 1 - lag_weeks
    lag_year = 2023 + absolute_week // 52
    lag_week = absolute_week % 52 + 1
    return artifacts.weather_by_week.get((lag_year, lag_week), {})


def _build_feature_row(request: ForecastRequest, artifacts: ForecastArtifacts) -> pd.DataFrame:
    values = {column: 0.0 for column in artifacts.feature_cols}
    category_values = {
        "Customer": request.customer,
        "Plant_Variety": request.plant_variety,
        "Size": request.size,
        "Mode": request.mode,
    }
    for column, value in category_values.items():
        if value not in artifacts.category_values[column]:
            raise HTTPException(status_code=422, detail=f"Unknown {column} value: {value}")
        dummy_name = f"{column}_{value}"
        if dummy_name in values:  # The dropped first category is represented by all zeros.
            values[dummy_name] = 1.0

    max_temp = float(request.avg_max_temp_c)
    min_temp = float(request.avg_min_temp_c)
    rainfall = float(request.weekly_rainfall_mm)
    degree_days = max(0.0, ((max_temp + min_temp) / 2.0 - 15.0) * 7.0)
    raw_numeric = {
        "Ordered_Quantity": float(request.ordered_quantity),
        "Avg_Max_Temp_C": max_temp,
        "Avg_Min_Temp_C": min_temp,
        "Weekly_Rainfall_mm": rainfall,
        "Temp_Diff_C": max_temp - min_temp,
        "Growing_Degree_Days_Weekly": degree_days,
        "WEEK_sin": math.sin(2.0 * math.pi * request.week / 52.0),
        "WEEK_cos": math.cos(2.0 * math.pi * request.week / 52.0),
    }

    for lag in (2, 4):
        lagged = _lagged_weather(artifacts, request.year, request.week, lag)
        for feature, weather_feature in (
            (f"Avg_Max_Temp_C_Lag{lag}W", "Avg_Max_Temp_C"),
            (f"Weekly_Rainfall_mm_Lag{lag}W", "Weekly_Rainfall_mm"),
            (f"Growing_Degree_Days_Weekly_Lag{lag}W", "Growing_Degree_Days_Weekly"),
        ):
            raw_numeric[feature] = _finite_or_nan(lagged.get(weather_feature))

    for column, number in raw_numeric.items():
        if column in values:
            values[column] = number
        flag = f"{column}_missing"
        if flag in values:
            values[flag] = float(not math.isfinite(number))

    row = pd.DataFrame([values], columns=artifacts.feature_cols)
    numeric_frame = row.reindex(columns=artifacts.numeric_cols).astype(float)
    for column in artifacts.numeric_cols:
        numeric_frame[column] = numeric_frame[column].fillna(artifacts.imputation_medians[column])
    row.loc[:, artifacts.numeric_cols] = artifacts.scaler.transform(numeric_frame)
    if not np.isfinite(row.to_numpy(dtype=np.float64)).all():
        raise HTTPException(status_code=503, detail="C4 feature preparation produced invalid values.")
    return row


def predict(request: ForecastRequest) -> ForecastResponse:
    artifacts = load_artifacts()
    row = _build_feature_row(request, artifacts)
    model_input = row.loc[:, artifacts.feature_cols].to_numpy(dtype=np.float32)

    started = time.perf_counter()
    prediction = float(artifacts.model.predict(model_input)[0])
    inference_ms = (time.perf_counter() - started) * 1000.0
    if not math.isfinite(prediction):
        raise HTTPException(status_code=503, detail="The C4 model returned an invalid prediction.")

    dispatched = max(0, int(round(prediction)))
    gap = float(dispatched - request.ordered_quantity)
    status = "Deficit / Risk" if gap < 0 else "Surplus" if gap > 0 else "Balanced"
    fulfillment = (dispatched / request.ordered_quantity * 100.0
                   if request.ordered_quantity > 0 else None)
    return ForecastResponse(
        customer=request.customer,
        plant_variety=request.plant_variety,
        year=request.year,
        week=request.week,
        ordered_quantity=request.ordered_quantity,
        predicted_dispatched_quantity=dispatched,
        supply_demand_gap=gap,
        fulfillment_rate_percent=round(fulfillment, 1) if fulfillment is not None else None,
        status=status,
        model_used=MODEL_LABEL,
        inference_latency_ms=round(inference_ms, 3),
    )
