"""Request and response schemas for the C4 forecasting API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

current_year = datetime.now().year


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: str = Field(min_length=1, max_length=120)
    plant_variety: str = Field(min_length=1, max_length=160)
    size: str = Field(min_length=1, max_length=80)
    mode: str = Field(min_length=1, max_length=80)
    ordered_quantity: float = Field(gt=0, allow_inf_nan=False)
    year: int = Field(
        ...,
        ge=current_year,
        le=current_year + 1,
        description="Forecast year must be the current year or the next year.",
    )
    week: int = Field(ge=1, le=52)
    avg_max_temp_c: float = Field(ge=-20, le=70, allow_inf_nan=False)
    avg_min_temp_c: float = Field(ge=-20, le=70, allow_inf_nan=False)
    weekly_rainfall_mm: float = Field(ge=0, le=5000, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_temperature_range(self) -> "ForecastRequest":
        if self.avg_max_temp_c <= self.avg_min_temp_c:
            raise ValueError("avg_max_temp_c must be greater than avg_min_temp_c")
        return self


class ForecastResponse(BaseModel):
    customer: str
    plant_variety: str
    year: int
    week: int
    ordered_quantity: float
    predicted_dispatched_quantity: int
    supply_demand_gap: float
    fulfillment_rate_percent: float | None
    status: str
    model_used: str
    inference_latency_ms: float
