"""Pydantic request schemas."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DiagnoseOptions(BaseModel):
    species: Optional[str] = Field(None, max_length=80)
    plant_id: Optional[str] = Field(None, max_length=80)
    top_k: int = Field(3, ge=1, le=10)
    submit_to_review: bool = Field(True)


class DiagnoseJsonRequest(DiagnoseOptions):
    image_base64: str = Field(..., min_length=16)


class ReviewResolveRequest(BaseModel):
    label: str = Field(..., min_length=2, max_length=40)
    reviewer: str = Field(..., min_length=2, max_length=80)
    notes: Optional[str] = Field(None, max_length=500)
    add_to_prototype: bool = Field(True)