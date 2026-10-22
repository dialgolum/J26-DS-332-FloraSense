"""Pydantic response schemas."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ExportFlag = Literal["PASS", "FAIL", "QUARANTINE"]


class ClassScore(BaseModel):
    label: str
    probability: float


class GateVerdict(BaseModel):
    status: Literal["CONFIRMED", "UNKNOWN_ANOMALY"]
    tau: float
    accepted: bool
    confidence: float
    margin: float
    entropy: float
    automated_decision: bool


class Diagnosis(BaseModel):
    disease_class: str
    top_prediction: str
    top_k: List[ClassScore]
    probabilities: Dict[str, float]


class ModelInfo(BaseModel):
    backbone: str
    metric: str
    temperature: float
    n_classes: int
    classes: List[str]


class DiagnoseResponse(BaseModel):
    request_id: str
    timestamp: str
    species: Optional[str] = None
    plant_id: Optional[str] = None
    diagnosis: Diagnosis
    gate: GateVerdict
    export_quality_flag: ExportFlag
    advisory: str
    review_ticket_id: Optional[str] = None
    latency_ms: float
    model: ModelInfo


class TelemetryRecord(BaseModel):
    species: Optional[str]
    disease_class: str
    confidence: float
    timestamp: str


class PrototypeInfo(BaseModel):
    name: str
    display_name: str
    n_support: int
    kind: str
    updated_at: str


class PrototypeListResponse(BaseModel):
    count: int
    prototypes: List[PrototypeInfo]


class EnrollmentResponse(BaseModel):
    class_name: str
    n_support: int
    kind: str
    cohesion: float
    nearest_class: Optional[str]
    nearest_similarity: Optional[float]
    warnings: List[str]
    total_classes: int


class ReviewTicket(BaseModel):
    ticket_id: str
    status: str
    created_at: str
    species: Optional[str] = None
    plant_id: Optional[str] = None
    image_file: str
    top_prediction: str
    confidence: float
    probabilities: Dict[str, float]
    resolution: Optional[dict] = None


class ReviewListResponse(BaseModel):
    stats: Dict[str, int]
    tickets: List[ReviewTicket]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    prototypes_loaded: int
    tau: float
    temperature: Optional[float] = None
    review_pending: int = 0