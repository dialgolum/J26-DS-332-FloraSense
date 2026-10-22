"""Diagnosis endpoints + review endpoints."""
from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from services.c2_disease_detection import config
from services.c2_disease_detection.api.routes.register_disease import get_engine, require_admin
from services.c2_disease_detection.api.schemas.requests import DiagnoseJsonRequest, DiagnoseOptions, ReviewResolveRequest
from services.c2_disease_detection.api.schemas.responses import (
    ClassScore, Diagnosis, DiagnoseResponse, GateVerdict, ModelInfo, ReviewListResponse, ReviewTicket, TelemetryRecord
)
from services.c2_disease_detection.gating.threshold_gate import advisory_for, export_quality_flag
from services.c2_disease_detection.staging.dynamic_enrollment import DynamicEnrollment, EnrollmentError

router = APIRouter(prefix="/diagnose", tags=["diagnosis"])
review_router = APIRouter(prefix="/review", tags=["review"])


def get_queue(request: Request):
    return request.app.state.review_queue


def emit_telemetry(record: TelemetryRecord) -> None:
    payload = record.model_dump()
    try:
        config.TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.TELEMETRY_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    if config.C4_TELEMETRY_URL:
        try:
            httpx.post(config.C4_TELEMETRY_URL, json=payload, timeout=3.0)
        except httpx.HTTPError:
            pass


def _run_diagnosis(request: Request, image_bytes: bytes, opts: DiagnoseOptions, filename: Optional[str],
                   background: BackgroundTasks) -> DiagnoseResponse:
    engine = get_engine(request)
    if len(engine.store) == 0:
        raise HTTPException(status_code=503, detail="No disease prototypes registered.")
    if len(image_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {config.MAX_UPLOAD_MB} MB.")
    try:
        result = engine.diagnose(image_bytes, top_k=opts.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    d = result.decision
    ticket_id = None
    if not d.accepted and opts.submit_to_review:
        ticket = get_queue(request).submit(image_bytes, d.top_prediction, d.confidence, d.probabilities,
                                           opts.species, opts.plant_id, filename)
        ticket_id = ticket["ticket_id"]

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    background.add_task(emit_telemetry, TelemetryRecord(species=opts.species, disease_class=d.disease_class,
                                                        confidence=round(d.confidence, 4), timestamp=stamp))
    names = engine.store.names
    return DiagnoseResponse(
        request_id=uuid.uuid4().hex[:12],
        timestamp=stamp,
        species=opts.species,
        plant_id=opts.plant_id,
        diagnosis=Diagnosis(disease_class=d.disease_class, top_prediction=d.top_prediction,
                            top_k=[ClassScore(**c) for c in d.top_k], probabilities=d.probabilities),
        gate=GateVerdict(status=d.status, tau=d.tau, accepted=d.accepted, confidence=d.confidence,
                         margin=d.margin, entropy=d.entropy, automated_decision=d.accepted),
        export_quality_flag=export_quality_flag(d),
        advisory=advisory_for(d),
        review_ticket_id=ticket_id,
        latency_ms=round(result.latency_ms, 1),
        model=ModelInfo(backbone=engine.model.backbone_name, metric=engine.metric,
                        temperature=engine.temperature, n_classes=len(names), classes=names),
    )


@router.post("", response_model=DiagnoseResponse)
async def diagnose(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    species: Optional[str] = Form(None),
    plant_id: Optional[str] = Form(None),
    top_k: int = Form(3),
    submit_to_review: bool = Form(True),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    try:
        opts = DiagnoseOptions(species=species, plant_id=plant_id, top_k=top_k, submit_to_review=submit_to_review)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _run_diagnosis(request, data, opts, file.filename, background)


@router.post("/json", response_model=DiagnoseResponse)
def diagnose_json(request: Request, body: DiagnoseJsonRequest, background: BackgroundTasks):
    try:
        raw = body.image_base64.split(",", 1)[-1]
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="image_base64 is not valid base64.")
    return _run_diagnosis(request, data, body, None, background)


@review_router.get("", response_model=ReviewListResponse)
def list_review(request: Request, status: Optional[str] = None):
    q = get_queue(request)
    return ReviewListResponse(stats=q.stats(), tickets=[ReviewTicket(**t) for t in q.list(status)])


@review_router.get("/{ticket_id}/image")
def review_image(request: Request, ticket_id: str):
    q = get_queue(request)
    try:
        path = q.image_path(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown ticket.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing.")
    return FileResponse(path)


@review_router.post("/{ticket_id}/resolve", response_model=ReviewTicket, dependencies=[Depends(require_admin)])
def resolve_review(request: Request, ticket_id: str, body: ReviewResolveRequest):
    q, engine = get_queue(request), get_engine(request)
    label = body.label.strip().lower()
    if label != "discard" and label not in engine.store:
        raise HTTPException(status_code=422, detail=f"'{label}' is not a registered class.")
    try:
        if q.get(ticket_id)["status"] == "RESOLVED":
            raise ValueError(f"Ticket '{ticket_id}' is already resolved")
        if label != "discard" and body.add_to_prototype:
            image = q.image_path(ticket_id).read_bytes()
            DynamicEnrollment(engine.embed_images, engine.store).add_to_class(label, [image])
            engine.save_prototypes()
        ticket = q.resolve(ticket_id, label, body.reviewer, body.notes)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown ticket.")
    except (ValueError, EnrollmentError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return ReviewTicket(**ticket)