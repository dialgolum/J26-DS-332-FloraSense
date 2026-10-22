"""Dynamic registration of new disease prototypes."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile

from services.c2_disease_detection import config
from services.c2_disease_detection.api.schemas.responses import (
    EnrollmentResponse,
    PrototypeInfo,
    PrototypeListResponse,
)
from services.c2_disease_detection.staging.dynamic_enrollment import DynamicEnrollment, EnrollmentError

router = APIRouter(tags=["registration"])


def require_admin(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if x_api_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


def get_engine(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")
    return engine


async def read_uploads(files: List[UploadFile]) -> List[bytes]:
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    out = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"File '{f.filename}' is empty.")
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File '{f.filename}' exceeds {config.MAX_UPLOAD_MB} MB.")
        out.append(data)
    return out


def _to_response(engine, summary: dict) -> EnrollmentResponse:
    return EnrollmentResponse(**summary, total_classes=len(engine.store))


@router.post("/register-disease", response_model=EnrollmentResponse, dependencies=[Depends(require_admin)])
async def register_disease(
    request: Request,
    class_name: str = Form(...),
    display_name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    engine = get_engine(request)
    images = await read_uploads(files)
    enroller = DynamicEnrollment(engine.embed_images, engine.store)
    try:
        summary = enroller.enroll_new_class(class_name, images, display_name)
    except EnrollmentError as exc:
        code = 409 if "already exists" in str(exc) else 422
        raise HTTPException(status_code=code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    engine.save_prototypes()
    return _to_response(engine, summary)


@router.post("/prototypes/{name}/support", response_model=EnrollmentResponse, dependencies=[Depends(require_admin)])
async def add_support(request: Request, name: str, files: List[UploadFile] = File(...)):
    engine = get_engine(request)
    images = await read_uploads(files)
    try:
        summary = DynamicEnrollment(engine.embed_images, engine.store).add_to_class(name, images)
    except EnrollmentError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    engine.save_prototypes()
    return _to_response(engine, summary)


@router.get("/prototypes", response_model=PrototypeListResponse)
def list_prototypes(request: Request):
    engine = get_engine(request)
    recs = engine.store.records()
    return PrototypeListResponse(
        count=len(recs),
        prototypes=[
            PrototypeInfo(name=r.name, display_name=r.display_name, n_support=r.n_support,
                          kind=r.kind, updated_at=r.updated_at)
            for r in recs
        ],
    )


@router.delete("/prototypes/{name}", dependencies=[Depends(require_admin)])
def delete_prototype(request: Request, name: str):
    engine = get_engine(request)
    if name in config.CORE_CLASSES:
        raise HTTPException(status_code=403, detail="Core classes cannot be deleted.")
    try:
        engine.store.remove_class(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown class '{name}'.")
    engine.save_prototypes()
    return {"deleted": name, "remaining": len(engine.store)}