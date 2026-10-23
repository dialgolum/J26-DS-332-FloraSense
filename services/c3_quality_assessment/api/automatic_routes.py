"""Bounded, ephemeral front-image upload; no workbook or saved-session writes."""
import threading
import base64
import binascii
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from .routes import get_sheets, get_ai_config, _validated_plant, _find_row
from ..model_inference import predict_feature

from ..automatic_leaf import MAX_UPLOAD_BYTES, InvalidImage, analyze_image, decode_image

router = APIRouter(prefix="/api/v1/c3", tags=["C3 automatic prototype"])
_capacity = threading.BoundedSemaphore(2)


def _infer(payload, content_type, grid_profile_confirmed):
    image, native_size = decode_image(payload, content_type)
    return analyze_image(image, native_size, grid_profile_confirmed=grid_profile_confirmed)


@router.post("/automatic/front-image")
async def inspect_front_image(
    request: Request,
    grid_profile_confirmed: bool = Query(default=False),
) -> dict:
    """Raw JPEG/PNG request body, not multipart; uploads are not persisted."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only image/jpeg and image/png request bodies are accepted.")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared_size = int(length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length.") from exc
        if declared_size < 0:
            raise HTTPException(400, "Invalid Content-Length.")
        if declared_size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 25 MiB limit.")
    if not _capacity.acquire(blocking=False):
        raise HTTPException(429, "Two image inspections are already running. Try again shortly.")
    try:
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Image exceeds the 25 MiB limit.")
            payload.extend(chunk)
        try:
            result = await run_in_threadpool(_infer, bytes(payload), content_type,
                                             grid_profile_confirmed)
        except InvalidImage as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return {"data": result, "meta": {"persisted": False, "phase": 1,
                "source": "uploaded_image_opencv_prototype"}}
    finally:
        _capacity.release()


MAX_INSPECTION_BYTES = 3 * ((MAX_UPLOAD_BYTES + 2) // 3 * 4) + 4096


def _inspect_upload(payload, sheets, config):
    if not isinstance(payload, dict):
        raise HTTPException(422, "Expected an inspection object.")
    plant_id = payload.get("plant_id")
    requirement_id = payload.get("requirement_id")
    if not isinstance(plant_id, str) or plant_id not in {f"P{i:02d}" for i in range(1, 9)} or not isinstance(requirement_id, str):
        raise HTTPException(422, "Choose a valid plant variety and buyer requirement.")
    _validated_plant(sheets, plant_id)
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID", requirement_id, "Requirement profile")
    if str(requirement.Plant_ID).strip() != plant_id:
        raise HTTPException(422, "Requirement profile belongs to a different plant type.")
    images = payload.get("images")
    if not isinstance(images, dict) or not {"Front", "Root"}.issubset(images):
        raise HTTPException(422, "Front and Root images are required.")
    if set(images) - {"Front", "Root", "Stem"}:
        raise HTTPException(422, "Only Front, Root and Stem images are accepted.")
    # Validate every image before inference; never read sample annotations or measurements.
    with tempfile.TemporaryDirectory(prefix="c3-inspection-") as directory:
        paths = {}
        for view, image in images.items():
            if not isinstance(image, dict) or not isinstance(image.get("data"), str):
                raise HTTPException(422, f"Invalid {view} image.")
            if len(image["data"]) > ((MAX_UPLOAD_BYTES + 2) // 3 * 4):
                raise HTTPException(413, "Image exceeds the 25 MiB limit.")
            try:
                raw = base64.b64decode(image["data"], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise HTTPException(422, f"Invalid {view} image encoding.") from exc
            content_type = image.get("content_type")
            if not isinstance(content_type, str) or content_type not in {"image/jpeg", "image/png"}:
                raise HTTPException(415, "Only JPEG and PNG images are accepted.")
            decode_image(raw, content_type)
            path = Path(directory) / (view + (".jpg" if content_type == "image/jpeg" else ".png"))
            path.write_bytes(raw)
            paths[view] = path
        observations = {}
        tasks = [("root_B", "root", "Root")]
        if plant_id in {f"P{i:02d}" for i in range(2, 8)}:
            tasks.append(("leaf_B", "leaf", "Front"))
        for key, task, view in tasks:
            observation = {"status": "unavailable", "advisory": True,
                           "reason": "Model unavailable. Manual inspection required."}
            if plant_id == "P08":
                observation.update(status="unsupported", reason="No compatible model evidence for this plant variety.")
            elif config.get(f"{task}_checkpoint"):
                try:
                    prediction = predict_feature(paths[view], config[f"{task}_checkpoint"], task,
                                                 variant="B", device=config.get("device", "cpu"))
                    labels = {"root": {"Acceptable", "Not Acceptable"}, "leaf": {"Both Sides", "One Side"}}
                    if prediction.get("status") == "success" and prediction.get("predicted_label") in labels[task]:
                        observation = {"status": "success", "advisory": True,
                                       "prediction": {"predicted_label": prediction["predicted_label"]}}
                except Exception:
                    # Never return checkpoint paths or internal inference diagnostics.
                    pass
            observations[key] = observation
    checks = ["Height"] + (["Leaf count"] if plant_id in {"P01", "P08"} else []) + ["Overall visual quality"]
    return {"Plant_ID": plant_id, "Requirement_ID": requirement_id,
            "Buyer_ID": str(requirement.Buyer_ID), "Buyer_Rule_Result": "REVIEW",
            "AI_Observations": observations,
            "Checks": [{"Check": name, "Presentation_Status": "PENDING"} for name in checks],
            "source": "uploaded_plant", "advisory": True}


@router.post("/inspections/upload")
async def inspect_uploaded_plant(request: Request, sheets=Depends(get_sheets), config=Depends(get_ai_config)):
    """Bounded JSON/base64 transport avoids a new multipart dependency."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise HTTPException(415, "Expected an application/json image upload.")
    if not _capacity.acquire(blocking=False):
        raise HTTPException(429, "Two image inspections are already running. Try again shortly.")
    try:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_INSPECTION_BYTES:
                raise HTTPException(413, "Inspection upload exceeds the total size limit.")
            body.extend(chunk)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(422, "Invalid inspection upload.") from exc
        try:
            result = await run_in_threadpool(_inspect_upload, payload, sheets, config)
        except InvalidImage as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return {"data": result, "meta": {"persisted": False}}
    finally:
        _capacity.release()
