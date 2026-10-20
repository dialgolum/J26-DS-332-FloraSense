"""Bounded, ephemeral front-image upload; no workbook or saved-session writes."""
import threading

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

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
