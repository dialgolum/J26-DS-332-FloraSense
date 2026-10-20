"""Read-only C3 reference-data and saved height-session endpoints."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data_loader import ROOT, load_dataset, resolve_image_path
from ..p01_height_measurement import (
    RESULTS_DIR,
    load_buyer_range,
    load_measurement_session,
    reevaluate_session,
)

router = APIRouter(prefix="/api/v1/c3", tags=["C3 quality assessment"])
IMAGE_ROOT = (ROOT / "data" / "quality_assessment" / "original").resolve()
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")


class SessionReevaluationRequest(BaseModel):
    requirement_id: str = Field(min_length=1, max_length=100)


def get_sheets() -> dict[str, pd.DataFrame]:
    """Dependency for workbook loading; overrideable in API tests."""
    try:
        return load_dataset()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="C3 workbook data could not be loaded.") from exc


def get_results_dir() -> Path:
    """Dependency for the saved-session directory; overrideable in API tests."""
    return RESULTS_DIR


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): _json_value(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def _find_row(frame: pd.DataFrame, column: str, value: str, label: str) -> pd.Series:
    selected = frame[frame[column].astype(str).str.strip() == value]
    if len(selected) != 1:
        raise HTTPException(status_code=404, detail=f"{label} not found: {value}")
    return selected.iloc[0]


def _validated_plant(sheets: dict[str, pd.DataFrame], plant_id: str) -> None:
    _find_row(sheets["Plant_Types"], "Plant_ID", plant_id, "Plant type")


def _validated_sample(sheets: dict[str, pd.DataFrame], sample_id: str) -> pd.Series:
    return _find_row(sheets["Samples"], "Sample_ID", sample_id, "Sample")


def _session_path(results_dir: Path, session_id: str) -> Path:
    if not SESSION_ID_PATTERN.fullmatch(session_id) or Path(session_id).name != session_id:
        raise HTTPException(status_code=400, detail="Invalid height session reference.")
    root = results_dir.resolve()
    path = (root / session_id).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Invalid height session reference.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Height session not found: {session_id}")
    return path


@router.get("/plants")
def list_plants(sheets: dict[str, pd.DataFrame] = Depends(get_sheets)) -> dict:
    return {"data": _records(sheets["Plant_Types"]), "meta": {"count": len(sheets["Plant_Types"])}}


@router.get("/requirements")
def list_requirements(
    plant_id: str = Query(min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    _validated_plant(sheets, plant_id)
    rows = sheets["Requirement_Profiles"]
    result = rows[rows["Plant_ID"].astype(str).str.strip() == plant_id]
    return {"data": _records(result), "meta": {"count": len(result), "plant_id": plant_id}}


@router.get("/samples")
def list_samples(
    plant_id: str = Query(min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    _validated_plant(sheets, plant_id)
    rows = sheets["Samples"]
    result = rows[rows["Plant_ID"].astype(str).str.strip() == plant_id]
    return {"data": _records(result), "meta": {"count": len(result), "plant_id": plant_id}}


def _sample_images(sheets: dict[str, pd.DataFrame], sample_id: str) -> pd.DataFrame:
    _validated_sample(sheets, sample_id)
    images = sheets["Images"]
    return images[images["Sample_ID"].astype(str).str.strip() == sample_id]


@router.get("/samples/{sample_id}/images")
def list_sample_images(
    sample_id: str,
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    rows = _sample_images(sheets, sample_id)
    data = []
    for image in _records(rows):
        image_id = str(image["Image_ID"])
        image["preview_url"] = f"/api/v1/c3/images/{image_id}/preview"
        data.append(image)
    return {"data": data, "meta": {"count": len(data), "sample_id": sample_id}}


@router.get("/images/{image_id}/preview")
def preview_image(image_id: str, sheets: dict[str, pd.DataFrame] = Depends(get_sheets)) -> FileResponse:
    row = _find_row(sheets["Images"], "Image_ID", image_id, "Image")
    if pd.isna(row["Relative_Path"]) or not str(row["Relative_Path"]).strip():
        raise HTTPException(status_code=404, detail=f"Image path is missing for {image_id}.")
    image_path = resolve_image_path(row["Relative_Path"]).resolve()
    if not image_path.is_relative_to(IMAGE_ROOT):
        raise HTTPException(status_code=403, detail="Image is outside approved project image directories.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file is missing for {image_id}.")
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported image type.")
    return FileResponse(
        image_path,
        media_type=media_type,
        filename=image_path.name,
        content_disposition_type="inline",
    )


@router.get("/samples/{sample_id}/height-sessions")
def list_height_sessions(
    sample_id: str,
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
    results_dir: Path = Depends(get_results_dir),
) -> dict:
    _validated_sample(sheets, sample_id)
    sessions = []
    invalid_count = 0
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json"), key=lambda item: item.name):
            try:
                session = load_measurement_session(path)
            except ValueError:
                invalid_count += 1
                continue
            if str(session.get("sample_id", "")).strip() == sample_id:
                sessions.append({
                    "session_id": path.name,
                    "sample_id": sample_id,
                    "image_id": session.get("image_id"),
                    "requirement_id": session.get("requirement_id"),
                    "timestamp": session.get("timestamp"),
                    "leaf_count": len(session.get("leaves", [])),
                    "scale_px_per_cm": session.get("scale_px_per_cm"),
                    "calibration_verified": session.get("calibration_verified"),
                    "scale_reliable": session.get("scale_reliable"),
                })
    return {"data": sessions, "meta": {"count": len(sessions), "sample_id": sample_id,
                                          "invalid_session_files_skipped": invalid_count}}


def _annotation_context(sheets: dict, sample_id: str, requirement_id: str) -> pd.Series:
    sample = _validated_sample(sheets, sample_id)
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID",
                            requirement_id, "Requirement profile")
    if str(sample["Plant_ID"]).strip() != str(requirement["Plant_ID"]).strip():
        raise HTTPException(status_code=422, detail="Requirement profile belongs to a different plant type.")
    return sample


@router.get("/samples/{sample_id}/height-assessments")
def recorded_height_assessments(
    sample_id: str,
    requirement_id: str = Query(min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    _annotation_context(sheets, sample_id, requirement_id)
    rows = sheets["Height_Assessments"]
    rows = rows[(rows["Sample_ID"].astype(str).str.strip() == sample_id)
                & (rows["Requirement_ID"].astype(str).str.strip() == requirement_id)]
    return {"data": _records(rows), "meta": {
        "source": "recorded_manual_annotation", "sheet": "Height_Assessments",
        "sample_id": sample_id, "requirement_id": requirement_id, "count": len(rows),
    }}


@router.get("/samples/{sample_id}/quality-annotations")
def recorded_quality_annotations(
    sample_id: str,
    requirement_id: str = Query(min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    sample = _annotation_context(sheets, sample_id, requirement_id)
    rows = sheets["Quality_Annotations"]
    rows = rows[rows["Sample_ID"].astype(str).str.strip() == sample_id]
    return {"data": _records(rows), "meta": {
        "source": "recorded_manual_annotation", "sheet": "Quality_Annotations",
        "scope": "sample", "sample_id": sample_id, "requirement_id": requirement_id,
        "count": len(rows), "sample_observations": _records(sample.to_frame().T)[0],
    }}


@router.post("/height-sessions/{session_id}/reevaluate")
def reevaluate_height_session(
    session_id: str,
    request: SessionReevaluationRequest,
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
    results_dir: Path = Depends(get_results_dir),
) -> dict:
    path = _session_path(results_dir, session_id)
    try:
        session = load_measurement_session(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sample_id = str(session.get("sample_id", "")).strip()
    sample = _validated_sample(sheets, sample_id)
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID",
                            request.requirement_id.strip(), "Requirement profile")
    if str(requirement["Plant_ID"]).strip() != str(sample["Plant_ID"]).strip():
        raise HTTPException(status_code=422, detail="Requirement profile belongs to a different plant type.")

    try:
        result = reevaluate_session(path, request.requirement_id.strip())
        buyer_range = load_buyer_range(request.requirement_id.strip())
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    leaves = []
    for index, leaf in enumerate(session["leaves"], start=1):
        height = leaf["height_cm"]
        valid = isinstance(height, (int, float)) and height > 0
        leaves.append({
            "leaf_number": leaf.get("leaf_number", index),
            "emergence_px": leaf["emergence_px"],
            "tip_px": leaf["tip_px"],
            "height_cm": height,
            "within_requirement": (
                bool(buyer_range.minimum_cm <= height <= buyer_range.maximum_cm)
                if valid else None
            ),
        })
    return {
        "data": {
            "assessment_type": "Height Assessment",
            "sample_id": sample_id,
            "plant_id": str(sample["Plant_ID"]).strip(),
            "image_id": session.get("image_id"),
            "requirement_id": result["Requirement_ID"],
            "buyer_id": result["Buyer_ID"],
            "range_cm": result["Range_cm"],
            "leaf_count": len(session["leaves"]),
            "valid_leaves": result["Valid_Leaves"],
            "leaves_in_range": result["Leaves_In_Range"],
            "percentage_in_range": result["Percentage_In_Range"],
            "status": result["Status"],
            "reasons": result["Reasons"],
            "scale_px_per_cm": session["scale_px_per_cm"],
            "calibration_verified": session["calibration_verified"],
            "scale_reliable": session["scale_reliable"],
            "session_id": path.name,
            "leaves": leaves,
        },
        "meta": {"source": "saved_height_measurement_session"},
    }
