"""Read-only C3 reference-data and saved height-session endpoints."""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data_loader import ROOT, load_dataset, resolve_image_path
from ..run_assessment import run_assessment
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


def _validated_sample(sheets: dict[str, pd.DataFrame], sample_id: str, plant_id: str | None = None) -> pd.Series:
    rows = sheets["Samples"]
    if plant_id is not None:
        _validated_plant(sheets, plant_id)
        rows = rows[rows.Plant_ID.eq(plant_id)]
    return _find_row(rows, "Sample_ID", sample_id, "Sample")


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


def _sample_images(sheets: dict[str, pd.DataFrame], sample_id: str, plant_id: str | None = None) -> pd.DataFrame:
    sample = _validated_sample(sheets, sample_id, plant_id)
    images = sheets["Images"]
    return images[images.Sample_ID.eq(sample_id) & images.Plant_ID.eq(sample.Plant_ID)]


@router.get("/samples/{sample_id}/images")
def list_sample_images(
    sample_id: str,
    plant_id: str | None = Query(default=None, min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
) -> dict:
    rows = _sample_images(sheets, sample_id, plant_id)
    data = []
    for image in _records(rows):
        image.pop("Relative_Path", None)
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
    plant_id: str | None = Query(default=None, min_length=1),
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
    results_dir: Path = Depends(get_results_dir),
) -> dict:
    sample = _validated_sample(sheets, sample_id, plant_id)
    # Saved click-measurement sessions belong to the existing P01 workflow.
    if sample.Plant_ID != "P01":
        return {"data": [], "meta": {"count": 0, "sample_id": sample_id}}
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
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID",
                            requirement_id, "Requirement profile")
    sample = _validated_sample(sheets, sample_id, requirement.Plant_ID)
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
    rows = rows[rows.Sample_ID.eq(sample_id) & rows.Plant_ID.eq(sample.Plant_ID)]
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
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID",
                            request.requirement_id.strip(), "Requirement profile")
    sample = _validated_sample(sheets, sample_id, "P01")
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


class AssessmentRequest(BaseModel):
    # Paths are server configuration, never arbitrary client input.
    model_config = {"extra": "forbid"}
    plant_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    sample_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    requirement_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    ai_enabled: bool = False
    compare_leaf_a: bool = False


def get_ai_config() -> dict:
    """Explicit server-side checkpoints; no default/best-fold selection."""
    return {"root_checkpoint": os.getenv("C3_ROOT_B_CHECKPOINT") or None,
            "leaf_checkpoint": os.getenv("C3_LEAF_B_CHECKPOINT") or None,
            "leaf_a_checkpoint": os.getenv("C3_LEAF_A_CHECKPOINT") or None,
            "device": os.getenv("C3_INFERENCE_DEVICE", "cpu")}


def _presentation_check(check: dict, missing_measurement: bool | None = None) -> dict:
    """Add presentation fields without changing original status/reason."""
    status, reason = check["Status"], check["Reason"].casefold()
    pending = status == "MANUAL REVIEW" and any(
        token in reason for token in ("missing", "incomplete", "not implemented", "not assessed"))
    if missing_measurement is not None:
        pending = status == "MANUAL REVIEW" and missing_measurement
    return {**check, "Presentation_Status": (
        status if status in {"PASS", "FAIL"} else "PENDING" if pending else "REVIEW"),
        "Missing_Measurement": pending and any(
            token in check["Check"].casefold() for token in ("height", "count", "measurement"))}


def _public_assessment(result: dict, sheets: dict) -> dict:
    """Strip local paths and diagnostic exception text at the HTTP boundary."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in {"image_path", "checkpoint_path", "Relative_Path", "Image"}}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return re.sub(r"(?<!\S)(?:[A-Za-z]:[\\/]|\\\\|/)[^\s\"'<>]+", "[redacted]", value)
        return _json_value(value)

    public = clean(result)
    heights = sheets["Height_Assessments"]
    heights = heights[heights.Sample_ID.eq(result["Sample_ID"]) & heights.Requirement_ID.eq(result["Requirement_ID"])]
    missing_height = heights.empty or (len(heights) == 1 and any(
        pd.isna(heights.iloc[0].get(column)) for column in ("Leaves_In_Range", "Total_Leaves_Assessed")))
    annotations = sheets["Quality_Annotations"]
    annotations = annotations[annotations.Sample_ID.eq(result["Sample_ID"]) & annotations.Plant_ID.eq(result["Plant_ID"])]
    missing_leaf_count = (len(annotations) == 1 and (
        pd.isna(annotations.iloc[0].get("Total_Leaf_Count")) or
        pd.isna(result["Requirement_Profile"].get("Min_Total_Leaf_Count"))))
    missing = {"Height": missing_height, "Leaf count": missing_leaf_count,
               "Visual quality": False}  # Missing/duplicated annotations need expert review, not a measurement.
    public["Checks"] = [_presentation_check(check, missing.get(check["Check"])) for check in public["Checks"]]
    public["Rule_Assessment"]["Checks"] = [dict(check) for check in public["Checks"]]
    public["Missing_Measurements"] = [check["Check"] for check in public["Checks"]
                                      if check["Missing_Measurement"]]
    reasons = [f"Rule {check['Check']}: {check['Reason']}" for check in public["Checks"]
               if check["Status"] == "MANUAL REVIEW"]
    for key, observation in public["AI_Observations"].items():
        if observation["status"] in {"error", "unavailable"}:
            observation["reason"] = ("Requested AI observation is unavailable; verify server checkpoint and sample image configuration."
                                     if observation["status"] == "unavailable" else
                                     "AI inference failed; buyer-rule decision retained.")
            prediction = observation.get("prediction")
            if prediction and "error" in prediction:
                prediction["error"] = {"code": "FEATURE_INFERENCE_ERROR", "message": observation["reason"]}
        if observation.get("requires_manual_review"):
            reasons.append(f"{key}: {observation['reason']}")
            if observation.get("comparison_status", "").startswith("expert_annotation_"):
                reasons.append(f"{key}: Expert annotation is unavailable or unclear; AI does not replace it.")
            if observation.get("comparison_status") == "disagreement":
                reasons.append(f"{key}: AI disagrees with the expert annotation.")
    public["Manual_Review"]["reasons"] = reasons
    public["Manual_Review"]["required"] = bool(reasons)
    return public


@router.post("/assessments")
def assess_selected_sample(
    request: AssessmentRequest,
    sheets: dict[str, pd.DataFrame] = Depends(get_sheets),
    config: dict = Depends(get_ai_config),
) -> dict:
    _validated_plant(sheets, request.plant_id)
    requirement = _find_row(sheets["Requirement_Profiles"], "Requirement_ID",
                            request.requirement_id, "Requirement profile")
    if requirement["Plant_ID"] != request.plant_id:
        raise HTTPException(422, "Requirement profile belongs to a different plant type.")
    samples = sheets["Samples"]
    samples = samples[samples.Plant_ID.eq(request.plant_id) & samples.Sample_ID.eq(request.sample_id)]
    if samples.empty:
        raise HTTPException(404, "Sample not found for the selected plant type.")
    if len(samples) != 1:
        raise HTTPException(409, "Selected composite sample identity is duplicated.")
    if request.compare_leaf_a and not request.ai_enabled:
        raise HTTPException(422, "Leaf A comparison requires AI to be enabled.")
    kwargs = dict(config)
    kwargs["leaf_a_checkpoint"] = config.get("leaf_a_checkpoint") if request.compare_leaf_a else None
    if request.compare_leaf_a and not kwargs["leaf_a_checkpoint"]:
        # Include the requested comparison as an unavailable advisory observation.
        kwargs["leaf_a_checkpoint"] = ""
    try:
        result = run_assessment(sheets, request.sample_id, request.requirement_id,
                                ai_enabled=request.ai_enabled, **kwargs)
        return {"data": _public_assessment(result, sheets), "error": None}
    except Exception as exc:
        raise HTTPException(503, "C3 assessment data could not be evaluated.") from exc
