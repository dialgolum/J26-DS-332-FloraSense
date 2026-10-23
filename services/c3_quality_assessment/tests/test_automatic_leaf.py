"""Focused safety, segmentation, calibration and abstention tests for Phase 1."""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.c3_quality_assessment.api.main import app
from services.c3_quality_assessment.api import automatic_routes
from services.c3_quality_assessment.automatic_leaf import (
    InvalidImage, analyze_image, decode_image, detect_grid, leaf_regions, vegetation_mask,
)
from services.c3_quality_assessment.evaluate_automatic_leaf import valid_count

client = TestClient(app)
ENDPOINT = "/api/v1/c3/automatic/front-image"


def encoded(size=(600, 600), format="PNG", image=None):
    stream = io.BytesIO()
    if image is None:
        image = Image.new("RGB", size, "white")
    image.save(stream, format=format)
    return stream.getvalue()


def grid_image(minor_spacing=10, y_spacing=None):
    image = np.full((600, 600, 3), 255, np.uint8)
    for axis, spacing in [("x", minor_spacing), ("y", y_spacing or minor_spacing)]:
        for n, coordinate in enumerate(range(10, 600, spacing)):
            major = n % 5 == 0
            color = (60, 60, 60) if major else (150, 150, 150)
            width = 2 if major else 1
            if axis == "x":
                cv2.line(image, (coordinate, 0), (coordinate, 599), color, width)
            else:
                cv2.line(image, (0, coordinate), (599, coordinate), color, width)
    return image


def test_upload_returns_ephemeral_manual_review_without_invented_count_or_height():
    response = client.post(ENDPOINT, content=encoded(), headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert response.json()["meta"]["persisted"] is False
    assert data["status"] == "MANUAL REVIEW"
    assert data["estimated_count"] is None  # no evidence is not a measured zero
    assert data["detected_region_count"] == 0
    assert data["confidence"] is None
    assert data["count_reliable"] is False
    assert data["leaf_heights"] == []
    assert data["quality_decision"] is None
    assert data["calibration"]["status"] == "calibration_unreliable"
    assert data["calibration"]["scale_px_per_cm"] is None
    assert data["annotated_preview"].startswith("data:image/jpeg;base64,")
    assert data["source"] == "uploaded_image_opencv_prototype"


@pytest.mark.parametrize("content_type,payload,status", [
    ("image/gif", b"GIF89a", 415),
    ("application/octet-stream", b"text", 415),
    ("image/png", b"not an image", 422),
    ("image/png", b"", 422),
    ("image/jpeg", encoded(), 415),
    ("image/png", encoded((127, 600)), 422),
    ("image/png", encoded((10001, 128)), 422),
    ("image/png", encoded()[:50], 422),
])
def test_upload_rejects_invalid_images(content_type, payload, status):
    response = client.post(ENDPOINT, content=payload, headers={"Content-Type": content_type})
    assert response.status_code == status
    assert response.json()["data"] is None
    assert response.json()["error"]["message"]


def test_byte_limits_check_declared_and_streamed_sizes(monkeypatch):
    monkeypatch.setattr(automatic_routes, "MAX_UPLOAD_BYTES", 64)
    declared = client.post(ENDPOINT, content=b"small", headers={"Content-Type": "image/png", "Content-Length": "65"})
    assert declared.status_code == 413
    streamed = client.post(ENDPOINT, content=iter([b"a" * 40, b"b" * 40]), headers={"Content-Type": "image/png"})
    assert streamed.status_code == 413


def test_pixel_limit_checked_before_decode(monkeypatch):
    import services.c3_quality_assessment.automatic_leaf as automatic_leaf
    monkeypatch.setattr(automatic_leaf, "MAX_PIXELS", 200_000)
    with pytest.raises(InvalidImage, match="megapixels"):
        decode_image(encoded(), "image/png")


def test_animated_png_rejected():
    stream = io.BytesIO()
    first = Image.new("RGB", (200, 200), "white")
    first.save(stream, format="PNG", save_all=True, append_images=[Image.new("RGB", (200, 200), "green")], duration=100)
    with pytest.raises(InvalidImage, match="multi-frame"):
        decode_image(stream.getvalue(), "image/png")


def test_exif_orientation_and_downsampling_have_explicit_coordinates():
    image = Image.new("RGB", (1600, 800), "white")
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    decoded, native = decode_image(stream.getvalue(), "image/jpeg")
    assert native == (800, 1600)
    assert decoded.shape[:2] == (1400, 700)


def test_separate_large_small_and_partial_regions_are_not_double_counted():
    image = np.full((600, 600, 3), 255, np.uint8)
    cv2.ellipse(image, (150, 200), (40, 90), 0, 0, 360, (30, 140, 40), -1)
    cv2.ellipse(image, (380, 120), (8, 18), 0, 0, 360, (30, 140, 40), -1)
    cv2.ellipse(image, (0, 400), (30, 55), 0, 0, 360, (30, 140, 40), -1)
    regions, labels, _ = leaf_regions(image, vegetation_mask(image))
    assert len(regions) == 3
    assert len({row["label"] for row in regions}) == 3
    assert any("image_border_possible_partial_leaf" in row["ambiguity_flags"] for row in regions)
    assert min(row["area_px"] for row in regions) < 600
    assert sum(row["area_px"] for row in regions) <= int((vegetation_mask(image) > 0).sum())
    prediction = analyze_image(image, (600, 600))
    assert prediction["estimated_count"] == 3
    assert prediction["status"] == "MANUAL REVIEW"  # synthetic success does not validate real-world accuracy
    assert all(row["endpoints_px"] is None and row["confidence"] is None for row in prediction["leaf_regions"])


def test_grid_requires_major_minor_identity_and_confirmed_paper():
    image = grid_image()
    mask = np.zeros((600, 600), np.uint8)
    candidate = detect_grid(image, mask, True)
    assert candidate["status"] == "grid_scale_candidate"
    assert candidate["observed_major_spacing_px"] == pytest.approx(50)
    assert candidate["observed_minor_spacing_px"] == pytest.approx(10)
    assert candidate["scale_px_per_cm"] == pytest.approx(50)
    assert len(candidate["evidence_tiles"]) >= 2
    result = detect_grid(image, mask, False)
    assert result["status"] == "calibration_unreliable"
    assert result["scale_px_per_cm"] is None
    result = analyze_image(image, (600, 600), grid_profile_confirmed=True)
    assert result["leaf_heights"] == []  # scale alone does not provide leaf endpoints


def test_regular_lines_without_hierarchy_do_not_get_mislabeled_as_centimeters():
    image = np.full((600, 600, 3), 255, np.uint8)
    for i in range(10, 600, 20):
        cv2.line(image, (i, 0), (i, 599), (90, 90, 90), 1)
        cv2.line(image, (0, i), (599, i), (90, 90, 90), 1)
    result = detect_grid(image, np.zeros((600, 600), np.uint8), True)
    assert result["status"] == "calibration_unreliable"
    assert result["scale_px_per_cm"] is None
    assert any("identity" in reason for reason in result["reasons"])


def test_missing_grid_and_inconsistent_axis_spacing_withhold_scale():
    blank = np.full((600, 600, 3), 255, np.uint8)
    mask = np.zeros((600, 600), np.uint8)
    assert detect_grid(blank, mask, True)["scale_px_per_cm"] is None
    inconsistent = grid_image(10, 15)
    result = detect_grid(inconsistent, mask, True)
    assert result["status"] == "calibration_unreliable"
    assert result["scale_px_per_cm"] is None


def test_rotated_hierarchical_grid_is_deskewed():
    image = grid_image()
    matrix = cv2.getRotationMatrix2D((300, 300), 7, 1)
    rotated = cv2.warpAffine(image, matrix, (600, 600), borderValue=(255, 255, 255))
    result = detect_grid(rotated, np.zeros((600, 600), np.uint8), True)
    assert abs(result["rotation_degrees"]) == pytest.approx(7, abs=1)
    if result["reliable"]:
        assert result["scale_px_per_cm"] == pytest.approx(50, abs=2)
    else:
        assert result["scale_px_per_cm"] is None
        assert result["reasons"]


def test_invalid_calibration_confirmation_rejected():
    response = client.post(ENDPOINT + "?grid_profile_confirmed=invalid", content=encoded(), headers={"Content-Type": "image/png"})
    assert response.status_code == 422


def test_upload_uses_confirmed_major_minor_profile_without_claiming_leaf_heights():
    payload = encoded(image=Image.fromarray(cv2.cvtColor(grid_image(), cv2.COLOR_BGR2RGB)))
    response = client.post(ENDPOINT + "?grid_profile_confirmed=true", content=payload,
                           headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["calibration"]["scale_px_per_cm"] == pytest.approx(50)
    assert data["calibration"]["major_grid_spacing_cm"] == 1
    assert data["calibration"]["minor_grid_spacing_cm"] == 0.2
    assert data["leaf_heights"] == []
    assert data["status"] == "MANUAL REVIEW"


def test_connected_vegetation_flags_possible_overlap_or_duplicate_fragments():
    image = np.full((600, 600, 3), 255, np.uint8)
    for x in (250, 320):
        cv2.ellipse(image, (x, 300), (45, 100), 0, 0, 360, (30, 140, 40), -1)
    result = analyze_image(image, (600, 600))
    assert result["status"] == "MANUAL REVIEW"
    assert any("connected_region_possible_overlap_or_duplicate_split" in row["ambiguity_flags"]
               for row in result["leaf_regions"])


def test_capacity_limit_does_not_start_more_inferences():
    assert automatic_routes._capacity.acquire(False)
    assert automatic_routes._capacity.acquire(False)
    try:
        response = client.post(ENDPOINT, content=encoded(), headers={"Content-Type": "image/png"})
        assert response.status_code == 429
    finally:
        automatic_routes._capacity.release()
        automatic_routes._capacity.release()


def test_fragmentation_limit_abstains_instead_of_reporting_a_truncated_count():
    image = np.full((600, 600, 3), 255, np.uint8)
    for x in range(15, 600, 20):
        for y in range(15, 600, 20):
            cv2.circle(image, (x, y), 3, (30, 140, 40), -1)
    result = analyze_image(image, (600, 600))
    assert result["estimated_count"] is None
    assert result["status"] == "MANUAL REVIEW"
    assert any("256-seed" in reason for reason in result["reasons"])


@pytest.mark.parametrize("value,expected", [(8, True), (8.0, True), (0, True), (None, False), (float("nan"), False), (float("inf"), False), (-1, False), (2.5, False)])
def test_evaluation_accepts_only_valid_ground_truth_counts(value, expected):
    assert valid_count(value) is expected


@pytest.fixture
def new_plant_upload(monkeypatch):
    import base64
    import pandas as pd
    # Deliberately no Samples, annotations or measurements: new plants have none.
    sheets = {"Plant_Types": pd.DataFrame([{"Plant_ID": f"P{i:02d}"} for i in range(1, 9)]),
              "Requirement_Profiles": pd.DataFrame([
                  {"Plant_ID": f"P{i:02d}", "Requirement_ID": f"R{i}", "Buyer_ID": "B1"} for i in range(1, 9)])}
    monkeypatch.setitem(app.dependency_overrides, automatic_routes.get_sheets, lambda: sheets)
    monkeypatch.setitem(app.dependency_overrides, automatic_routes.get_ai_config,
                        lambda: {"root_checkpoint": "root.pth", "leaf_checkpoint": "leaf.pth", "device": "cpu"})
    image = {"content_type": "image/png", "data": base64.b64encode(encoded()).decode()}
    return {"plant_id": "P02", "requirement_id": "R2", "images": {"Front": image.copy(), "Root": image.copy()}}


@pytest.mark.parametrize("plant", range(1, 9))
def test_new_plant_models_use_uploaded_views_and_always_require_review(new_plant_upload, monkeypatch, plant):
    from pathlib import Path
    calls = []

    def predict(path, checkpoint, task, variant, device):
        assert path.is_file()
        assert path.stem == ("Root" if task == "root" else "Front")
        assert checkpoint == f"{task}.pth" and variant == "B" and device == "cpu"
        calls.append(path)
        return {"status": "success", "predicted_label": "Acceptable" if task == "root" else "Both Sides",
                "checkpoint_path": "private/path", "confidence": .9}

    monkeypatch.setattr(automatic_routes, "predict_feature", predict)
    new_plant_upload.update(plant_id=f"P{plant:02d}", requirement_id=f"R{plant}")
    new_plant_upload["images"]["Stem"] = new_plant_upload["images"]["Front"].copy()
    response = client.post("/api/v1/c3/inspections/upload", json=new_plant_upload)
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    assert result["Buyer_Rule_Result"] == "REVIEW"
    assert "Sample_ID" not in result and "expert_annotation" not in response.text
    assert "private/path" not in response.text and "confidence" not in response.text
    assert response.json()["meta"]["persisted"] is False
    assert all(not path.exists() and not path.parent.exists() for path in calls)
    assert len(calls) == (0 if plant == 8 else 1 if plant == 1 else 2)
    assert {row["Check"] for row in result["Checks"]} == (
        {"Height", "Leaf count", "Overall visual quality"} if plant in {1, 8} else {"Height", "Overall visual quality"})
    assert all(row["Presentation_Status"] == "PENDING" for row in result["Checks"])
    assert result["AI_Observations"]["root_B"]["status"] == ("unsupported" if plant == 8 else "success")
    assert ("leaf_B" in result["AI_Observations"]) == (2 <= plant <= 7)


@pytest.mark.parametrize("problem, expected", [
    ("missing", 422), ("empty", 422), ("type", 415), ("corrupt", 422),
    ("encoding", 422), ("mismatch", 415), ("buyer", 422), ("unknown", 422),
    ("oversized", 413), ("stem", 422),
])
def test_new_plant_rejects_invalid_uploads_before_inference(new_plant_upload, monkeypatch, problem, expected):
    import base64
    payload = new_plant_upload
    root = payload["images"]["Root"]
    if problem == "missing": del payload["images"]["Front"]
    elif problem == "empty": root["data"] = ""
    elif problem == "type": root["content_type"] = "image/svg+xml"
    elif problem == "corrupt": root["data"] = base64.b64encode(b"corrupt").decode()
    elif problem == "encoding": root["data"] = "???"
    elif problem == "mismatch": root["content_type"] = "image/jpeg"
    elif problem == "buyer": payload["requirement_id"] = "R1"
    elif problem == "unknown": payload["plant_id"] = "P99"
    elif problem == "oversized": monkeypatch.setattr(automatic_routes, "MAX_INSPECTION_BYTES", 10)
    elif problem == "stem": payload["images"]["Stem"] = {"content_type": "image/png", "data": ""}
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid upload reached model inference")
    monkeypatch.setattr(automatic_routes, "predict_feature", unexpected)
    response = client.post("/api/v1/c3/inspections/upload", json=payload)
    assert response.status_code == expected, response.text


@pytest.mark.parametrize("failure", ["missing", "incompatible", "exception"])
def test_new_plant_unavailable_model_never_invents_prediction(new_plant_upload, monkeypatch, failure):
    if failure == "missing":
        monkeypatch.setitem(app.dependency_overrides, automatic_routes.get_ai_config, lambda: {})
    def fail(*args, **kwargs):
        if failure == "exception": raise RuntimeError("secret checkpoint path")
        return {"status": "error", "error": {"message": "secret checkpoint path"}}
    monkeypatch.setattr(automatic_routes, "predict_feature", fail)
    response = client.post("/api/v1/c3/inspections/upload", json=new_plant_upload)
    assert response.status_code == 200
    assert response.json()["data"]["Buyer_Rule_Result"] == "REVIEW"
    assert all(row["status"] == "unavailable" and "prediction" not in row
               for row in response.json()["data"]["AI_Observations"].values())
    assert "secret" not in response.text
