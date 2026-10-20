"""Focused tests for the read-only C3 API and saved-session reevaluation."""

from fastapi.testclient import TestClient
import pandas as pd

from services.c3_quality_assessment.api.main import app
from services.c3_quality_assessment.api.routes import get_results_dir, get_sheets
from services.c3_quality_assessment.data_loader import ROOT, load_dataset
from services.c3_quality_assessment.p01_height_measurement import RESULTS_DIR


client = TestClient(app)
app.dependency_overrides[get_sheets] = load_dataset
app.dependency_overrides[get_results_dir] = lambda: RESULTS_DIR


def test_reference_endpoints_return_real_workbook_records():
    plants = client.get("/api/v1/c3/plants")
    assert plants.status_code == 200
    assert plants.json()["data"][0]["Plant_ID"] == "P01"

    requirements = client.get("/api/v1/c3/requirements", params={"plant_id": "P01"})
    assert requirements.status_code == 200
    assert {row["Requirement_ID"] for row in requirements.json()["data"]} == {"R001", "R002"}

    samples = client.get("/api/v1/c3/samples", params={"plant_id": "P01"})
    assert samples.status_code == 200
    assert "S005" in {row["Sample_ID"] for row in samples.json()["data"]}

    preflight = client.options(
        "/api/v1/c3/plants",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_sample_images_return_safe_preview_url_and_image():
    images = client.get("/api/v1/c3/samples/S005/images")
    assert images.status_code == 200
    image = next(row for row in images.json()["data"] if row["View_Type"] == "Front")
    assert image["preview_url"] == f"/api/v1/c3/images/{image['Image_ID']}/preview"

    preview = client.get(image["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/")

    unknown = client.get("/api/v1/c3/images/../../secret/preview")
    assert unknown.status_code in {400, 404, 422}


def test_preview_rejects_a_workbook_path_outside_approved_image_root():
    sheets = load_dataset()
    malicious_row = sheets["Images"].iloc[0].copy()
    malicious_row["Image_ID"] = "OUTSIDE01"
    malicious_row["Relative_Path"] = str(ROOT / "README.md")
    sheets["Images"] = pd.concat(
        [sheets["Images"], malicious_row.to_frame().T], ignore_index=True
    )
    original_dependency = app.dependency_overrides[get_sheets]
    app.dependency_overrides[get_sheets] = lambda: sheets
    try:
        response = client.get("/api/v1/c3/images/OUTSIDE01/preview")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "HTTP_403"
    finally:
        app.dependency_overrides[get_sheets] = original_dependency


def test_real_s005_session_can_be_reevaluated_against_both_buyers():
    sessions = client.get("/api/v1/c3/samples/S005/height-sessions")
    assert sessions.status_code == 200
    saved = next(row for row in sessions.json()["data"] if row["session_id"].startswith("S005_"))
    assert saved["leaf_count"] == 9

    results = {}
    for requirement_id in ("R001", "R002"):
        response = client.post(
            f"/api/v1/c3/height-sessions/{saved['session_id']}/reevaluate",
            json={"requirement_id": requirement_id},
        )
        assert response.status_code == 200, response.text
        results[requirement_id] = response.json()["data"]
        assert results[requirement_id]["assessment_type"] == "Height Assessment"
        assert results[requirement_id]["sample_id"] == "S005"
        assert results[requirement_id]["valid_leaves"] == 9
        assert len(results[requirement_id]["leaves"]) == 9
        assert results[requirement_id]["reasons"] == []

    assert results["R001"]["range_cm"] == [20.0, 25.0]
    assert results["R001"]["leaves_in_range"] == 5
    assert results["R001"]["percentage_in_range"] == 55.55555555555556
    assert results["R001"]["status"] == "PASS"
    assert results["R002"]["range_cm"] == [25.0, 30.0]
    assert results["R002"]["leaves_in_range"] == 2
    assert results["R002"]["percentage_in_range"] == 22.22222222222222
    assert results["R002"]["status"] == "FAIL"


def test_unknown_ids_and_invalid_session_references_return_json_errors():
    unknown_plant = client.get("/api/v1/c3/requirements", params={"plant_id": "P999"})
    assert unknown_plant.status_code == 404
    assert unknown_plant.json()["data"] is None
    assert unknown_plant.json()["error"]["code"] == "HTTP_404"

    unknown_sample = client.get("/api/v1/c3/samples/S999/images")
    assert unknown_sample.status_code == 404

    invalid_session = client.post(
        "/api/v1/c3/height-sessions/not-a-session/reevaluate",
        json={"requirement_id": "R001"},
    )
    assert invalid_session.status_code == 400

    unknown_session = client.post(
        "/api/v1/c3/height-sessions/missing.json/reevaluate",
        json={"requirement_id": "R001"},
    )
    assert unknown_session.status_code == 404

    session_id = next(row["session_id"] for row in client.get(
        "/api/v1/c3/samples/S005/height-sessions"
    ).json()["data"] if row["sample_id"] == "S005")
    unknown_requirement = client.post(
        f"/api/v1/c3/height-sessions/{session_id}/reevaluate",
        json={"requirement_id": "R999"},
    )
    assert unknown_requirement.status_code == 404
