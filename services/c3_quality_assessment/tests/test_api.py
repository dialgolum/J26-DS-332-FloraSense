"""Focused tests for the read-only C3 API and saved-session reevaluation."""

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from services.c3_quality_assessment.api.main import app
from services.c3_quality_assessment.api.routes import get_results_dir, get_sheets
from services.c3_quality_assessment.data_loader import ROOT, load_dataset as _load_dataset
from services.c3_quality_assessment.p01_height_measurement import RESULTS_DIR


def load_dataset():
    """Historical read-only/session tests exercise the original P01 scope."""
    sheets = _load_dataset()
    for name, frame in sheets.items():
        if "Plant_ID" in frame:
            sheets[name] = frame[frame.Plant_ID.eq("P01")].copy()
    heights = sheets["Height_Assessments"]
    sheets["Height_Assessments"] = heights[heights.Requirement_ID.isin(["R001", "R002"])].copy()
    return sheets


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


def test_all_p01_images_and_annotations_match_workbook():
    sheets = load_dataset()
    for sample_id in sheets["Samples"]["Sample_ID"]:
        images = client.get(f"/api/v1/c3/samples/{sample_id}/images").json()["data"]
        expected = sheets["Images"][sheets["Images"].Sample_ID == sample_id]
        assert {row["Image_ID"] for row in images} == set(expected.Image_ID)
        for row in images:
            assert client.get(row["preview_url"]).status_code == 200
        for requirement_id in ("R001", "R002"):
            heights = client.get(f"/api/v1/c3/samples/{sample_id}/height-assessments",
                                 params={"requirement_id": requirement_id})
            assert heights.status_code == 200
            assert heights.json()["meta"]["source"] == "recorded_manual_annotation"
            expected_heights = sheets["Height_Assessments"]
            expected_heights = expected_heights[(expected_heights.Sample_ID == sample_id)
                                                & (expected_heights.Requirement_ID == requirement_id)]
            from services.c3_quality_assessment.api.routes import _records
            assert heights.json()["data"] == _records(expected_heights)
            quality = client.get(f"/api/v1/c3/samples/{sample_id}/quality-annotations",
                                 params={"requirement_id": requirement_id})
            assert quality.status_code == 200
            assert quality.json()["meta"]["scope"] == "sample"
            expected_quality = sheets["Quality_Annotations"]
            expected_quality = expected_quality[expected_quality.Sample_ID == sample_id]
            from services.c3_quality_assessment.api.routes import _records
            assert quality.json()["data"] == _records(expected_quality)


def test_multiple_images_and_missing_views():
    for sample_id, views in [("S004", {"Front", "Root"}),
                             ("S005", {"Front", "Root", "Focus"}),
                             ("S014", {"Front", "Root"})]:
        rows = client.get(f"/api/v1/c3/samples/{sample_id}/images").json()["data"]
        assert {row["View_Type"] for row in rows} == views
    sheets = load_dataset()
    assert "Side" in set(sheets["Images"].View_Type)


def test_missing_sessions_preserve_available_annotations(tmp_path):
    original = app.dependency_overrides[get_results_dir]
    app.dependency_overrides[get_results_dir] = lambda: tmp_path
    try:
        for sample_id in ("S004", "S014"):
            assert client.get(f"/api/v1/c3/samples/{sample_id}/height-sessions").json()["data"] == []
            height = client.get(f"/api/v1/c3/samples/{sample_id}/height-assessments",
                                params={"requirement_id": "R001"}).json()["data"]
            assert height[0]["Height_Result"] == "PASS"
    finally:
        app.dependency_overrides[get_results_dir] = original


def test_missing_annotations_and_sessions_are_empty_not_invented(tmp_path):
    sheets = load_dataset()
    for name in ("Height_Assessments", "Quality_Annotations", "Images"):
        sheets[name] = sheets[name][sheets[name].Sample_ID != "S004"]
    original_sheets = app.dependency_overrides[get_sheets]
    original_results = app.dependency_overrides[get_results_dir]
    app.dependency_overrides[get_sheets] = lambda: sheets
    app.dependency_overrides[get_results_dir] = lambda: tmp_path
    try:
        for endpoint in ("height-assessments", "quality-annotations", "height-sessions", "images"):
            response = client.get(f"/api/v1/c3/samples/S004/{endpoint}",
                                  params={"requirement_id": "R001"})
            assert response.status_code == 200
            assert response.json()["data"] == []
    finally:
        app.dependency_overrides[get_sheets] = original_sheets
        app.dependency_overrides[get_results_dir] = original_results


def test_annotation_validation_and_sample_wide_quality():
    for endpoint in ("height-assessments", "quality-annotations"):
        assert client.get(f"/api/v1/c3/samples/S999/{endpoint}",
                          params={"requirement_id": "R001"}).status_code == 404
        assert client.get(f"/api/v1/c3/samples/S004/{endpoint}",
                          params={"requirement_id": "R999"}).status_code == 404
        assert client.get(f"/api/v1/c3/samples/S004/{endpoint}").status_code == 422
    r1 = client.get("/api/v1/c3/samples/S005/quality-annotations",
                    params={"requirement_id": "R001"}).json()["data"]
    r2 = client.get("/api/v1/c3/samples/S005/quality-annotations",
                    params={"requirement_id": "R002"}).json()["data"]
    assert r1 == r2
    assert r1[0]["Total_Leaf_Count"] == 11
    assert r1[0]["Leaf_Distribution"] is None


@pytest.fixture
def assessment_data():
    from services.c3_quality_assessment.tests.test_rule_engine import make_sheets
    from services.c3_quality_assessment.api.routes import get_ai_config
    sheets = make_sheets()
    for frame in sheets.values():
        if "Plant_ID" in frame:
            frame["Plant_ID"] = "P02"
    sheets["Quality_Annotations"]["Leaf_Distribution"] = "Both Sides"
    sheets["Images"] = pd.DataFrame([
        {"Plant_ID": "P02", "Sample_ID": "S001", "View_Type": view, "Image_ID": view,
         "Relative_Path": f"data/quality_assessment/original/individual/P02/P02_S001_{view.lower()}.jpg"}
        for view in ("Root", "Front")])
    old = dict(app.dependency_overrides)
    app.dependency_overrides[get_sheets] = lambda: sheets
    app.dependency_overrides[get_ai_config] = lambda: {
        "root_checkpoint": "explicit-root.pth", "leaf_checkpoint": "explicit-leaf.pth",
        "leaf_a_checkpoint": "explicit-leaf-a.pth", "device": "cpu"}
    try:
        yield sheets
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old)


def assessment_request(**changes):
    return {"plant_id": "P02", "sample_id": "S001", "requirement_id": "R001", **changes}


def test_assessment_valid_disabled_and_composite_identity(assessment_data, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("AI must remain disabled")
    monkeypatch.setattr("services.c3_quality_assessment.run_assessment.predict_feature", unexpected)
    other = assessment_data["Samples"].copy()
    other["Plant_ID"] = "P01"
    assessment_data["Samples"] = pd.concat([other, assessment_data["Samples"]])
    response = client.post("/api/v1/c3/assessments", json=assessment_request())
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["sample_key"] == "P02:S001"
    assert data["Buyer_ID"] == "B001"
    assert data["Buyer_Rule_Result"] == data["Overall_Result"] == "PASS"
    assert data["AI_Observations"]["root_B"]["status"] == "disabled"
    assert data["Missing_Measurements"] == []
    assert all(row["Presentation_Status"] == "PASS" for row in data["Checks"])
    assert data["Checks"][2]["Reason"].startswith("4/8 leaves")


@pytest.mark.parametrize("changes,status", [
    ({"sample_id": "S999"}, 404), ({"requirement_id": "R999"}, 404),
    ({"plant_id": "P99"}, 404), ({"plant_id": "P01"}, 422),
    ({"sample_id": "../secret"}, 422), ({"compare_leaf_a": True}, 422),
    ({"root_checkpoint": "C:/secret/model.pth"}, 422)])
def test_assessment_invalid_requests(assessment_data, changes, status):
    if changes == {"plant_id": "P01"}:
        other = assessment_data["Plant_Types"].copy()
        other["Plant_ID"] = "P01"
        assessment_data["Plant_Types"] = pd.concat([other, assessment_data["Plant_Types"]])
    response = client.post("/api/v1/c3/assessments", json=assessment_request(**changes))
    assert response.status_code == status
    assert response.json()["data"] is None
    assert response.json()["error"]["code"]


@pytest.mark.parametrize("mode,presentation", [("absent", "PENDING"), ("null", "PENDING"),
                                                ("invalid", "REVIEW"), ("borderline", "REVIEW"),
                                                ("duplicate", "REVIEW"), ("fail", "FAIL")])
def test_assessment_height_presentation_preserves_rule(assessment_data, mode, presentation):
    heights = assessment_data["Height_Assessments"]
    if mode == "absent":
        assessment_data["Height_Assessments"] = heights.iloc[:0]
    elif mode == "duplicate":
        assessment_data["Height_Assessments"] = pd.concat([heights, heights])
    else:
        heights["Leaves_In_Range"] = {"null": None, "invalid": -1, "borderline": 3, "fail": 1}[mode]
    response = client.post("/api/v1/c3/assessments", json=assessment_request())
    assert response.status_code == 200
    data = response.json()["data"]
    height = next(row for row in data["Checks"] if row["Check"] == "Height")
    assert height["Presentation_Status"] == presentation
    assert height["Status"] == ("FAIL" if mode == "fail" else "MANUAL REVIEW")
    assert data["Buyer_Rule_Result"] == height["Status"]
    assert height["Missing_Measurement"] == (presentation == "PENDING")


def test_assessment_ai_enabled_comparison_and_paths(assessment_data, monkeypatch):
    assessment_data["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
    calls = []
    def predict(image, checkpoint, task, **kwargs):
        calls.append((image, checkpoint, kwargs["variant"]))
        return {"status": "success", "predicted_label": "Not Acceptable" if task == "root" else "One Side",
                "confidence": .9, "architecture": "resnet18", "model_variant": kwargs["variant"],
                "dataset_version": "c3_features_v2_20261022", "checkpoint_id": "sha256:test",
                "checkpoint_path": "C:/private/model.pth", "class_probabilities": {}}
    monkeypatch.setattr("services.c3_quality_assessment.run_assessment.predict_feature", predict)
    response = client.post("/api/v1/c3/assessments", json=assessment_request(ai_enabled=True, compare_leaf_a=True))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["Buyer_Rule_Result"] == "PASS"  # Expert Both Sides passes despite AI disagreement
    assert len(data["Disagreements"]) == 3
    assert data["AI_Observations"]["leaf_B"]["observation_type"] == "branch_origin_distribution"
    assert [call[1] for call in calls] == ["explicit-root.pth", "explicit-leaf.pth", "explicit-leaf-a.pth"]
    assert "_root.jpg" in calls[0][0] and "_front.jpg" in calls[1][0]
    assert "image_path" not in response.text and "checkpoint_path" not in response.text
    assert "private" not in response.text and str(ROOT) not in response.text
    assert data["Manual_Review"]["required"]
    assert not data["Manual_Review"]["affects_buyer_rule_result"]


@pytest.mark.parametrize("failure", ["exception", "structured", "missing_image", "missing_checkpoint", "unsupported"])
def test_assessment_ai_failures_are_advisory(assessment_data, monkeypatch, failure):
    from services.c3_quality_assessment.api.routes import get_ai_config
    def predict(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("C:/secret/model.pth /private/secrets dangerous diagnostic")
        return {"status": "error", "error": {"code": "bad_model", "message": "C:/secret/model.pth private diagnostic"}}
    monkeypatch.setattr("services.c3_quality_assessment.run_assessment.predict_feature", predict)
    plant_id = "P02"
    if failure == "missing_image":
        assessment_data["Images"] = assessment_data["Images"].iloc[:0]
    if failure == "missing_checkpoint":
        app.dependency_overrides[get_ai_config] = lambda: {"root_checkpoint": None, "leaf_checkpoint": None}
    if failure == "unsupported":
        plant_id = "P08"
        for frame in assessment_data.values():
            if "Plant_ID" in frame:
                frame["Plant_ID"] = plant_id
    response = client.post("/api/v1/c3/assessments", json=assessment_request(ai_enabled=True, plant_id=plant_id))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["Buyer_Rule_Result"] == "PASS"
    assert data["AI_Observations"]["root_B"]["status"] in {"error", "unavailable", "unsupported"}
    assert data["Manual_Review"]["required"]
    assert "secret" not in response.text and "diagnostic" not in response.text


def test_assessment_workbook_and_evaluation_errors_are_safe(assessment_data, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("C:/secret/workbook.xlsx")
    monkeypatch.setattr("services.c3_quality_assessment.api.routes.run_assessment", broken)
    response = client.post("/api/v1/c3/assessments", json=assessment_request())
    assert response.status_code == 503
    assert "secret" not in response.text
    # Use the real dependency to exercise its existing safe load failure boundary.
    from services.c3_quality_assessment.api.routes import get_sheets
    app.dependency_overrides.pop(get_sheets)
    monkeypatch.setattr("services.c3_quality_assessment.api.routes.load_dataset", broken)
    response = client.get("/api/v1/c3/plants")
    assert response.status_code == 503
    assert "secret" not in response.text


def test_assessment_duplicate_composite_is_rejected(assessment_data):
    assessment_data["Samples"] = pd.concat([assessment_data["Samples"], assessment_data["Samples"]])
    response = client.post("/api/v1/c3/assessments", json=assessment_request())
    assert response.status_code == 409


def test_assessment_ai_cannot_promote_fail(assessment_data, monkeypatch):
    assessment_data["Quality_Annotations"]["Root_Distribution"] = "Not Acceptable"
    monkeypatch.setattr("services.c3_quality_assessment.run_assessment.predict_feature",
                        lambda *args, **kwargs: {"status": "success", "predicted_label": "Acceptable"})
    response = client.post("/api/v1/c3/assessments", json=assessment_request(ai_enabled=True))
    assert response.status_code == 200
    assert response.json()["data"]["Buyer_Rule_Result"] == "FAIL"
    assert response.json()["data"]["Disagreements"][0]["expert_label"] == "Not Acceptable"


def test_assessment_current_workbook_and_missing_measurements(assessment_data):
    app.dependency_overrides[get_sheets] = _load_dataset
    response = client.post("/api/v1/c3/assessments", json=assessment_request(requirement_id="R003"))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["sample_key"] == "P02:S001"
    assert data["Buyer_Rule_Result"] == "MANUAL REVIEW"
    height = next(row for row in data["Checks"] if row["Check"] == "Height")
    assert height["Presentation_Status"] == "PENDING"
    assert data["Missing_Measurements"] == ["Height"]


def test_frontend_composite_image_and_annotation_delivery(assessment_data):
    app.dependency_overrides[get_sheets] = _load_dataset
    for plant_id, requirement_id in (("P01", "R001"), ("P02", "R003")):
        images = client.get("/api/v1/c3/samples/S001/images", params={"plant_id": plant_id})
        assert images.status_code == 200
        rows = images.json()["data"]
        assert rows and all(row["Plant_ID"] == plant_id for row in rows)
        assert "Relative_Path" not in images.text
        for row in rows:
            assert client.get(row["preview_url"]).status_code == 200
        quality = client.get("/api/v1/c3/samples/S001/quality-annotations",
                             params={"requirement_id": requirement_id})
        assert quality.status_code == 200
        assert all(row["Plant_ID"] == plant_id for row in quality.json()["data"])
        height = client.get("/api/v1/c3/samples/S001/height-assessments",
                            params={"requirement_id": requirement_id})
        assert height.status_code == 200
        assert client.get("/api/v1/c3/samples/S001/height-sessions",
                          params={"plant_id": plant_id}).status_code == 200
    # P01 sessions must never be offered for a different plant with the same local ID.
    assert client.get("/api/v1/c3/samples/S001/height-sessions",
                      params={"plant_id": "P02"}).json()["data"] == []
    assert client.get("/api/v1/c3/samples/S001/images", params={"plant_id": "P99"}).status_code == 404


def test_frontend_missing_image_returns_safe_error(assessment_data):
    assessment_data["Images"].loc[0, "Relative_Path"] = "data/quality_assessment/original/absent.jpg"
    images = client.get("/api/v1/c3/samples/S001/images", params={"plant_id": "P02"})
    assert images.status_code == 200
    url = images.json()["data"][0]["preview_url"]
    missing = client.get(url)
    assert missing.status_code == 404
    assert missing.json()["error"]["message"]
    assert str(ROOT) not in missing.text


@pytest.mark.parametrize("label,expected,presentation", [
    ("Both Sides", "PASS", "PASS"), ("One Side", "FAIL", "FAIL"),
    (None, "MANUAL REVIEW", "REVIEW"), ("", "MANUAL REVIEW", "REVIEW"),
    ("Unknown", "MANUAL REVIEW", "REVIEW")])
def test_leaf_rule_api_statuses(assessment_data, label, expected, presentation):
    assessment_data["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
    assessment_data["Quality_Annotations"]["Leaf_Distribution"] = label
    response = client.post("/api/v1/c3/assessments", json=assessment_request())
    assert response.status_code == 200
    result = response.json()["data"]
    leaf = next(check for check in result["Checks"] if check["Check"] == "Leaf distribution")
    assert leaf["Status"] == expected
    assert leaf["Presentation_Status"] == presentation
    assert result["Buyer_Rule_Result"] == expected
