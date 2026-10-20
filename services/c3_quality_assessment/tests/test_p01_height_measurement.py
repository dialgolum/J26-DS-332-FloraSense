"""Focused tests for the isolated P01 image measurement prototype."""

import math
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
import pandas as pd
from services.c3_quality_assessment.p01_height_measurement import (
    BuyerRange,
    add_leaf_measurement_point,
    classify_measurements,
    display_to_original,
    fit_dimensions,
    load_buyer_range,
    pixels_per_centimetre,
    leaf_distance_cm,
    undo_leaf_measurement,
    load_measurement_session,
    reevaluate_session,
    save_measurement_session,
)


class TestCalibration(unittest.TestCase):
    def test_pixels_per_centimetre(self):
        self.assertEqual(pixels_per_centimetre(60, 2, verified=True), 30)

    def test_unverified_or_invalid_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            pixels_per_centimetre(60, 2, verified=False)
        for pixels, spacing in ((0, 1), (10, 0), (math.inf, 1)):
            with self.subTest(pixels=pixels, spacing=spacing):
                with self.assertRaises(ValueError):
                    pixels_per_centimetre(pixels, spacing, verified=True)


class TestLeafDistance(unittest.TestCase):
    def test_uses_emergence_to_tip_distance_for_horizontal_and_vertical_leaves(self):
        self.assertEqual(leaf_distance_cm((10, 10), (10, 160), 30), 5)
        self.assertAlmostEqual(leaf_distance_cm((10, 10), (130, 170), 40), 5)

    def test_distance_is_unchanged_by_portrait_landscape_rotation(self):
        portrait_distance = leaf_distance_cm((10, 10), (130, 170), 40)
        landscape_distance = leaf_distance_cm((10, 10), (-150, 130), 40)
        self.assertAlmostEqual(portrait_distance, landscape_distance)

    def test_invalid_coordinates_or_scale(self):
        for args in (((0, 0), (10, 10), 0), ((0, 0), (math.nan, 10), 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                leaf_distance_cm(*args)


class TestLeafClickState(unittest.TestCase):
    def test_clicks_pair_emergence_and_tip_and_reset_for_each_leaf(self):
        measurements = []
        pending = None
        pending = add_leaf_measurement_point((10, 20), measurements, pending)
        self.assertEqual(pending, (10, 20))
        self.assertEqual(measurements, [])

        pending = add_leaf_measurement_point((40, 80), measurements, pending)
        self.assertIsNone(pending)
        self.assertEqual(measurements, [((10, 20), (40, 80))])

        pending = add_leaf_measurement_point((15, 25), measurements, pending)
        pending = add_leaf_measurement_point((60, 90), measurements, pending)
        self.assertIsNone(pending)
        self.assertEqual(measurements[-1], ((15, 25), (60, 90)))

    def test_undo_clears_pending_click_before_removing_completed_pair(self):
        measurements = [((1, 2), (3, 4))]
        self.assertIsNone(undo_leaf_measurement(measurements, (5, 6)))
        self.assertEqual(len(measurements), 1)
        self.assertIsNone(undo_leaf_measurement(measurements, None))
        self.assertEqual(measurements, [])


class TestImageDisplayGeometry(unittest.TestCase):
    def test_fit_preserves_aspect_and_fits_portrait_image_without_crop(self):
        fitted = fit_dimensions(5862, 8134, 1400, 900)
        self.assertLessEqual(fitted[0], 1400)
        self.assertLessEqual(fitted[1], 900)
        self.assertAlmostEqual(fitted[0] / fitted[1], 5862 / 8134, places=2)
        self.assertLess(fitted[0], 5862)

    def test_coordinate_mapping_returns_native_image_pixels(self):
        native = display_to_original(324, 450, (5862, 8134), (648, 900))
        self.assertAlmostEqual(native[0], 2931)
        self.assertAlmostEqual(native[1], 4067)


class TestBuyerRangeAndDecision(unittest.TestCase):
    def test_reads_ranges_from_workbook(self):
        r001 = load_buyer_range("R001")
        r002 = load_buyer_range("R002")
        profiles = pd.read_excel(
            Path(__file__).resolve().parents[3]
            / "data/quality_assessment/annotations/quality_dataset.xlsx",
            sheet_name="Requirement_Profiles",
        )
        for result in (r001, r002):
            row = profiles[profiles["Requirement_ID"] == result.requirement_id].iloc[0]
            self.assertEqual((result.minimum_cm, result.maximum_cm),
                             (row["Min_Height_cm"], row["Max_Height_cm"]))

    def test_range_and_thresholds(self):
        buyer = BuyerRange("R001", "B001", 20, 25)
        args = dict(calibration_verified=True, scale_reliable=True)
        self.assertEqual(classify_measurements([20, 20, 26, 26], buyer, **args)["Status"], "PASS")
        self.assertEqual(classify_measurements([26, 27, 28], buyer, **args)["Status"], "FAIL")
        self.assertEqual(classify_measurements([20, 20, 26, 26, 26], buyer, **args)["Status"], "MANUAL REVIEW")

    def test_missing_or_unreliable_inputs_return_manual_review(self):
        buyer = BuyerRange("R001", "B001", 20, 25)
        base = dict(heights_cm=[21], buyer_range=buyer,
                    calibration_verified=True, scale_reliable=True)
        self.assertEqual(classify_measurements([], buyer, **{k: v for k, v in base.items() if k not in {"heights_cm", "buyer_range"}})["Status"], "MANUAL REVIEW")
        self.assertEqual(classify_measurements([21, 21], buyer,
                                              calibration_verified=True,
                                              scale_reliable=True)["Status"], "MANUAL REVIEW")
        for change in ({"calibration_verified": False}, {"scale_reliable": False}):
            args = {**base, **change}
            self.assertEqual(classify_measurements(args.pop("heights_cm"), args.pop("buyer_range"), **args)["Status"], "MANUAL REVIEW")
        result = classify_measurements([None, -1, math.nan], buyer,
                                      calibration_verified=True, scale_reliable=True)
        self.assertEqual(result["Status"], "MANUAL REVIEW")
        self.assertEqual(result["Valid_Leaves"], 0)


class TestSavedMeasurementSessions(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f"_session_test_{uuid4().hex}"
        self.root.mkdir()
        self.image = self.root / "P01_S004_front.jpg"
        self.image.touch()
        self.workbook = self.root / "profiles.xlsx"
        with pd.ExcelWriter(self.workbook) as writer:
            pd.DataFrame([
                {"Requirement_ID": "R001", "Plant_ID": "P01", "Buyer_ID": "B001",
                 "Min_Height_cm": 20, "Max_Height_cm": 25},
                {"Requirement_ID": "R002", "Plant_ID": "P01", "Buyer_ID": "B002",
                 "Min_Height_cm": 30, "Max_Height_cm": 35},
            ]).to_excel(writer, sheet_name="Requirement_Profiles", index=False)
            pd.DataFrame([{
                "Image_ID": "IMG0004", "Sample_ID": "S004",
                "Relative_Path": str(self.image),
            }]).to_excel(writer, sheet_name="Images", index=False)
        self.results_dir = self.root / "results"
        self.measurements = [((0, 0), (320, 0)) for _ in range(3)]

    def tearDown(self):
        shutil.rmtree(self.root)

    def save(self):
        return save_measurement_session(
            self.image, "R001", [(10, 10), (20, 10)], 10, self.measurements,
            calibration_verified=True, scale_reliable=True,
            results_dir=self.results_dir, workbook=self.workbook,
        )

    def test_save_load_and_unique_session_files(self):
        first = self.save()
        second = self.save()
        self.assertNotEqual(first, second)
        session = load_measurement_session(first)
        self.assertEqual(session["sample_id"], "S004")
        self.assertEqual(session["image_id"], "IMG0004")
        self.assertEqual(session["calibration_points"], [[10, 10], [20, 10]])
        self.assertEqual(session["scale_px_per_cm"], 10)
        self.assertEqual(session["leaves"][0]["height_cm"], 32)
        self.assertTrue(session["calibration_verified"])
        self.assertTrue(session["scale_reliable"])
        self.assertIn("timestamp", session)

    def test_saved_measurements_can_be_reevaluated_for_another_requirement(self):
        saved = self.save()
        result = reevaluate_session(saved, "R002", workbook=self.workbook)
        self.assertEqual(result["Status"], "PASS")
        self.assertEqual(result["Buyer_ID"], "B002")
        self.assertEqual(result["Range_cm"], [30, 35])
        self.assertEqual(result["Sample_ID"], "S004")

    def test_malformed_and_incomplete_sessions_are_rejected(self):
        bad_json = self.root / "bad.json"
        bad_json.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_measurement_session(bad_json)

        missing = self.root / "missing.json"
        missing.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_measurement_session(missing)

        incomplete = self.save()
        data = json.loads(incomplete.read_text(encoding="utf-8"))
        del data["leaves"][0]["tip_px"]
        incomplete.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Each leaf"):
            load_measurement_session(incomplete)


if __name__ == "__main__":
    unittest.main()
