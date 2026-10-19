"""Focused tests for the isolated P01 image measurement prototype."""

import math
import unittest
from services.c3_quality_assessment.p01_height_measurement import (
    BuyerRange,
    classify_measurements,
    display_to_original,
    fit_dimensions,
    load_buyer_range,
    pixels_per_centimetre,
    vertical_height_cm,
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


class TestVerticalHeight(unittest.TestCase):
    def test_uses_vertical_delta_and_not_diagonal_distance(self):
        self.assertEqual(vertical_height_cm(300, 150, 30), 5)

    def test_invalid_coordinates_or_scale(self):
        for args in ((0, 10, 0), (0, math.nan, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                vertical_height_cm(*args)


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
        self.assertEqual((r001.minimum_cm, r001.maximum_cm), (20, 25))
        self.assertEqual((r002.minimum_cm, r002.maximum_cm), (30, 35))

    def test_range_and_thresholds(self):
        buyer = BuyerRange("R001", "B001", 20, 25)
        args = dict(calibration_verified=True, scale_reliable=True,
                    reference_point=(100, 200))
        self.assertEqual(classify_measurements([20, 20, 26, 26], buyer, **args)["Status"], "PASS")
        self.assertEqual(classify_measurements([26, 27, 28], buyer, **args)["Status"], "FAIL")
        self.assertEqual(classify_measurements([20, 20, 26, 26, 26], buyer, **args)["Status"], "MANUAL REVIEW")

    def test_missing_or_unreliable_inputs_return_manual_review(self):
        buyer = BuyerRange("R001", "B001", 20, 25)
        base = dict(heights_cm=[21], buyer_range=buyer,
                    calibration_verified=True, scale_reliable=True,
                    reference_point=(1, 1))
        self.assertEqual(classify_measurements([], buyer, **{k: v for k, v in base.items() if k not in {"heights_cm", "buyer_range"}})["Status"], "MANUAL REVIEW")
        self.assertEqual(classify_measurements([21, 21], buyer,
                                              calibration_verified=True,
                                              scale_reliable=True,
                                              reference_point=(1, 1))["Status"], "MANUAL REVIEW")
        for change in ({"calibration_verified": False}, {"scale_reliable": False}, {"reference_point": None}):
            args = {**base, **change}
            self.assertEqual(classify_measurements(args.pop("heights_cm"), args.pop("buyer_range"), **args)["Status"], "MANUAL REVIEW")
        result = classify_measurements([None, -1, math.nan], buyer,
                                      calibration_verified=True, scale_reliable=True,
                                      reference_point=(1, 1))
        self.assertEqual(result["Status"], "MANUAL REVIEW")
        self.assertEqual(result["Valid_Leaves"], 0)


if __name__ == "__main__":
    unittest.main()
