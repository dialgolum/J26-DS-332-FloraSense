"""Interactive, image-based P01 leaf-height measurement prototype.

Run with ``python -m services.c3_quality_assessment.p01_height_measurement``.
The interactive window requires OpenCV with GUI support (opencv-python).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "data" / "quality_assessment" / "annotations" / "quality_dataset.xlsx"
DEFAULT_DISPLAY_SIZE = (1280, 720)


def fit_dimensions(
    original_width: int,
    original_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Return aspect-preserving display dimensions without cropping or upscaling."""
    if min(original_width, original_height, max_width, max_height) <= 0:
        raise ValueError("Image and display dimensions must be positive.")
    scale = min(1.0, max_width / original_width, max_height / original_height)
    return max(1, round(original_width * scale)), max(1, round(original_height * scale))


def display_to_original(
    x: float,
    y: float,
    original_size: tuple[int, int],
    display_size: tuple[int, int],
) -> tuple[float, float]:
    """Map a point from the displayed fitted image to native-image pixels."""
    original_width, original_height = original_size
    display_width, display_height = display_size
    if min(original_width, original_height, display_width, display_height) <= 0:
        raise ValueError("Image and display dimensions must be positive.")
    return (
        x * original_width / display_width,
        y * original_height / display_height,
    )


def resolve_input_image(image_path: Path) -> Path:
    """Resolve an image argument from the working directory or repository root."""
    path = image_path.expanduser()
    if not path.is_absolute():
        working_directory_path = path.resolve()
        path = working_directory_path if working_directory_path.is_file() else ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    return path


def enhance_display_only(image, cv2):
    """Moderately strengthen grid contrast and edge visibility for display only."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(10, 10))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    softened = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.2, softened, -0.2, 0)


@dataclass(frozen=True)
class BuyerRange:
    requirement_id: str
    buyer_id: str
    minimum_cm: float
    maximum_cm: float


def load_buyer_range(requirement_id: str, workbook: Path = WORKBOOK) -> BuyerRange:
    """Read a P01 buyer height range from the existing workbook."""
    profiles = pd.read_excel(workbook, sheet_name="Requirement_Profiles")
    profiles.columns = profiles.columns.str.strip()
    rows = profiles[
        (profiles["Requirement_ID"].astype(str).str.strip() == requirement_id)
        & (profiles["Plant_ID"].astype(str).str.strip() == "P01")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one P01 profile for {requirement_id}; found {len(rows)}")
    row = rows.iloc[0]
    low, high = row.get("Min_Height_cm"), row.get("Max_Height_cm")
    if pd.isna(low) or pd.isna(high) or float(low) > float(high):
        raise ValueError(f"Invalid height range in profile {requirement_id}")
    return BuyerRange(requirement_id, str(row["Buyer_ID"]), float(low), float(high))


def pixels_per_centimetre(
    calibration_pixel_distance: float,
    grid_spacing_cm: float,
    *,
    verified: bool,
) -> float:
    """Return a one-axis scale after the operator verifies the grid spacing."""
    if not verified:
        raise ValueError("Graph-paper spacing must be verified with a ruler or reliable capture information.")
    if not math.isfinite(calibration_pixel_distance) or calibration_pixel_distance <= 0:
        raise ValueError("Calibration pixel distance must be a positive finite value.")
    if not math.isfinite(grid_spacing_cm) or grid_spacing_cm <= 0:
        raise ValueError("Grid spacing must be a positive finite number of centimetres.")
    return calibration_pixel_distance / grid_spacing_cm


def vertical_height_cm(
    reference_y: float, tip_y: float, scale_px_per_cm: float
) -> float:
    """Calculate vertical image-coordinate difference, ignoring horizontal offset."""
    if not all(math.isfinite(value) for value in (reference_y, tip_y, scale_px_per_cm)):
        raise ValueError("Coordinates and scale must be finite numbers.")
    if scale_px_per_cm <= 0:
        raise ValueError("Scale must be positive.")
    return abs(reference_y - tip_y) / scale_px_per_cm


def classify_measurements(
    heights_cm: Sequence[float | None],
    buyer_range: BuyerRange,
    *,
    calibration_verified: bool,
    scale_reliable: bool,
    reference_point: tuple[float, float] | None,
    minimum_measurements: int = 3,
) -> dict:
    """Summarize valid heights and apply P01's configured buyer range."""
    reasons = []
    if not calibration_verified:
        reasons.append("Graph-paper calibration is unverified.")
    if not scale_reliable:
        reasons.append("Tilt or perspective may make the single image scale unreliable.")
    if reference_point is None:
        reasons.append("The stem emergence reference point is missing.")

    valid = []
    invalid_count = 0
    for value in heights_cm:
        if value is None:
            invalid_count += 1
        else:
            try:
                height = float(value)
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if not math.isfinite(height) or height <= 0:
                invalid_count += 1
                continue
            valid.append(height)

    if not valid:
        reasons.append("No valid leaf heights were measured.")
    elif len(valid) < minimum_measurements:
        reasons.append(
            f"Only {len(valid)} valid leaf height(s); at least "
            f"{minimum_measurements} are required for this prototype."
        )
    if invalid_count:
        reasons.append(f"{invalid_count} measurement(s) are missing or invalid.")

    within = sum(buyer_range.minimum_cm <= h <= buyer_range.maximum_cm for h in valid)
    percentage = (100.0 * within / len(valid)) if valid else None
    if reasons:
        status = "MANUAL REVIEW"
    elif percentage >= 50:
        status = "PASS"
    elif percentage < 30:
        status = "FAIL"
    else:
        status = "MANUAL REVIEW"
    return {
        "Status": status,
        "Valid_Leaves": len(valid),
        "Leaves_In_Range": within,
        "Percentage_In_Range": percentage,
        "Heights_cm": valid,
        "Reasons": reasons,
    }


def run_interactive(image_path: Path, requirement_id: str, grid_spacing_cm: float,
                    calibration_verified: bool, scale_reliable: bool,
                    minimum_measurements: int = 3,
                    max_display_size: tuple[int, int] = DEFAULT_DISPLAY_SIZE,
                    display_enhancement: bool = True) -> dict:
    """Use OpenCV clicks to calibrate, set a shared reference, and mark leaf tips."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the interactive UI. Install opencv-python.") from exc

    resolved_image_path = resolve_input_image(Path(image_path))
    image = cv2.imread(str(resolved_image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Could not load image in colour (BGR): {resolved_image_path}")
    original_height, original_width = image.shape[:2]
    print(f"Resolved image: {resolved_image_path}")
    print(f"Original image dimensions: {original_width} x {original_height} pixels (colour BGR)")

    shown_image = enhance_display_only(image, cv2) if display_enhancement else image.copy()
    display_width, display_height = fit_dimensions(
        original_width, original_height, *max_display_size
    )
    if (display_width, display_height) != (original_width, original_height):
        shown_image = cv2.resize(
            shown_image, (display_width, display_height), interpolation=cv2.INTER_AREA
        )
    print(f"Displayed image dimensions: {display_width} x {display_height} pixels (full image, aspect-preserved)")
    profile = load_buyer_range(requirement_id)
    points: list[tuple[int, int]] = []
    calibration_points: list[tuple[int, int]] = []
    reference: tuple[int, int] | None = None
    mode = "calibration"
    result: dict = {}
    window = "P01 leaf height | c calibration, r reference, t tips, u undo, right-click delete, q finish"

    def to_display(point: tuple[float, float]) -> tuple[int, int]:
        x = round(point[0] * display_width / original_width)
        y = round(point[1] * display_height / original_height)
        return x, y

    def redraw() -> None:
        canvas = shown_image.copy()
        calibration_markers = result.get("calibration_points", calibration_points)
        for point in calibration_markers:
            visible = to_display(point)
            cv2.drawMarker(canvas, visible, (0, 0, 0), cv2.MARKER_DIAMOND, 18, 4)
            cv2.drawMarker(canvas, visible, (255, 0, 255), cv2.MARKER_DIAMOND, 14, 2)
        for point in points:
            visible = to_display(point)
            cv2.circle(canvas, visible, 7, (0, 0, 0), 3)
            cv2.circle(canvas, visible, 5, (0, 255, 255), -1)
        if reference is not None:
            visible_reference = to_display(reference)
            cv2.drawMarker(canvas, visible_reference, (255, 255, 255), cv2.MARKER_CROSS, 24, 4)
            cv2.drawMarker(canvas, visible_reference, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            for index, point in enumerate(points, 1):
                height = vertical_height_cm(reference[1], point[1], result.get("scale", 1))
                label_point = to_display(point)
                cv2.putText(canvas, f"{index}: {height:.1f} cm", (label_point[0] + 8, label_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(canvas, f"{index}: {height:.1f} cm", (label_point[0] + 8, label_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        title = f"Mode: {mode} | Buyer range: {profile.minimum_cm:g}-{profile.maximum_cm:g} cm"
        cv2.putText(canvas, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(canvas, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(window, canvas)

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal reference
        if event == cv2.EVENT_RBUTTONDOWN and mode == "tips" and points:
            nearest = min(range(len(points)), key=lambda index: math.dist(to_display(points[index]), (x, y)))
            if math.dist(to_display(points[nearest]), (x, y)) <= 15:
                points.pop(nearest)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        native_x, native_y = display_to_original(
            x, y, (original_width, original_height), (display_width, display_height)
        )
        native_point = (round(native_x), round(native_y))
        if mode == "calibration":
            calibration_points.append(native_point)
            if len(calibration_points) == 2:
                distance = math.dist(calibration_points[0], calibration_points[1])
                result["scale"] = pixels_per_centimetre(
                    distance, grid_spacing_cm, verified=True
                )
                result["calibration_points"] = calibration_points.copy()
                calibration_points.clear()
                result["calibrated"] = True
        elif mode == "reference":
            reference = native_point
        elif mode == "tips":
            if reference is not None:
                points.append(native_point)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_click)
    print("Select two adjacent graph-paper points spanning the configured spacing.")
    print("Keys: c calibration, r reference point, t leaf tips, u undo, right-click a tip to remove, q finish")
    print("Simple scale assumes graph paper and vertical measurement share a plane with negligible perspective.")
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("c"):
            mode = "calibration"
            points.clear()
            calibration_points.clear()
            result.pop("calibrated", None)
            result.pop("scale", None)
            result.pop("calibration_points", None)
        elif key == ord("r"):
            mode = "reference"
        elif key == ord("t"):
            mode = "tips"
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("q") or key == 27:
            break
        redraw()
    cv2.destroyAllWindows()

    scale = result.get("scale")
    heights = [vertical_height_cm(reference[1], point[1], scale) for point in points] if reference and scale else []
    assessment = classify_measurements(
        heights, profile,
        calibration_verified=calibration_verified and bool(result.get("calibrated")),
        scale_reliable=scale_reliable,
        reference_point=reference,
        minimum_measurements=minimum_measurements,
    )
    assessment.update({"Requirement_ID": profile.requirement_id, "Buyer_ID": profile.buyer_id,
                       "Range_cm": [profile.minimum_cm, profile.maximum_cm],
                       "Scale_px_per_cm": scale, "Image": str(resolved_image_path)})
    print(assessment)
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Existing P01 image path")
    parser.add_argument("--requirement", required=True, choices=("R001", "R002"))
    parser.add_argument("--grid-spacing-cm", type=float, required=True,
                        help="Physical spacing selected between calibration points")
    parser.add_argument("--confirm-calibration", action="store_true",
                        help="Confirm grid spacing was checked with a ruler/capture information")
    parser.add_argument("--confirm-scale-reliable", action="store_true",
                        help="Confirm image is sufficiently aligned and perspective is negligible")
    parser.add_argument("--minimum-leaves", type=int, default=3,
                        help="Minimum valid leaf measurements required (default: 3)")
    parser.add_argument("--max-display-width", type=int, default=DEFAULT_DISPLAY_SIZE[0])
    parser.add_argument("--max-display-height", type=int, default=DEFAULT_DISPLAY_SIZE[1])
    parser.add_argument("--no-display-enhancement", action="store_true",
                        help="Display the original colours without contrast/sharpening")
    args = parser.parse_args()
    run_interactive(args.image, args.requirement, args.grid_spacing_cm,
                    args.confirm_calibration, args.confirm_scale_reliable,
                    args.minimum_leaves,
                    (args.max_display_width, args.max_display_height),
                    not args.no_display_enhancement)


if __name__ == "__main__":
    main()
