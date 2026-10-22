"""Conservative, untrained P01 OpenCV baseline; never an accepted quality decision."""
from __future__ import annotations

import base64
import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 60_000_000
MAX_DIMENSION = 10_000
MIN_DIMENSION = 128
PROCESS_MAX = 1400
ALGORITHM_VERSION = "p01-opencv-baseline-v1"


class InvalidImage(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def decode_image(payload: bytes, content_type: str):
    """Check headers before decoding; reject animation and mismatched MIME types."""
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InvalidImage("Image exceeds the 25 MiB limit.", 413)
    expected = {"image/jpeg": "JPEG", "image/png": "PNG"}.get(content_type)
    if expected is None:
        raise InvalidImage("Only JPEG and PNG images are accepted.", 415)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as original:
                if original.format != expected:
                    raise InvalidImage("Image bytes do not match the declared JPEG/PNG format.", 415)
                width, height = original.size
                if (min(width, height) < MIN_DIMENSION or max(width, height) > MAX_DIMENSION
                        or width * height > MAX_PIXELS):
                    raise InvalidImage("Image must be at least 128 px per side, at most 10000 px per side and 60 megapixels.")
                if getattr(original, "n_frames", 1) != 1:
                    raise InvalidImage("Animated or multi-frame images are not accepted.", 415)
                original.verify()
            with Image.open(io.BytesIO(payload)) as original:
                oriented = ImageOps.exif_transpose(original)
                native_size = oriented.size
                oriented.thumbnail((PROCESS_MAX, PROCESS_MAX))
                rgb = np.asarray(oriented.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), native_size
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Image is corrupt, truncated, or unsafe to decode.") from exc


def vegetation_mask(image, *, saturation=35, excess_green=12):
    """Broad green/yellow selection; no image-coordinate assumptions about orientation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.int16))
    green = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 100)
    green &= (hsv[:, :, 1] >= saturation) & (hsv[:, :, 2] >= 30)
    green &= (2 * g - r - b >= excess_green)
    mask = green.astype(np.uint8) * 255
    # A single small closing fills variegation holes without discarding small leaves.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def leaf_regions(image, mask):
    """Distance-peak markers partition the mask; each pixel belongs to at most one region."""
    original_shape = mask.shape
    # Watershed discards boundary pixels. Padding keeps partial-leaf markers alive.
    mask = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if not np.any(distance):
        return [], np.zeros(original_shape, np.int32), 0
    local_max = cv2.dilate(distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    peaks = ((distance >= local_max - 1e-5) & (distance >= 2.0)).astype(np.uint8)
    number, seed_labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)
    if number - 1 > 256:
        # Bound work on noisy/adversarial inputs; no truncated count is reported.
        return [], np.zeros(original_shape, np.int32), number - 1
    markers = np.zeros(mask.shape, np.int32)
    markers[mask == 0] = 1
    _, components = cv2.connectedComponents(mask)
    component_seeds = {}
    for label in range(1, number):
        x, y = np.rint(centroids[label]).astype(int)
        # Centroid can fall outside a non-convex peak plateau; choose an actual peak pixel.
        ys, xs = np.where(seed_labels == label)
        i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        markers[ys[i], xs[i]] = label + 1
        component = int(components[ys[i], xs[i]])
        component_seeds[component] = component_seeds.get(component, 0) + 1
    topography = 255 - cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    labels = cv2.watershed(cv2.cvtColor(topography, cv2.COLOR_GRAY2BGR), markers)
    labels[mask == 0] = 1
    labels = labels[2:-2, 2:-2]
    mask = mask[2:-2, 2:-2]
    components = components[2:-2, 2:-2]
    regions = []
    min_area = max(24, round(mask.size * 0.000015))
    for label in range(2, number + 1):
        region_mask = (labels == label).astype(np.uint8)
        area = int(region_mask.sum())
        if area < min_area:
            continue
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < 12:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(cv2.contourArea(contour) / hull_area) if hull_area else 0.0
        flags = []
        contour_x, contour_y = contour[0, 0]
        if component_seeds.get(int(components[contour_y, contour_x]), 0) > 1:
            flags.append("connected_region_possible_overlap_or_duplicate_split")
        if solidity < 0.7:
            flags.append("irregular_region_possible_overlap_or_fragmentation")
        if x <= 3 or y <= 3 or x + w >= mask.shape[1] - 3 or y + h >= mask.shape[0] - 3:
            flags.append("image_border_possible_partial_leaf")
        if area < 150:
            flags.append("small_region_possible_partial_leaf_or_stem")
        regions.append({"region_id": len(regions) + 1, "label": label,
                        "bbox_px": [x, y, w, h], "area_px": area,
                        "solidity": round(solidity, 4), "ambiguity_flags": flags,
                        "confidence": None, "endpoints_px": None})
    return regions, labels, number - 1


def _grid_axis(tile, contrast, axis):
    """Require a nested five-minor-interval major-line hierarchy, not just periodicity."""
    height, width = tile.shape
    kernel = (max(20, width // 3), 1) if axis == "y" else (1, max(20, height // 3))
    lines = cv2.morphologyEx(tile, cv2.MORPH_OPEN, np.ones((kernel[1], kernel[0]), np.uint8))
    reduction = 1 if axis == "y" else 0
    projection = (lines > 0).mean(axis=reduction)
    contrast_profile = contrast.mean(axis=reduction)
    indices = np.flatnonzero(projection >= 0.45)
    if not len(indices):
        return None
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    centers = np.array([group.mean() for group in groups])
    gaps = np.diff(centers)
    if len(gaps) < 10:
        return None
    minor_spacing = float(np.median(gaps))
    if minor_spacing < 4 or np.max(np.abs(gaps - minor_spacing)) / minor_spacing > 0.2:
        return None
    widths = np.array([len(group) for group in groups])
    strengths = np.array([contrast_profile[group].max() for group in groups])
    major = (widths >= max(2, np.median(widths) * 1.6)) | (strengths >= max(12, np.median(strengths) * 1.5))
    major_centers = centers[major]
    major_gaps = np.diff(major_centers)
    if len(major_gaps) < 2:
        return None
    expected_major = minor_spacing * 5
    if np.max(np.abs(major_gaps - expected_major)) / expected_major > 0.12:
        return None
    major_spacing = float(np.median(major_gaps))
    return {"minor_spacing_px": minor_spacing, "major_spacing_px": major_spacing,
            "minor_line_count": len(centers), "major_line_count": len(major_centers),
            "major_to_minor_ratio": major_spacing / minor_spacing,
            "minor_gap_cv": float(np.std(gaps) / np.mean(gaps)),
            "identity_evidence": "stronger/thicker lines separated by five minor intervals"}


def _deskew_grid(image, mask):
    """Estimate a dominant line orientation modulo 90 degrees; grid tests still gate acceptance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40,
                            minLineLength=max(40, min(gray.shape) // 8), maxLineGap=10)
    rotation = 0.0
    line_count = 0
    if lines is not None:
        segments = lines.reshape(-1, 4).astype(float)
        differences = segments[:, 2:] - segments[:, :2]
        lengths = np.linalg.norm(differences, axis=1)
        angles = (np.degrees(np.arctan2(differences[:, 1], differences[:, 0])) + 45) % 90 - 45
        histogram, bins = np.histogram(angles, bins=np.arange(-45, 46, 1), weights=lengths)
        best = int(np.argmax(histogram))
        center = (bins[best] + bins[best + 1]) / 2
        selected = np.abs(angles - center) <= 1.5
        line_count = int(selected.sum())
        if line_count >= 6 and lengths[selected].sum() >= lengths.sum() * 0.4:
            rotation = float(np.median(angles[selected]))
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1)
    deskewed = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))
    deskewed_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    return deskewed, deskewed_mask, rotation, matrix, line_count


def detect_grid(image, mask, grid_profile_confirmed=False):
    """P01 paper: 1 cm major / 0.2 cm minor; never assign cm to unidentified lines."""
    deskewed, mask, rotation, matrix, line_count = _deskew_grid(image, mask)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    dark[cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0] = 0
    height, width = gray.shape
    tiles = []
    for row in range(3):
        for col in range(3):
            y0, y1 = row * height // 3, (row + 1) * height // 3
            x0, x1 = col * width // 3, (col + 1) * width // 3
            tile = dark[y0:y1, x0:x1]
            local_contrast = contrast[y0:y1, x0:x1]
            if min(tile.shape) < 60:
                continue
            x = _grid_axis(tile, local_contrast, "x")
            y = _grid_axis(tile, local_contrast, "y")
            if x and y:
                tiles.append({"tile_bbox_px": [x0, y0, x1 - x0, y1 - y0], "x": x, "y": y})
    reasons = []
    major_spacing = None
    minor_spacing = None
    if len(tiles) < 2:
        reasons.append("Major/minor line identity is ambiguous: fewer than two tiles show the five-to-one grid hierarchy in both axes.")
    else:
        majors = np.array([t[axis]["major_spacing_px"] for t in tiles for axis in ("x", "y")])
        minors = np.array([t[axis]["minor_spacing_px"] for t in tiles for axis in ("x", "y")])
        major_spacing = float(np.median(majors))
        minor_spacing = float(np.median(minors))
        if np.max(np.abs(majors - major_spacing)) / major_spacing > 0.12:
            reasons.append("Grid spacing varies across axes or tiles: perspective or inconsistent line detection; single scale withheld.")
    if not grid_profile_confirmed:
        reasons.append("This image's 1 cm major / 0.2 cm minor P01 graph-paper profile has not been confirmed.")
    reliable = not reasons
    return {"status": "grid_scale_candidate" if reliable else "calibration_unreliable",
            "reliable": reliable, "confidence": None,
            "major_grid_spacing_cm": 1.0, "minor_grid_spacing_cm": 0.2,
            "grid_profile_confirmed": grid_profile_confirmed,
            "observed_major_spacing_px": major_spacing, "observed_minor_spacing_px": minor_spacing,
            "scale_px_per_cm": major_spacing / 1.0 if reliable else None,
            "rotation_degrees": rotation, "orientation_supporting_segments": line_count,
            "deskew_matrix": matrix.tolist(), "evidence_coordinate_space": "deskewed processing image",
            "evidence_tiles": tiles, "reasons": reasons,
            "limitation": "Background-plane scale only; plant coplanarity and individual leaf endpoints are not verified. Perspective variation is rejected, not rectified."}


def analyze_image(image, native_size, *, grid_profile_confirmed=False):
    mask = vegetation_mask(image)
    regions, labels, seed_count = leaf_regions(image, mask)
    strict_regions, _, _ = leaf_regions(image, vegetation_mask(image, saturation=55, excess_green=24))
    component_count = cv2.connectedComponents(mask)[0] - 1
    reasons = ["Unvalidated color/shape baseline cannot certify that every visible leaf is counted once.",
               "Individual leaf emergence and tip endpoints are not reliably detected; no leaf heights reported."]
    if not regions:
        reasons.append("No separable green leaf regions detected; absence of detections does not establish zero leaves.")
    if seed_count > 256:
        reasons.append("Excessive segmentation fragmentation exceeded the 256-seed processing limit; count withheld.")
    if len(strict_regions) != len(regions):
        reasons.append("Count changes with color thresholds; variegation, stems or background interference may affect segmentation.")
    if any("connected_region_possible_overlap_or_duplicate_split" in r["ambiguity_flags"] for r in regions):
        reasons.append("Connected vegetation required splitting: overlapping leaves, stems or duplicate fragments are possible.")
    if any(r["ambiguity_flags"] for r in regions):
        reasons.append("Small, irregular or border-touching regions require inspection for partial leaves and occlusion.")
    calibration = detect_grid(image, mask, grid_profile_confirmed)
    if not calibration["reliable"]:
        reasons.append("calibration_unreliable: " + " ".join(calibration["reasons"]))
    preview = image.copy()
    for region in regions:
        color = (0, 165, 255)
        contours, _ = cv2.findContours((labels == region["label"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, color, 2)
        x, y, w, h = region["bbox_px"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 1)
        cv2.putText(preview, str(region["region_id"]), (x, max(18, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(preview, str(region["region_id"]), (x, max(18, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        del region["label"]
    cv2.putText(preview, "CANDIDATE REGIONS - MANUAL REVIEW", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode annotated preview.")
    return {"algorithm_version": ALGORITHM_VERSION, "source": "uploaded_image_opencv_prototype",
            "plant_id": "P01", "status": "MANUAL REVIEW", "count_reliable": False,
            "estimated_count": len(regions) if regions else None, "detected_region_count": len(regions),
            "confidence": None, "confidence_note": "No calibrated statistical confidence is available for this untrained baseline.",
            "reasons": reasons, "leaf_regions": regions, "leaf_heights": [],
            "calibration": calibration, "quality_decision": None,
            "native_dimensions": list(native_size), "processing_dimensions": [image.shape[1], image.shape[0]],
            "coordinate_space": "EXIF-oriented processing image; multiply x/y by native/processing dimensions for native coordinates",
            "diagnostics": {"foreground_fraction": round(float((mask > 0).mean()), 6),
                            "vegetation_components": component_count, "distance_peak_seeds": seed_count,
                            "strict_threshold_region_count": len(strict_regions)},
            "annotated_preview": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}
