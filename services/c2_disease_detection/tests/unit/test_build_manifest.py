import csv

import pytest

from services.c2_disease_detection.training.scripts.build_manifest import build_manifest


def test_build_manifest_uses_image_directories_and_skips_unsupported_files(tmp_path):
    raw_dir = tmp_path / "images"
    class_dir = raw_dir / "Apple___Apple_scab"
    class_dir.mkdir(parents=True)
    (class_dir / "leaf.JPG").write_bytes(b"image")
    (class_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    output_csv = tmp_path / "output" / "manifest.csv"

    image_count = build_manifest(raw_dir, output_csv)

    assert image_count == 1
    with output_csv.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    assert rows == [
        {
            "filename": "leaf.JPG",
            "relative_path": str(class_dir / "leaf.JPG").replace("\\", "/"),
            "plant_species": "Ornamental Foliage",
            "disease_class": "apple___apple_scab",
            "symptom_description": "Field nursery foliar condition.",
            "source": "Mike Flora Nursery",
        }
    ]


def test_build_manifest_raises_when_image_directory_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "missing", tmp_path / "manifest.csv")
