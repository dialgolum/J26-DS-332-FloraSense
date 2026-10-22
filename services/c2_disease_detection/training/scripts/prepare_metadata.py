"""
Preprocessing step 1: clean your manifest and check every photo.
Output: metadata.csv + evaluation/reports/preprocessing_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

from services.c2_disease_detection import config
from services.c2_disease_detection.training.scripts.nursery_data import load_metadata

ImageFile.LOAD_TRUNCATED_IMAGES = True
MIN_SIDE = 224
DARK, BRIGHT = 35, 225
NEAR_DUP_BITS = 4

def _blur_score(gray: np.ndarray) -> float:
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())

def _dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def inspect_image(path: str) -> Dict:
    try:
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
            im.draft("RGB", (640, 640))
            small = ImageOps.exif_transpose(im).convert("RGB")
            small.thumbnail((256, 256))
            gray = np.asarray(small.convert("L"), dtype=np.float32)
            return dict(width=w, height=h, brightness=float(gray.mean()),
                        blur_score=_blur_score(gray), dhash=_dhash(small), ok=True)
    except Exception as exc:
        return dict(width=0, height=0, brightness=0.0, blur_score=0.0, dhash=0, ok=False, error=str(exc))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/c2_disease/dataset_manifest1.csv"))
    ap.add_argument("--out", type=Path, default=config.METADATA_PATH)
    ap.add_argument("--session-gap", type=float, default=60.0)
    ap.add_argument("--blur-percentile", type=float, default=5.0)
    args = ap.parse_args()

    df, problems = load_metadata(args.manifest, session_gap=args.session_gap)
    if df.empty:
        raise SystemExit("No usable rows in the manifest.\n" + "\n".join(problems))

    info = [inspect_image(p) for p in tqdm(df["path"], desc="checking photos")]
    meta = pd.DataFrame(info)
    df = pd.concat([df.reset_index(drop=True), meta], axis=1)
    unreadable = df[~df["ok"]]
    if len(unreadable):
        problems.append(f"{len(unreadable)} photos could not be opened and were dropped.")
        df = df[df["ok"]].reset_index(drop=True)

    blur_cut = float(np.percentile(df["blur_score"], args.blur_percentile))
    flags: List[List[str]] = [[] for _ in range(len(df))]
    for i, r in df.iterrows():
        if min(r["width"], r["height"]) < MIN_SIDE:
            flags[i].append("small")
        if r["brightness"] < DARK:
            flags[i].append("too_dark")
        if r["brightness"] > BRIGHT:
            flags[i].append("too_bright")
        if r["blur_score"] <= blur_cut:
            flags[i].append("blurry")
    near_pairs = 0
    for _, grp in df.groupby(["label", "plant_id"]):
        idx = list(grp.index)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if bin(int(df.at[idx[a], "dhash"]) ^ int(df.at[idx[b], "dhash"])).count("1") <= NEAR_DUP_BITS:
                    near_pairs += 1
                    if "near_duplicate" not in flags[idx[b]]:
                        flags[idx[b]].append("near_duplicate")
    df["flags"] = [";".join(f) for f in flags]

    def rel(p: str) -> str:
        try:
            return Path(p).resolve().relative_to(config.DATA_DIR.resolve()).as_posix()
        except ValueError:
            return p

    out = pd.DataFrame({
        "filepath": df["path"].map(rel),
        "label": df["label"],
        "species": df["species"],
        "variety": df["variety"],
        "plant_id": df["plant_id"],
        "date": df["date"],
        "width": df["width"],
        "height": df["height"],
        "brightness": df["brightness"].round(1),
        "blur_score": df["blur_score"].round(1),
        "flags": df["flags"]
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    per_class = df["label"].value_counts().reindex(list(config.CORE_CLASSES) + [config.OOD_FOLDER], fill_value=0)
    per_species = df.groupby("species")["label"].count().sort_values(ascending=False)
    per_variety = df.groupby(["species", "variety"])["label"].count()
    flagged = df[df["flags"] != ""]
    advice = []
    for c, n in per_class.items():
        target = 15 if c == "unknown" else 30
        if n < target:
            advice.append(f"'{c}' has {n} photos; aim for at least {target}.")
    report = {
        "photos_kept": int(len(df)),
        "photos_per_class": {k: int(v) for k, v in per_class.items()},
        "species_clean": {k: int(v) for k, v in per_species.items()},
        "varieties_clean": {f"{sp} - {var}": int(v) for (sp, var), v in per_variety.items()},
        "pseudo_plant_groups": int(df["plant_id"].nunique()),
        "image_size_px": {
            "min_side_median": float(np.median(np.minimum(df["width"], df["height"]))),
            "min_side_smallest": int(np.minimum(df["width"], df["height"]).min())
        },
        "flag_counts": {f: int(df["flags"].str.contains(f).sum()) for f in ("small", "too_dark", "too_bright", "blurry", "near_duplicate")},
        "near_duplicate_pairs": near_pairs,
        "problems": problems,
        "advice": advice,
        "flagged_files": flagged.assign(file=flagged["path"].map(lambda p: Path(p).name))[["file", "label", "flags"]].head(60).to_dict("records"),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nmetadata.csv written: {args.out}  ({len(out)} photos, {report['pseudo_plant_groups']} plant groups)")
    print("\nPhotos per class:\n" + per_class.to_string())
    print(f"\nClean species list ({len(per_species)}):\n" + per_species.to_string())
    print("\nFlag counts:", report["flag_counts"], "| near-duplicate pairs:", near_pairs)
    for p in problems:
        print("WARNING:", p)
    for a in advice:
        print("ADVICE:", a)

if __name__ == "__main__":
    main()