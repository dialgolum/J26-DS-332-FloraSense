"""
Pack nursery PHOTOS + code into one zip for Google Colab via Google Drive.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from services.c2_disease_detection import config
from services.c2_disease_detection.training.scripts.nursery_data import load_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    ap.add_argument("--out", type=Path, default=Path("c2_bundle.zip"))
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--with-code", action="store_true", default=True)
    args = ap.parse_args()

    df, problems = load_metadata(args.metadata)
    for p in problems:
        print("WARNING:", p)
    if df.empty:
        raise SystemExit("No usable rows in metadata.csv.")

    n_code = n_img = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        if args.with_code:
            z.writestr("services/__init__.py", "")
            for p in config.SERVICE_DIR.rglob("*"):
                if p.is_file() and not any(x in p.parts for x in ("__pycache__", ".pytest_cache", "reports")) and p.suffix != ".pyc":
                    z.write(p, p.relative_to(config.REPO_ROOT).as_posix())
                    n_code += 1
        lines = ["filepath,label,species,plant_id,date"]
        used = set()
        clean = lambda v: str(v).replace(",", " ")
        for _, r in df.iterrows():
            src = Path(r["path"])
            arc = f"data/c2_disease/raw/{r['label']}/{src.stem}.jpg"
            if arc in used:
                arc = f"data/c2_disease/raw/{r['label']}/{src.stem}_{n_img}.jpg"
            used.add(arc)
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=92)
            z.writestr(arc, buf.getvalue())
            lines.append(",".join([arc.replace("data/c2_disease/", ""), r["label"], clean(r["species"]),
                                   clean(r["plant_id"]), clean(r["date"])]))
            n_img += 1
        z.writestr("data/c2_disease/metadata.csv", "\n".join(lines) + "\n")
    print(f"Bundle written: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {n_img} photos, {n_code} code files)")


if __name__ == "__main__":
    main()