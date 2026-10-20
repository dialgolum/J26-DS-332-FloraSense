import csv, re, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

RAW = Path("data/raw/images")
OUT = Path("data/metadata.csv")
EXTS = {".jpg", ".jpeg", ".png"}
LIGHTING = "shade_net"      # same for all your photos
ANGLE = "oblique_45"        # same for all your photos

def date_from_name(name):
    m = re.match(r"(\d{4})(\d{2})(\d{2})_\d{6}", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else ""

rows, bad = [], []
for p in sorted(RAW.rglob("*")):
    if p.suffix.lower() not in EXTS:
        continue
    parts = p.relative_to(RAW).parts
    if len(parts) != 5:   # species/variety/bed/section/file
        bad.append(f"wrong folder depth: {p}")
        continue
    species, variety, batch_id, plant_id, _ = parts
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        bad.append(f"unreadable: {p}")
        continue
    rows.append({
        "image_path": p.as_posix(),
        "species": species.replace("_", " "),
        "variety": variety,
        "plant_id": plant_id,
        "batch_id": batch_id,
        "capture_date": date_from_name(p.name),
        "lighting": LIGHTING,
        "angle": ANGLE,
        "file_hash": hashlib.md5(p.read_bytes()).hexdigest(),
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} images written to {OUT}")
if bad:
    print("\nPROBLEMS:"); [print(" -", b) for b in bad]

dups = [h for h, c in Counter(r["file_hash"] for r in rows).items() if c > 1]
print(f"Exact duplicate files: {len(dups)}")
print("Missing capture_date:", sum(1 for r in rows if not r["capture_date"]))

plants, imgs, beds = defaultdict(set), Counter(), defaultdict(set)
for r in rows:
    k = (r["species"], r["variety"])
    plants[k].add(r["plant_id"]); beds[k].add(r["batch_id"]); imgs[k] += 1
print(f"\n{'class':<40}{'images':>7}{'beds':>6}{'sections':>9}")
for k in sorted(imgs):
    print(f"{k[0]+' / '+k[1]:<40}{imgs[k]:>7}{len(beds[k]):>6}{len(plants[k]):>9}")