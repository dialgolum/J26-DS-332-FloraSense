import re, math, shutil
from pathlib import Path
from datetime import datetime

INCOMING = Path("data/incoming")
FINAL = Path("data/raw/images")
SECTIONS = 4          # sections per bed (reduced automatically for small beds)
DRY_RUN = False       # True = only print, False = copy files
EXTS = {".jpg", ".jpeg", ".png"}

def stamp(p):
    m = re.match(r"(\d{8})_(\d{6})", p.name)            # 20261001_101038.jpg
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(p.stat().st_mtime)    # fallback, less reliable

total = 0
for bed_dir in sorted(INCOMING.glob("*/*/*")):
    if not bed_dir.is_dir():
        continue
    species, variety, bed = bed_dir.parts[-3:]
    files = sorted((f for f in bed_dir.iterdir() if f.suffix.lower() in EXTS), key=stamp)
    if not files:
        continue
    n_sec = min(SECTIONS, max(1, len(files) // 10))     # at least ~10 photos per section
    size = math.ceil(len(files) / n_sec)
    print(f"{species}/{variety}/{bed}: {len(files)} photos -> {n_sec} sections")
    total += len(files)
    for i, f in enumerate(files):
        sec = i // size + 1
        plant_id = f"{variety}-{bed}-S{sec}"
        dest = FINAL / species / variety / bed / plant_id / f.name
        if DRY_RUN:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(f, dest)

print(f"\nTotal photos found: {total}")
print("Dry run only. Set DRY_RUN = False and run again." if DRY_RUN else "Done. Files copied.")