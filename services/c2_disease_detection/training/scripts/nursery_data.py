"""
Read metadata, validate it, and create plant-level train/val/test splits.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from services.c2_disease_detection import config

PATH_COLS = ("filepath", "relative_path", "path", "image_path", "file_path", "filename", "file", "image", "image_name")
LABEL_COLS = ("label", "disease_class", "class", "category", "condition")
SPECIES_COLS = ("species", "plant_species", "plant")
PLANT_COLS = ("plant_id", "plantid", "plant_tag", "tag", "batch_id", "batch")
DATE_COLS = ("date", "date_taken", "captured")

LABEL_ALIASES = {
    "healthy": "healthy", "normal": "healthy",
    "bacterial": "bacterial", "bacteria": "bacterial",
    "fungi": "fungi", "fungal": "fungi", "fungus": "fungi",
    "abiotic": "abiotic",
    "unknown": "unknown", "unhealthy": "unknown",
}
TS_PATTERN = re.compile(r"(20\d{2})(\d{2})(\d{2})[_\-]?(\d{2})(\d{2})(\d{2})")

def clean_species(raw: str) -> tuple[str, str]:
    s = (raw or "").strip().replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"(?i)cholrophytum", "Chlorophytum", s)
    if not s:
        return "", ""
    base, _, cultivar = s.partition("'")
    words = base.split()
    base = " ".join(w.capitalize() if i == 0 else w.lower() for i, w in enumerate(words))
    cultivar = " ".join(cultivar.replace("'", " ").split()).title()
    name = f"{base} '{cultivar}'" if cultivar else base
    return name, f"{base.lower()}|{cultivar.lower()}"

def parse_timestamp(name: str):
    m = TS_PATTERN.search(name)
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(*[int(x) for x in m.groups()])
    except ValueError:
        return pd.NaT

def derive_session_ids(df: pd.DataFrame, gap_seconds: float = 60.0) -> pd.Series:
    ids = pd.Series(index=df.index, dtype=object)
    sub = df[df["ts"].notna()].sort_values("ts")
    if sub.empty:
        return ids
    new = (sub["species_key"] != sub["species_key"].shift()) | (sub["ts"].diff().dt.total_seconds() > gap_seconds)
    ids.loc[sub.index] = ["auto-%03d" % n for n in new.cumsum()]
    return ids

def _first_col(df: pd.DataFrame, names) -> Optional[str]:
    return next((n for n in names if n in df.columns), None)

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _resolve(raw: str, label: str, data_dir: Path) -> Optional[Path]:
    p = Path(raw.strip().replace("\\", "/"))
    name = p.name
    candidates = [
        p,
        config.REPO_ROOT / p,
        data_dir / p,
        data_dir.parent.parent / p,
        data_dir / "raw" / p,
        data_dir / "raw" / label / name,
        config.RAW_DIR / label / name,
        Path("data/c2_disease/raw") / label / name,
    ]
    return next((c for c in candidates if c.is_file()), None)

def load_metadata(csv_path: Path, data_dir: Path = config.DATA_DIR,
                  session_gap: float = 60.0) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in df.columns]
    pcol, lcol = _first_col(df, PATH_COLS), _first_col(df, LABEL_COLS)
    if not pcol or not lcol:
        raise SystemExit(f"metadata needs a path column {PATH_COLS[:3]}... and a label column {LABEL_COLS}. Found: {list(df.columns)}")
    scol, icol, dcol = _first_col(df, SPECIES_COLS), _first_col(df, PLANT_COLS), _first_col(df, DATE_COLS)
    if "relative_path" in df.columns and df["relative_path"].str.strip().ne("").any():
        pcol = "relative_path"

    rows, bad_label, missing, blank = [], [], [], 0
    for _, r in df.iterrows():
        raw_path, raw_label = r[pcol].strip(), r[lcol].strip()
        if not raw_path and not raw_label:
            blank += 1
            continue
        label = LABEL_ALIASES.get(raw_label.lower())
        if label is None:
            bad_label.append(f"{raw_path} ({raw_label!r})")
            continue
        path = _resolve(raw_path, label, data_dir)
        if path is None:
            missing.append(raw_path)
            continue
        species, key = clean_species(r[scol]) if scol else ("", "")
        ts = parse_timestamp(path.name)
        date = r[dcol].strip() if dcol else ""
        if not date and pd.notna(ts):
            date = ts.date().isoformat()
        rows.append(dict(path=str(path), label=label, species=species, species_key=key,
                         plant_id=r[icol].strip() if icol else "", date=date, ts=ts))
    if blank:
        problems.append(f"{blank} empty rows ignored.")
    if bad_label:
        problems.append(f"{len(bad_label)} rows have an unrecognised label (skipped).")
    if missing:
        problems.append(f"{len(missing)} rows point to files that do not exist (skipped).")
    out = pd.DataFrame(rows)
    if out.empty:
        return out, problems + ["No usable rows."]
    out["ts"] = pd.to_datetime(out["ts"])

    out["md5"] = out["path"].map(lambda p: _md5(Path(p)))
    n_labels = out.groupby("md5")["label"].transform("nunique")
    conflict = n_labels > 1
    if conflict.any():
        names = sorted({Path(p).name for p in out[conflict]["path"]})
        problems.append(f"LABEL CONFLICT: {int(conflict.sum())} files filed under different classes (dropped): {names[:6]}")
        out = out[~conflict]
    dup = out.duplicated("md5", keep="first")
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate images removed.")
        out = out[~dup]
    out = out[~out.duplicated("path", keep="first")].reset_index(drop=True)

    need = out["plant_id"].eq("")
    if need.any():
        derived = derive_session_ids(out[need], session_gap)
        out.loc[need, "plant_id"] = derived.fillna("")
        n_auto = int(derived.notna().sum())
        if n_auto:
            problems.append(f"Built {derived.nunique()} pseudo plant ids from timestamps + species.")
    out["group"] = [
        f"plant:{pid}" if pid else f"day:{sp}|{d}" if (sp and d) else f"img:{path}"
        for pid, sp, d, path in zip(out["plant_id"], out["species"], out["date"], out["path"])
    ]
    return out.drop(columns=["md5", "species_key", "ts"]), problems

def make_splits(df: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.25,
                seed: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    df = df.copy()
    df["split"] = "ood"
    core_idx = df.index[df["label"].isin(config.CORE_CLASSES)]
    core = df.loc[core_idx]
    if len(core) < 8 or core["group"].nunique() < 4:
        raise SystemExit("Need at least 8 images in 4+ plant groups across the classes to split.")
    classes = sorted(core["label"].unique())
    best, best_score = None, -1
    for attempt in range(80):
        try:
            outer = GroupShuffleSplit(1, test_size=test_fraction, random_state=seed + attempt)
            trval, test = next(outer.split(core, groups=core["group"]))
            inner = GroupShuffleSplit(1, test_size=val_fraction / (1 - test_fraction), random_state=seed + attempt)
            tr, va = next(inner.split(core.iloc[trval], groups=core.iloc[trval]["group"]))
        except ValueError:
            continue
        te_lab, va_lab, tr_lab = (core.iloc[test]["label"], core.iloc[trval].iloc[va]["label"],
                                  core.iloc[trval].iloc[tr]["label"])
        score = sum(min((te_lab == c).sum(), 3) + min((va_lab == c).sum(), 3) + min((tr_lab == c).sum(), 3)
                    for c in classes)
        if score > best_score:
            best_score, best = score, (trval[tr], trval[va], test)
        if score == 9 * len(classes):
            break
    if best is None:
        raise SystemExit("Could not create group-level splits. Check plant_id values.")
    tr, va, te = best
    df.loc[core.index[tr], "split"] = "train"
    df.loc[core.index[va], "split"] = "val"
    df.loc[core.index[te], "split"] = "test"
    for c in classes:
        for sp in ("train", "val", "test"):
            n = int(((df["label"] == c) & (df["split"] == sp)).sum())
            if (sp == "test" and n < 3) or (sp == "val" and n < 2) or (sp == "train" and n < 6):
                notes.append(f"class '{c}' has only {n} image(s) in {sp}.")
    return df, notes

def load_splits(path: Path = config.SPLITS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def summarise(df: pd.DataFrame) -> str:
    table = pd.crosstab(df["label"], df["split"]).reindex(columns=["train", "val", "test", "ood"], fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.to_string()

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out", type=Path, default=config.SPLITS_PATH)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    df, problems = load_metadata(args.metadata, args.data_dir)
    for p in problems:
        print("WARNING:", p)
    if df.empty:
        raise SystemExit("Nothing usable in the metadata file.")
    df, notes = make_splits(df, args.val_fraction, args.test_fraction, args.seed)
    print("\nImages per class and split (plant-level):\n" + summarise(df))
    print(f"\nSpecies: {sorted(set(df['species']) - {''})}")
    for n in notes:
        print("NOTE:", n)
    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["group"]).to_csv(args.out, index=False)
        print(f"\nSplits saved -> {args.out}")

if __name__ == "__main__":
    main()