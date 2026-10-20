import pandas as pd
from pathlib import Path

N_FOLDS = 3
df = pd.read_csv("data/metadata.csv")
df["label"] = df["species"] + " | " + df["variety"]
df["fold"] = ""

for label, g in df.groupby("label"):
    sections = sorted(g["plant_id"].unique())      # S1, S2, ... in order
    if len(sections) < 3:
        print(f"WARNING: {label} has only {len(sections)} sections, too few for train/val/test")
    test_sec, rest = sections[-1], sections[:-1]
    mask = df["label"] == label
    df.loc[mask & (df.plant_id == test_sec), "fold"] = "test"
    for i, sec in enumerate(rest):
        df.loc[mask & (df.plant_id == sec), "fold"] = str(i % N_FOLDS)

# no section may appear in more than one split
assert (df.groupby("plant_id")["fold"].nunique() == 1).all(), "LEAK"
print("No plant_id appears in more than one split.\n")

print(pd.crosstab(df["label"], df["fold"]).to_string())
Path("data/splits").mkdir(parents=True, exist_ok=True)
df.drop(columns="label").to_csv("data/splits/splits.csv", index=False)
print("\nSaved data/splits/splits.csv")