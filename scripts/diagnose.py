import numpy as np, pandas as pd
from collections import Counter

df = pd.read_csv("data/splits/splits.csv"); df["fold"] = df["fold"].astype(str)
X = np.load("data/processed/embeddings_bioclip.npy")
y = (df.species + " | " + df.variety).values

def prototype(Xtr, ytr, Xte):
    cls = np.unique(ytr)
    P = np.stack([Xtr[ytr == c].mean(0) for c in cls])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return cls[(Xte @ P.T).argmax(1)]

# 1) which classes fail?
errs, per_cls = Counter(), {}
for k in "012":
    val = (df.fold == k).values
    tr = ((df.fold != k) & (df.fold != "test")).values
    pred = prototype(X[tr], y[tr], X[val])
    for t, p in zip(y[val], pred):
        per_cls.setdefault(t, []).append(t == p)
        if t != p: errs[(t, p)] += 1
print("== Per-class validation accuracy ==")
for c, r in sorted(per_cls.items(), key=lambda kv: np.mean(kv[1])):
    print(f"{c:<45}{np.mean(r):.3f}  (n={len(r)})")
print("\n== Most common mistakes (true -> predicted) ==")
for (t, p), n in errs.most_common(8):
    print(f"{n:>3}  {t}  ->  {p}")

# 2) near-duplicate check: how similar is each validation image to its closest training image?
sims = []
for k in "012":
    val = np.where(df.fold == k)[0]
    tr = np.where((df.fold != k) & (df.fold != "test"))[0]
    sims += (X[val] @ X[tr].T).max(1).tolist()
sims = np.array(sims)
print("\n== Max cosine similarity to a training image ==")
print("median %.3f | 90th pct %.3f | share > 0.97: %.1f%%" %
      (np.median(sims), np.percentile(sims, 90), 100 * (sims > 0.97).mean()))

# 3) hold out a whole bed (only for varieties grown in 2+ beds)
print("\n== Whole-bed holdout ==")
for v, g in df[df.fold != "test"].groupby(["species", "variety"]):
    beds = g.batch_id.unique()
    if len(beds) < 2: continue
    accs = []
    for b in beds:
        te = ((df.species == v[0]) & (df.variety == v[1]) & (df.batch_id == b) & (df.fold != "test")).values
        tr = ((df.fold != "test") & ~te).values
        accs.append((prototype(X[tr], y[tr], X[te]) == y[te]).mean())
    print(f"{v[0]} | {v[1]:<22} beds={len(beds)}  held-out-bed acc: {np.mean(accs):.3f}")