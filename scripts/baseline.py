import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/splits/splits.csv")
df["fold"] = df["fold"].astype(str)
X = np.load("data/processed/embeddings_bioclip.npy")
y = (df.species + " | " + df.variety).values

def prototype(Xtr, ytr, Xte):
    cls = np.unique(ytr)
    P = np.stack([Xtr[ytr == c].mean(0) for c in cls])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return cls[(Xte @ P.T).argmax(1)]

def logreg(Xtr, ytr, Xte):
    return LogisticRegression(max_iter=3000, C=10).fit(Xtr, ytr).predict(Xte)

methods = {"prototype (nearest class mean)": prototype, "logistic regression": logreg}

print("== Cross-validation on validation folds ==")
for name, fn in methods.items():
    accs, f1s = [], []
    for k in "012":
        val = (df.fold == k).values
        tr = ((df.fold != k) & (df.fold != "test")).values
        pred = fn(X[tr], y[tr], X[val])
        accs.append(accuracy_score(y[val], pred))
        f1s.append(f1_score(y[val], pred, average="macro"))
    print(f"{name}: acc {np.mean(accs):.3f} ± {np.std(accs):.3f} | macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")