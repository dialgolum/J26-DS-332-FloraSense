import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/splits/splits.csv"); df["fold"] = df["fold"].astype(str)
X = np.load("data/processed/embeddings_bioclip.npy")
y = (df.species + " | " + df.variety).values

train = (df.fold != "test").values            # the test split stays unused for the final report
clf = LogisticRegression(max_iter=3000, C=10).fit(X[train], y[train])

Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
joblib.dump(clf, "models/checkpoints/classifier.joblib")
print("saved; classes:", list(clf.classes_))