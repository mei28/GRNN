import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from GRNN import GRNN, encode
from icecream import ic
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    print("loaded iris dataset.")

    skf = StratifiedKFold(n_splits=5)
    scores = []

    for i, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        x_trn, y_trn = X[trn_idx], y[trn_idx]
        x_val, y_val = X[val_idx], y[val_idx]
        grnn = GRNN(sigma=0.5)
        grnn.fit(x_trn, y_trn)
        pred = list(map(encode, grnn.predict(x_val)))
        score = accuracy_score(y_val, pred)
        ic(score)
        scores.append(score)
