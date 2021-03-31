import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from GRNN import GRNN, encode


def test_grnn_instance():
    assert isinstance(GRNN(), GRNN)


def test_grnn():
    gr = GRNN()

    X = np.array([[1, 0], [0, 1], [2, 1], [-2, -2]])
    y = np.array([1, 0, 1, 0])
    X_ = np.array([[3, 0]])
    gr.fit(X, y)
    assert gr.predict(X_) == 1


def test_encode():
    y = [0.2, 1.0, 1.6, 2.4, 2.6, 3]
    true_y = [0, 1.0, 2, 2, 2, 2]
    assert list(map(encode, y)) == true_y


if __name__ == "__main__":
    pass
