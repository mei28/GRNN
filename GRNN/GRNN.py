from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GRNN(BaseEstimator, RegressorMixin):
    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = sigma

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.is_fitted_ = True
        return self

    def predict(self, X):
        # check if fit() had been called
        check_is_fitted(self, ["X_", "y_"])

        Kernel_def_ = RBF(length_scale=self.sigma)
        K = Kernel_def_(self.X_, X)
        # if the distance is very high/low, the nan must be prevented
        K = np.nan_to_num(K)

        # validate input
        X = check_array(X)
        psum = K.sum(axis=0).T
        psum = np.nan_to_num(psum)
        return np.nan_to_num(np.dot(self.y_.T, K) / psum)
