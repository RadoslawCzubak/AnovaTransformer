from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np


class NoSelector(BaseEstimator, SelectorMixin):
    def _get_support_mask(self):
        return self.mask

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.mask = np.ones(X.shape[1], dtype=1)
        return X