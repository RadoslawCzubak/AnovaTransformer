from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np


class AnovaSelector(BaseEstimator, SelectorMixin):
    def _get_support_mask(self):
        return self.mask

    def __init__(self, k_features=5):
        self.k_features = k_features

    def fit_transform(self, X, y=None, **fit_params):
        f_scores = []
        for feature in X.T:
            f_scores.append(self.calculate_f_value(feature, y))
        self.scores_ = np.array(f_scores)
        self.set_mask(X)
        return self.get_best_columns(X)

    @staticmethod
    def calculate_f_value(X, y=None):
        clss = np.unique(y)
        clss.sort(axis=0)

        means = []
        for cls in clss:
            means.append(np.mean(X[y == cls]))
        means = np.array(means)
        mean_of_total = np.mean(X)
        total_ss = np.sum((X - mean_of_total) ** 2)  # sum of squares of all classes
        within_ss = 0

        for cls in clss:
            within_ss += np.sum((X[y == cls] - means[cls]) ** 2)  # sum of squares within classes

        p_w = len(X) - len(np.unique(y))  # degrees of freedom (within)
        p_b = len(np.unique(y)) - 1  # degrees of freedom (between)
        F_val = ((total_ss - within_ss) / p_b) / (within_ss / p_w)
        return F_val

    def get_scores(self):
        return self.scores_

    def get_best_columns(self, X):
        idx = np.argpartition(self.scores_, -self.k_features)[-self.k_features:]
        indices = idx[np.argsort((-self.scores_)[idx])]
        best_cols = X[:, sorted(indices)]
        return best_cols

    def set_mask(self, X):
        indices = np.argpartition(self.scores_, -self.k_features)[-self.k_features:]
        self.mask = np.zeros(X.shape[1], dtype=bool)
        for index in range(len(self.mask)):
            if index in indices:
                self.mask[index] = True
