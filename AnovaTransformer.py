from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class AnovaFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, n_features=5):
        f_scores = []
        for feature in X.T:
            f_scores.append(self.calculate_f_value(feature, y))
        self.scores_ = np.array(f_scores)

        # get n columns with highest score
        idx = np.argpartition(self.scores_, -n_features)[-n_features:]
        indices = idx[np.argsort((-self.scores_)[idx])]
        new_X = X[:, sorted(indices)]
        return new_X

    @staticmethod
    def calculate_f_value(X, y=None):
        X0 = X[y == 0]
        X1 = X[y == 1]
        mean_of_total = np.mean(X)
        mean_0 = np.mean(X0)
        mean_1 = np.mean(X1)
        total_ss = np.sum((X - mean_of_total) ** 2)  # sum of squares of all classes
        within_ss = np.sum((X0 - mean_0) ** 2) + np.sum((X1 - mean_1) ** 2)  # sum of squares within classes
        p_w = len(X) - len(np.unique(y))  # degrees of freedom (within)
        p_b = len(np.unique(y)) - 1  # degrees of freedom (between)
        F_val = ((total_ss - within_ss) / p_b) / (within_ss / p_w)
        return F_val

    def get_scores(self):
        return self.scores_
