from sklearn.datasets import make_classification
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from AnovaTransformer import AnovaFTransformer

np.set_printoptions(suppress=True)
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_features=20,
    n_redundant=4,
    n_informative=5,
    random_state=1234,
)


def calculate_f_value(X, y=None):
    X0 = X[y == 0]
    X1 = X[y == 1]
    mean_of_total = np.mean(X)
    mean_0 = np.mean(X0)
    mean_1 = np.mean(X1)
    total_ss = np.sum((X - mean_of_total) ** 2)  # sum of squares of all classes
    within_ss = np.sum((X0 - mean_0) ** 2) + np.sum((X1 - mean_1) ** 2)  # sum of squares within classes
    if len(X0) == len(X1):
        between_ss = ((mean_0 - mean_of_total) ** 2 + (mean_1 - mean_of_total) ** 2) * len(
            X0)  # sum of squares between classes
    p_w = len(X) - len(np.unique(y))  # degrees of freedom - denominator
    p_b = len(np.unique(y)) - 1  # degrees of freedom - numerator
    if 'between_ss' in locals():
        F = (between_ss / p_b) / (
                within_ss / p_w)  # only when it is possible to calculate sum of squares between classes
    else:
        F = ((total_ss - within_ss) / p_b) / (within_ss / p_w)
    return F


anova = AnovaFTransformer()
X_trans = anova.fit_transform(X, y, n_features=4)

skb = SelectKBest(f_classif, k=4)
skb.fit(X, y)
print("--------")
print(skb.transform(X))
print("--------")
print(X_trans)
print("--------")
print(skb.scores_)
print("--------")
print(anova.scores_)

print("--------")
print(X.shape)
print("--------")
print(X_trans.shape)
