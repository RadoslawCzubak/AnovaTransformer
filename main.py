from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.datasets import make_classification
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest
from AnovaTransformer import AnovaFTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
np.set_printoptions(suppress=True)
X, y = make_classification(
    n_samples=20,
    n_classes=3,
    n_features=20,
    n_redundant=4,
    n_informative=10,
    random_state=1234,
    n_clusters_per_class=1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=7777),
}

folds = 5

scores = np.empty((len(clfs), folds))
scores_pca = np.empty((len(clfs), folds))
scores_anova = np.empty((len(clfs), folds))

for fold_index, (train, test) in enumerate(skf.split(X, y)):
    for clf_index, clf_name in enumerate(clfs):

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        # base

        clf = clone(clfs[clf_name])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores[clf_index][fold_index] = accuracy_score(y_test, y_pred)

        # pca

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        # print(
        #     f"Explained variance ratio: {sum(pca.explained_variance_ratio_[:3])}")

        X_pca_train, X_pca_test = X_pca[train], X_pca[test]

        clf = clone(clfs[clf_name])
        clf.fit(X_pca_train, y_train)
        y_pred = clf.predict(X_pca_test)
        scores_pca[clf_index][fold_index] = accuracy_score(y_test, y_pred)

        # anova

        X_anova = AnovaFTransformer().fit_transform(X, y, n_features=3)
        X_anova_train, X_anova_test = X_anova[train], X_anova[test]

        clf = clone(clfs[clf_name])
        clf.fit(X_anova_train, y_train)
        y_pred = clf.predict(X_anova_test)
        scores_anova[clf_index][fold_index] = accuracy_score(y_test, y_pred)

print("BAZOWE: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(
        f"{clf_name}: {np.mean(scores, axis=1)[clf_index]} ({np.std(scores, axis=1)[clf_index]:.3})")
print("PCA: ")

for clf_index, clf_name in (enumerate(clfs)):
    print(
        f"{clf_name}: {np.mean(scores_pca, axis=1)[clf_index]} ({np.std(scores_pca, axis=1)[clf_index]:.3})")
print("ANOVA: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(
        f"{clf_name}: {np.mean(scores_anova, axis=1)[clf_index]} ({np.std(scores_anova, axis=1)[clf_index]:.3})")


ranks = []
