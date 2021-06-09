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
    n_informative=2,
    random_state=1234,
    n_clusters_per_class=1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=7777),
}

# ds = {'anova': AnovaFTransformer().fit_transform(X, y, n_features=3), 'skb': SelectKBest(

folds = 5
# scores = np.zeros((len(clfs), len(ds), folds))
# for idx, (data_id, dataset) in enumerate(ds.items()):
#     for fold_index, (train, test) in enumerate(skf.split(dataset, y)):
#         for clf_index, clf_name in enumerate(clfs):
#             clf = clone(clfs[clf_name])
#             clf.fit(dataset[train], y[train])
#             y_pred = clf.predict(dataset[test])
#             scores[clf_index, idx, fold_index] = accuracy_score(
#                 y[test], y_pred)

scores = np.empty((len(clfs), folds))
scores_pca = np.empty((len(clfs), folds))
scores_skb = np.empty((len(clfs), folds))
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
        X_pca_train, X_pca_test = X_pca[train], X_pca[test]

        clf = clone(clfs[clf_name])
        clf.fit(X_pca_train, y_train)
        y_pred = clf.predict(X_pca_test)
        scores_pca[clf_index][fold_index] = accuracy_score(y_test, y_pred)

        # skb

        skb = SelectKBest(f_classif, k=3)
        X_skb = skb.fit_transform(X, y)
        X_skb_train, X_skb_test = X_skb[train], X_skb[test]
        # y_skb_train, y_skb_test = y_skb[train], y_skb[test]

        clf = clone(clfs[clf_name])
        clf.fit(X_skb_train, y_train)
        y_pred = clf.predict(X_skb_test)
        scores_skb[clf_index][fold_index] = accuracy_score(y_test, y_pred)

        # anova

        X_anova = AnovaFTransformer().fit_transform(X, y, n_features=3)
        X_anova_train, X_anova_test = X_anova[train], X_anova[test]

        clf = clone(clfs[clf_name])
        clf.fit(X_anova_train, y_train)
        y_pred = clf.predict(X_anova_test)
        scores_anova[clf_index][fold_index] = accuracy_score(y_test, y_pred)

print("BAZOWE: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(clf_name + ': ' + str(np.mean(scores, axis=1)
          [clf_index]) + ' ('+str(np.std(scores, axis=1)[clf_index]) + ')')
print("PCA: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(clf_name + ': ' + str(np.mean(scores_pca, axis=1)
          [clf_index]) + ' ('+str(np.std(scores_pca, axis=1)[clf_index]) + ')')
print("SkB: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(clf_name + ': ' + str(np.mean(scores_skb, axis=1)
          [clf_index]) + ' ('+str(np.std(scores_skb, axis=1)[clf_index]) + ')')
print("ANOVA: ")
for clf_index, clf_name in (enumerate(clfs)):
    print(clf_name + ': ' + str(np.mean(scores_anova, axis=1)
                                [clf_index]) + ' ('+str(np.std(scores_anova, axis=1)[clf_index]) + ')')


ranks = []


# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# for name, label in [('A', 0), ('B', 1), ('C', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(float)
# ax.scatter(X_trans[:, 0], X_trans[:, 1], X_trans[:, 2], c=y, cmap=plt.cm.nipy_spectral,
#            edgecolor='k')

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# plt.show()
