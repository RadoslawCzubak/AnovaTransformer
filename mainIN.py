from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.datasets import make_classification
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from AnovaSelector import AnovaSelector
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

np.set_printoptions(suppress=True)
X, y = make_classification(
    n_samples=10000,
    n_classes=3,
    n_features=20,
    n_redundant=4,
    n_informative=10,
    random_state=1234,
    n_clusters_per_class=1,
)

X = MinMaxScaler().fit_transform(X, y)

k_features = 10

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=7777),
}

datasets = {
    'Base': X,
    'Anova': AnovaSelector(k_features=k_features),
    'SKB': SelectKBest(chi2, k=k_features),
    'PCA': PCA(n_components=k_features)
}

folds = 5
scores = np.zeros((len(clfs), len(datasets), folds))
for dataset_idx, dataset_name in enumerate(datasets):
    for fold_index, (train, test) in enumerate(skf.split(X, y)):
        if dataset_idx == 0:
            data_train, data_test = datasets[dataset_name][train], datasets[dataset_name][test]
        else:
            data_train, data_test = datasets[dataset_name].fit_transform(X[train], y[train]), \
                                    datasets[dataset_name].transform(X[test])
        for clf_index, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(data_train, y[train])
            y_pred = clf.predict(data_test)
            scores[clf_index, dataset_idx, fold_index] = accuracy_score(
                y[test], y_pred)

print(datasets['Anova'].mask)

for dataset_id, dataset_name in enumerate(datasets):
    print(f"{dataset_name}")
    means = np.mean(scores[:, dataset_id, :], axis=1)
    std_devs = np.std(scores[:, dataset_id, :], axis=1)
    for clf_idx, clf_name in enumerate(clfs):
        print(f'{clf_name}: {str(means[clf_idx].round(2))}({str(std_devs[clf_idx].round(2))})')

alpha = .05
t_statistic = np.zeros((len(datasets), len(datasets)))
p_value = np.zeros((len(datasets), len(datasets)))

for i in range(len(datasets)):
    for j in range(len(datasets)):
        i_mean, j_mean = np.mean(scores[:, i, :], axis=1), np.mean(scores[:, j, :], axis=1)
        t_statistic[i, j], p_value[i, j] = ttest_rel(i_mean, j_mean)

headers = datasets.keys()
names_column = np.array([[key] for key in datasets.keys()])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(datasets), len(datasets)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(datasets), len(datasets)))
significance[p_value <= alpha] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print(f"Statistical significance (alpha = {alpha}):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
