import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import inf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier


def load_and_normilize_train_data(filepath, test_size=0.2, normilize=False):
    data = pd.read_csv(filepath, index_col=0)  # .to_numpy(dtype=object)

    data = data.replace([np.inf, -np.inf, np.nan], 0)
    data = data.drop(["smiles"], axis=1)
    cols = data.columns

    X = data.drop(['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"],
                  axis=1)
    col = X.columns
    if normilize:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=col)

    X = np.array(X)

    y = np.array(
        data[['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    y_train = y_train.astype('float32')
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    y_train[y_train == -inf] = 0.0
    y_train[y_train == inf] = 0.0
    X_train[X_train == -inf] = 0.0
    X_train[X_train == inf] = 0.0
    y_test[y_test == inf] = 0.0
    y_test[y_test == -inf] = 0.0
    X_test[X_test == inf] = 0.0
    X_test[X_test == -inf] = 0.0

    return X_train, X_test, y_train, y_test


def load_test_data(filepath, test_size=0.2, normilize=False):
    data = pd.read_csv(filepath, index_col=0)

    data = data.replace([np.inf, -np.inf, np.nan], 0)
    data = data.drop(["smiles"], axis=1)
    cols = data.columns

    if normilize:
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data = pd.DataFrame(data, columns=cols)

    X = np.array(data)

    X = X.astype("float32")

    X[X == -inf] = 0.0
    X[X == inf] = 0.0

    return X


def molecules_custom_dataset(filepath, normilize=False):
    data = pd.read_csv(filepath, index_col=0)

    data = data.replace([inf, -inf, np.nan], 0)

    data = data.drop(["smiles"], axis=1)
    cols = data.columns

    if normilize:
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data = pd.DataFrame(data, columns=cols)

    x = data.drop(
        ['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"], axis=1
    )
    X = np.array(x)

    y = np.array(
        data[['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"]])

    y = y.astype('float32')
    X = X.astype("float32")

    y[y == -inf] = 0.0
    y[y == inf] = 0.0
    X[X == -inf] = 0.0
    X[X == inf] = 0.0

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    return X, y


criterion_multi_classes = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
]

criterion_binary_loss = [
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
]


def load_models():
    model_RF = RandomForestClassifier(n_estimators=200, random_state=1234, max_features="auto", n_jobs=-1)

    model_SGD = SGDClassifier(loss="log", penalty="l2", max_iter=5000, learning_rate="adaptive", eta0=0.1, alpha=0.1,
                              tol=0.001, shuffle=True, random_state=1234, class_weight="balanced", n_jobs=-1)

    model_MLP = MLPClassifier(hidden_layer_sizes=3000, activation="relu", solver="adam", max_iter=3000, batch_size=64,
                              learning_rate="invscaling", learning_rate_init=0.001, random_state=1234)

    model_oneVsRest = OneVsRestClassifier(model_MLP)

    model_SVC = SVC(C=40, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                    tol=0.001, cache_size=200, class_weight="balanced", verbose=False, max_iter=- 1,
                    decision_function_shape='ovr', break_ties=True, random_state=1234)

    model_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=10,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto",
                                      random_state=1234, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                      min_impurity_split=None, class_weight=None, ccp_alpha=0.0)

    model_extratree = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=2,
                                          min_samples_leaf=1,
                                          min_weight_fraction_leaf=0.0, max_features='auto', random_state=None,
                                          max_leaf_nodes=None,
                                          min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                          ccp_alpha=0.0)

    model_KN = KNeighborsClassifier(n_neighbors=50, weights='uniform', algorithm='auto', leaf_size=300, p=2,
                                    metric='minkowski',
                                    metric_params=None, n_jobs=-1)

    model_perceptron = Perceptron(penalty="l2", alpha=0.0001, fit_intercept=True, max_iter=10000,
                                  tol=0.001, shuffle=True,
                                  verbose=0, eta0=1.0, n_jobs=None, random_state=1234, early_stopping=False,
                                  validation_fraction=0.1,
                                  n_iter_no_change=5, class_weight=None, warm_start=False)

    model_gaussian = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                                               max_iter_predict=100,
                                               warm_start=False, copy_X_train=True, random_state=None,
                                               multi_class='one_vs_rest',
                                               n_jobs=None)
    model_GB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                          criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                          min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                          min_impurity_split=None, init=None, random_state=None, max_features=None,
                                          verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
                                          n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    model_bernoulli = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    model_naivegaussian = GaussianNB(priors=None, var_smoothing=1e-09)
    return [model_RF, model_SGD, model_MLP, model_oneVsRest, model_SVC, model_DT, model_extratree, model_KN,
            model_perceptron, model_gaussian, model_GB, model_bernoulli, model_naivegaussian]


def load_model(model_name):
    models = {'RF': RandomForestClassifier(n_estimators=500, random_state=1234, max_features="auto", n_jobs=-1),
              'SGD': SGDClassifier(loss="log", penalty="l2", max_iter=5000, learning_rate="adaptive", eta0=0.1,
                                   alpha=0.1,
                                   tol=0.001, shuffle=True, random_state=1234, class_weight="balanced", n_jobs=-1),
              'MLP': MLPClassifier(hidden_layer_sizes=3000, activation="relu", solver="adam", max_iter=3000,
                                   batch_size=64,
                                   learning_rate="invscaling", learning_rate_init=0.001, random_state=1234),
              'oneVsRest': OneVsRestClassifier(
                  MLPClassifier(hidden_layer_sizes=3000, activation="relu", solver="adam", max_iter=3000, batch_size=64,
                                learning_rate="invscaling", learning_rate_init=0.001, random_state=1234)),
              'SVC': SVC(C=40, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                         tol=0.001, cache_size=200, class_weight="balanced", verbose=False, max_iter=- 1,
                         decision_function_shape='ovr', break_ties=True, random_state=1234),
              'DT': DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=10,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto",
                                           random_state=1234, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                           min_impurity_split=None, class_weight=None, ccp_alpha=0.0),
              'extratree': ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_features='auto', random_state=None,
                                               max_leaf_nodes=None,
                                               min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                               ccp_alpha=0.0),
              'KN': KNeighborsClassifier(n_neighbors=50, weights='uniform', algorithm='auto', leaf_size=300, p=2,
                                         metric='minkowski',
                                         metric_params=None, n_jobs=-1),
              'perceptron': Perceptron(penalty="l2", alpha=0.0001, fit_intercept=True, max_iter=10000,
                                       tol=0.001, shuffle=True,
                                       verbose=0, eta0=1.0, n_jobs=None, random_state=1234, early_stopping=False,
                                       validation_fraction=0.1,
                                       n_iter_no_change=5, class_weight=None, warm_start=False),
              'gaussian': GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                                                    max_iter_predict=100,
                                                    warm_start=False, copy_X_train=True, random_state=None,
                                                    multi_class='one_vs_rest',
                                                    n_jobs=None),
              'GB': GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                               min_impurity_split=None, init=None, random_state=None, max_features=None,
                                               verbose=0, max_leaf_nodes=None, warm_start=False,
                                               validation_fraction=0.1,
                                               n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0),
              'bernoulli': BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
              'naivegaussian': GaussianNB(priors=None, var_smoothing=1e-09)
              }
    return models[model_name]


models_encode_name = {0: "random forest", 1: "sgd", 2: "mlp", 3: "oneVsRest", 4: "svc", 5: "decision tree",
                      6: "extraTree",
                      7: "KN", 8: "perceptron", 9: "gaussian", 10: "GB", 11: "bernoulli", 12: "naivegaussian"}
