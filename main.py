#!/usr/local/bin/python3

import warnings
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
# from math import sqrt
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import RandomForestClassifier
# import time


def call_ac(est, X, Y):
    p = est.predict(X)
    return ac(p, Y)


def ac(pr, ta):
    A = 0
    size = 0
    for p, t in zip(pr, ta):
        A += 1 - (abs(p - t) / 10)
        size += 1
    return A / size


def call_rmse(est, X, y):
    p = est.predict(X)
    return RMSE(p, y)


def RMSE(predictions, targets):
    sq_e = 0
    size = 0
    minT = min(targets)
    maxT = max(targets)
    for predicted, target in zip(predictions, targets):
        sq_e += (abs(target - predicted)**2)
        size += 1

    if size == 0:
        print("size = 0")
        return 0
    if minT == maxT:
        print("all the same values")
        return np.sqrt(sq_e / size)
    else:
        print('got rmse')
        return np.sqrt(sq_e / size) / (maxT - minT)


def print_results(filename, delta):
    with open("delats_" + filename, 'w') as fo:
        for value in delta:
            fo.write(str(value) + "\n")


def readlines(filename, **kwargs):
    f = open(filename)
    feature_names = np.array(f.readline().replace(',', ';').replace(
        ' ', '_').split(";"))
    feature_names[-1] = feature_names[-1].replace('\n', '')
    shortFile = []
    i = 0
    while True:
        row = f.readline().replace(',', ';')
        if row == "":
            print("hit limit:", i)
            break
        shortFile.append(row)
        i += 1
    if shortFile:
        X, Y, Classes = parsedata(shortFile, len(feature_names))
        return X, Y, Classes, feature_names
    else:
        return None, None, None, None


def parsedata(shortFile, length):
    data = np.loadtxt(shortFile, dtype=float, delimiter=";")

    X = data[:, 0:-1]
    Y = data[:, -1]  # target        1*p     [all rows, last column]
    classes = np.unique(Y)

    return X, Y, classes


def train(alg, X, Y):
    model = alg
    fit_result = model.fit(X, Y)
    print('fit_result', fit_result)
    return model


def _predict(model, X, Y):
    P = model.predict(X)
    np.clip(P, 1, 10, out=P)
    # D = []
    D = P - Y
    return P, D


def test_model(alg, X, Y, X_train, Y_train, X_val, Y_val):
    model = train(alg, X_train, Y_train)
    P, D = _predict(model, X_val, Y_val)

    DX = D
    DA = abs(DX)
    print('Dx')
    print('max', DX.max())
    print('min', DX.min())
    print('max abs', DA.max())
    print('min abs', DA.min())
    print('rmse', RMSE(P, Y_val))
    scores = cross_val_score(model, X, Y, cv=5, scoring=call_ac)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return P


def test_all(algs, x, y, x_train, y_train, x_val, y_val):
    PX = 0
    i = 0
    Ps = []
    for alg in algs:
        Pi = test_model(alg, x, y, x_train, y_train, x_val, y_val)
        Ps.append(Pi)
        PX += Pi
        i += 1

    PX = PX / i
    DX = PX - y_val
    DA = abs(PX - y_val)
    print('D')
    print('max', DX.max())
    print('min', DX.min())
    print('max abs', DA.max())
    print('min abs', DA.min())
    print('rmse', RMSE(PX, y_val))
    # print first 25 vals and the predictions by each model
    # for i in range(25):
    # print(y_val[i], + ":" *[p[i] for p in Ps])
    return


def test_set(filename):
    x, y, classes, features = readlines(filename)
    algs = []
    algs.append(
        linear_model.LinearRegression(
            copy_X=True, fit_intercept=False, n_jobs=1, normalize=True))
    algs.append(
        RandomForestRegressor(
            bootstrap=True,
            criterion='mse',
            max_depth=None,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=26,
            n_jobs=-1,
            oob_score=True,
            random_state=None,
            verbose=0,
            warm_start=False))
    # algs.append(svm.LinearSVC(dual=False, penalty='l1'))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    for alg in algs:
        rmse, acc = test_alg_with_params(alg, x, y, x_train, y_train, x_val,
                                         y_val)
        row = [
            str(i),
            # str(alg(**p_set)).replace(',', '_'),
            str(alg(**p_set)).replace(',', ' ').replace('\n', '').replace(
                '\t', '').replace(' ', ''),
            str(i),
            str(rmse[0]),
            str(rmse[1]),
            str(i),
            str(acc[0]),
            str(acc[1])
        ]
        # rmse, acc = test_all(algs, x, y, x_train, y_train, x_val, y_val)


def test_alg_with_params(alg, x, y, x_train, y_train, x_val, y_val):

    model = train(alg, x_train, y_train)
    # P, D = _predict(model, x_val, y_val)

    # DX = D
    # DA = abs(DX)
    # print('Dx')
    # print('max', DX.max())
    # print('min', DX.min())
    # print('max abs', DA.max())
    # print('min abs', DA.min())
    # print('rmse', RMSE(P, Y_val))
    # rmse =  RMSE(P, y_val)
    scores = cross_val_score(model, x, y, cv=5, scoring=call_ac)
    acc_mean = scores.mean()
    acc_std = scores.std() * 2
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, x, y, cv=5, scoring=call_rmse)
    rmse_mean = scores.mean()
    rmse_std = scores.std() * 2
    print("rmse: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return (rmse_mean, rmse_std), (acc_mean, acc_std)


def test_alg(filename, alg_name, alg, param_sets):
    x, y, classes, features = readlines(filename)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    with open("results_" + filename, 'w') as fo:
        pass
        # fo.write()
    for i, p_set in enumerate(param_sets):
        rmse, acc = test_alg_with_params(
            alg(**p_set), x, y, x_train, y_train, x_val, y_val)
        row = [
            str(i),
            # str(alg(**p_set)).replace(',', '_'),
            str(alg(**p_set)).replace(',', ' ').replace('\n', '').replace(
                '\t', '').replace(' ', ''),
            str(i),
            str(rmse[0]),
            str(rmse[1]),
            str(i),
            str(acc[0]),
            str(acc[1])
        ]
        with open("results_" + filename, 'a') as fo:
            fo.write(", ".join(row) + "\n")


if __name__ == "__main__":
    red_sq = "squared_winequality-red.csv"
    red = "winequality-red.csv"
    white_sq = "squared_winequality-white.csv"
    white = "winequality-white.csv"
    both_sq = "squared_winequality-both.csv"
    file_names = [red_sq, red, white_sq, white, both_sq]

    # linear_param_sets = []
    # for i in range(0, 2):
    #     for j in range(0, 2):
    #         linear_param_sets.append({
    #             'fit_intercept': bool(i),
    #             'normalize': bool(j)
    #         })

    # rand_param_sets = []
    # crits = ['mse', 'mae']
    # max_f = ['auto', 'sqrt', 'log2'] + [*range(5, 11, 1)]
    # for n_ests in range(1, 30, 5):
    #     for crit in crits:
    #         for m_f in max_f:
    #             for oob in range(0, 2):
    #                 rand_param_sets.append({
    #                     'n_jobs': -1,
    #                     'n_estimators': int(n_ests),
    #                     'criterion': crit,
    #                     'max_features': m_f,
    #                     'oob_score': bool(oob)
    #                 })

    # for fname in file_names:
    #     test_alg(fname, "linear_regression", linear_model.LinearRegression,
    #              linear_param_sets)
    #     test_alg(fname, "random_forest", RandomForestRegressor,
    #              rand_param_sets)

    # algs.append(RandomForestRegressor(n_estimators=20, n_jobs=-1))

    # print('----red----')
    # test_set(red)
    print('----red_sq----')
    test_set(red_sq)
    # print('----white----')
    # test_set(white)
    # print('----white_sq----')
    # test_set(white_sq)
    # print('----both_sq----')
    # test_set(both_sq)
