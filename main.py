#!/usr/local/bin/python3

import warnings
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor


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
    D = P - Y
    return P, D


def test_model(alg, X, Y, X_train, Y_train, X_val, Y_val):
    model = train(alg, X_train, Y_train)
    P, D = _predict(model, X_val, Y_val)

    DX = D
    DA = abs(DX)

    return DX, DA


def test_model_cross(alg, X, Y, X_train, Y_train, X_val, Y_val):
    P = cross_val_predict(alg, X, Y, cv=5, n_jobs=-1)
    DX = P - Y
    DA = abs(DX)
    return DX, DA, P


def get_co(alg, X, Y):
    return train(alg, X, Y).coef_


def test_set(filename):
    # test a specific set against both algs, with optimum params
    x, y, classes, features = readlines(filename)
    # add two algs to be tested
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

    # split data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    # test each alg against set
    for i, alg in enumerate(algs):
        rmse, acc = test_alg_with_params(alg, x, y, x_train, y_train, x_val,
                                         y_val)
        # DX, DA = test_model(alg, x, y, x_train, y_train, x_val, y_val)
        DX, DA, P = test_model_cross(alg, x, y, x_train, y_train, x_val, y_val)
        if i == 0:
            coef = get_co(alg, x, y)
            print(type(coef))
            print(coef)
            with open('coefs_' + filename, 'w') as fo:
                fo.write(','.join(features) + '\n')
                fo.write(','.join([str(c) for c in coef]) + '\n')
        # write deltas to file
        with open("deltas_" + str(i) + filename, 'w') as fo:
            fo.write("\n".join([str(P) + "," + str(D) for D, P in zip(DX, P)]))
        # create row
        row = [
            str(i),
            str(alg).replace(',', ' ').replace('\n', '').replace('\t',
                                                                 '').replace(
                                                                     ' ', ' '),
            str(i),
            str(rmse[0]),
            str(rmse[1]),
            str(i),
            str(acc[0]),
            str(acc[1])
        ]
        # write row to file
        with open("results_" + filename, 'a') as fo:
            fo.write(", ".join(row) + "\n")


# ---- decide on params -----
def test_alg_with_params(alg, x, y, x_train, y_train, x_val, y_val):
    # test alg with specific param set
    model = train(alg, x_train, y_train)
    # get accuracy
    scores = cross_val_score(model, x, y, cv=5, scoring=call_ac)
    acc_mean = scores.mean()
    acc_std = scores.std() * 2
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # get RMSE
    scores = cross_val_score(model, x, y, cv=5, scoring=call_rmse)
    rmse_mean = scores.mean()
    rmse_std = scores.std() * 2
    print("rmse: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # return as tuples
    return (rmse_mean, rmse_std), (acc_mean, acc_std)


def test_alg_with_param_set(filename, alg_name, alg, param_sets):
    # test the alg with each param_set
    x, y, classes, features = readlines(filename)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # create blank file
    with open("results_" + filename, 'w') as fo:
        pass
    # test the alg against each param set
    for i, p_set in enumerate(param_sets):
        # retrieve the rmse and acc for alg and paramset
        rmse, acc = test_alg_with_params(
            alg(**p_set), x, y, x_train, y_train, x_val, y_val)
        # create row to be saved to file
        row = [
            str(i),
            str(alg(**p_set)).replace(',', ' ').replace('\n', '').replace(
                '\t', '').replace(' ', ''),
            str(i),
            str(rmse[0]),
            str(rmse[1]),
            str(i),
            str(acc[0]),
            str(acc[1])
        ]
        # write row to file
        with open("results_" + filename, 'a') as fo:
            fo.write(", ".join(row) + "\n")


if __name__ == "__main__":
    red_sq = "squared_winequality-red.csv"
    red = "winequality-red.csv"
    white_sq = "squared_winequality-white.csv"
    white = "winequality-white.csv"
    both_sq = "squared_winequality-both.csv"
    file_names = [red_sq, red, white_sq, white, both_sq]

    # Used to eval the best param sets to use.
    # Should have used a built in scikit learn instead
    # ---- create param sets for linear regression -----
    linear_param_sets = []
    for i in range(0, 2):
        for j in range(0, 2):
            linear_param_sets.append({
                'fit_intercept': bool(i),
                'normalize': bool(j)
            })

    # ----- create param sets for random forest -----
    rand_param_sets = []
    crits = ['mse', 'mae']
    max_f = ['auto', 'sqrt', 'log2'] + [*range(5, 11, 1)]
    for n_ests in range(1, 30, 5):
        for crit in crits:
            for m_f in max_f:
                for oob in range(0, 2):
                    rand_param_sets.append({
                        'n_jobs': -1,
                        'n_estimators': int(n_ests),
                        'criterion': crit,
                        'max_features': m_f,
                        'oob_score': bool(oob)
                    })

    # test each param set, and save results to file
    # for fname in file_names:
    #     test_alg_with_param_set(fname, "linear_regression", linear_model.LinearRegression,
    #              linear_param_sets)
    #     test_alg_with_param_set(fname, "random_forest", RandomForestRegressor,
    #              rand_param_sets)

    # ---- Train and Test ----
    # actually train and test model against dataset
    print('----red----')
    test_set(red)
    # print('----red_sq----')
    # test_set(red_sq)
    # print('----white----')
    # test_set(white)
    # print('----white_sq----')
    # test_set(white_sq)
    # print('----both_sq----')
    # test_set(both_sq)
