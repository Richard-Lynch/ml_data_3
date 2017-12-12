#!/usr/local/bin/python3

import warnings
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
from math import sqrt
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import time

# import tensorflow as tf

# from functools import wraps

# from algs import *
# from algs import sklAlg
# from algs import tensorAlg
# from algs import loadAlgs
# from algs import allMetrics
# from algs import alg_types
# from metrics import *
# from printer import printResults


def call_ac(est, X, Y):
    p = est.predict(X)
    return ac(p, y)


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
    total = 0
    minT = min(targets)
    maxT = max(targets)
    MinT = None
    MaxT = None
    for predicted, target in zip(predictions, targets):
        sq_e += (abs(target - predicted)**2)
        # print("taget:", target)
        # print("predicted:", predicted)
        # print("cur sq_e:", sq_e)
        # if MinT == None or MinT > target:
        # MinT = target
        # if MaxT == None or MaxT < target:
        # MaxT = target
        # if abs(predicted - target) < 0.0001:
        #     sq_e += 1
        size += 1

    if size == 0:
        print("size = 0")
        return 0
    # elif (maxT - minT) == 0:
    # print("dif = 0")
    # return 1
    else:
        print('got rmse')
        # print("size:", size)
        # print("sq_e:", sq_e)
        # print("sqrt:", np.sqrt(sq_e / size))
        # print("max -min :", maxT - minT)
        # print("Max,  Min :", MaxT, MinT)
        # print("Max - Min :", MaxT - MinT)
        return np.sqrt(sq_e / size) / (maxT - minT)


def readlines(filename, **kwargs):
    f = open(filename)
    feature_names = np.array(f.readline().replace(',', ';').replace(
        ' ', '_').split(";"))
    print(feature_names)
    feature_names[-1] = feature_names[-1].replace('\n', '')
    print(feature_names)
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


def print_results(filename, delta):
    with open("results_" + filename, 'w') as fo:
        for value in delta:
            print(value)
            fo.write(str(value) + "\n")
        # for row in shortFile:
        #     print('row', row)
        #     new_row = []
        #     for value in row:
        #         print('v', value)
        #         print(type(value))
        #         new_row.append(str(float(value)))
        #         new_row.append(str(float(value)**2))
        #     del new_row[-1]
        #     new_row_s = ", ".join(new_row)
        #     fo.write(new_row_s + '\n')


def parsedata(shortFile, length):
    data = np.loadtxt(shortFile, dtype=float, delimiter=";")
    # shortFile, dtype=float, delimiter=";", usecols=range(1, length - 1))
    # clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=length - 1)
    # features      n*p     [all rows, 1st column to last-1 column]
    X = data[:, 0:-1]
    Y = data[:, -1]  # target        1*p     [all rows, last column]

    classes = np.unique([1, 2])
    print('X')
    print(X)
    print('Y')
    print(Y)

    return X, Y, classes  # , class_name2num #, class_num2name


def train(alg, X, Y):
    model = alg
    # print('training:', model)

    fit_result = model.fit(X, Y)
    print('fit_result', fit_result)
    return model


def _predict(model, X, Y):
    # print('predicting:', model)
    # print('shape x', X.shape)
    P = model.predict(X)
    np.clip(P, 1, 10, out=P)
    D = []
    # print('calcing d')
    D = P - Y
    return P, D


def test_model(alg, X, Y, X_train, Y_train, X_val, Y_val):
    model = train(alg, X_train, Y_train)
    P, D = _predict(model, X_val, Y_val)

    print('D')
    DX = D
    DA = abs(DX)
    print('Dx')
    # print(DX[:10])
    print('max', DX.max())
    print('min', DX.min())
    print('max abs', DA.max())
    print('min abs', DA.min())
    print('rmse', RMSE(P, Y_val))
    scores = cross_val_score(model, X, Y, cv=5, scoring=call_ac)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # print('P')
    # print(P)
    return P


if __name__ == "__main__":
    filename = "squared_winequality-red.csv"
    x, y, classes, features = readlines(filename)
    # x, y, classes, features = readlines("squared_winequality-white.csv")
    # x, y, classes, features = readlines("winequality-red.csv")
    # x, y, classes, features = readlines("winequality-white.csv")

    alg1 = linear_model.LinearRegression()
    alg2 = svm.SVR()
    alg3 = KNeighborsRegressor(n_neighbors=100)
    alg4 = RadiusNeighborsRegressor(radius=1000.0)
    # alg4 = RadiusNeighborsRegressor(radius=10000.0)
    # alg5 = svm.SVR(kernel='linear')
    alg6 = RandomForestRegressor()
    alg7 = linear_model.LogisticRegression()
    alg8 = linear_model.LinearRegression(normalize=True)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=0)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    # x_train = x[:-1000]
    # y_train = y[:-1000]
    # x_val = x[-1000:, :]
    # y_val = y[-1000:]

    PX = 0
    PI = []
    i = 0

    Pi = test_model(alg1, x, y, x_train, y_train, x_val, y_val)
    PX = Pi
    i += 1

    Pi = test_model(alg2, x, y, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    Pi = test_model(alg3, x, y, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    # Pi = test_model(alg4, x_train, y_train, x_val, y_val)
    # P_rad = Pi
    # PX += Pi
    # i += 1

    Pi = test_model(alg6, x, y, x_train, y_train, x_val, y_val)
    P_for = Pi
    PX += Pi
    i += 1

    Pi = test_model(alg7, x, y, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    Pi = test_model(alg8, x, y, x_train, y_train, x_val, y_val)
    P_lin = Pi
    PX += Pi
    i += 1

    PX = PX / i
    # print('P3')
    # print(P3)

    DX = PX - y_val
    DA = abs(PX - y_val)
    print('Dx')
    # print(DX[:10])
    print('max', DX.max())
    print('min', DX.min())
    print('max abs', DA.max())
    print('min abs', DA.min())
    print('rmse', RMSE(PX, y_val))

    D_lin = P_lin - y_val
    print_results(filename, D_lin)

    # print('radius')
    # print(P_rad[:10])
    # print(P_for[:10])
    # print(y_val[:10])
