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
        limit = i
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
    data = np.loadtxt(
        shortFile, dtype=float, delimiter=";", usecols=range(1, length - 1))
    clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=length - 1)
    # features      n*p     [all rows, 1st column to last-1 column]
    X = data[:, 0:-1]
    Y = data[:, -1]  # target        1*p     [all rows, last column]

    classes = np.unique(clas)

    return X, Y, classes  # , class_name2num #, class_num2name


def train(model, alg, X, Y):
    if model is None:
        if alg is not None:
            model = alg
        else:
            return False

    fit_result = model.fit(X, Y)
    print('fit_result', fit_result)
    return model


def predict(model, X, Y):
    P = model.predict(X)
    D = []
    D = abs(P - Y)
    return P, D


def test_model(alg, X_train, Y_train, X_val, Y_val):
    model = train(None, alg, X_train, Y_train)
    P, D = predict(model, X_val, Y_val)

    print('D')
    print(D)
    print('max', D.max())
    print('min', D.min())
    print('rmse', RMSE(P, Y_val))

    # print('P')
    # print(P)
    return P


if __name__ == "__main__":
    x, y, classes, features = readlines("winequality-red.csv")

    alg1 = linear_model.LinearRegression()
    alg2 = svm.SVR()
    alg3 = KNeighborsRegressor(n_neighbors=3)
    alg4 = RadiusNeighborsRegressor(radius=1.0)

    x_train = x[:-100]
    y_train = y[:-100]
    x_val = x[:-10, :]
    y_val = y[:-10]

    PI = []
    i = 0

    Pi = test_model(alg1, x_train, y_train, x_val, y_val)
    PX = Pi
    i += 1

    Pi = test_model(alg2, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    Pi = test_model(alg3, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    Pi = test_model(alg4, x_train, y_train, x_val, y_val)
    PX += Pi
    i += 1

    PX = PX / i
    # print('P3')
    # print(P3)

    DX = abs(PX - y_val)
    print('Dx')
    print(DX)
    print('max', DX.max())
    print('min', DX.min())
    print('rmse', RMSE(PX, y_val))
