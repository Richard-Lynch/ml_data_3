#!/usr/local/bin/python3

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
