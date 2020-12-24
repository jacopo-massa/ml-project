import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
from numpy import loadtxt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import make_scorer

# PyTorch, TensorFlow.keras, SciKit
# -> 1: API low lvl, same performance of TensorFlow
# -> 2: API high lvl (easier, more concise)
# -> 3: exploit Model Selection + using another SW tool

# Un'altra rete per il MONK <- semplice, con Keras

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_tr(its=False):
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TR.csv")
    train = loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)

    x = train[:, :-2]
    y = train[:, -2:]

    if its:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        return x_train, y_train, x_test, y_test
    else:
        return x, y


def read_ts():
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TS.csv")
    test = loadtxt(file, delimiter=',', usecols=range(1, 11), dtype=np.float64)

    return test


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def euclidean_distance_score(y_true, y_pred):
    return np.mean(euclidean_distance_loss(y_true, y_pred))


scorer = make_scorer(euclidean_distance_score, greater_is_better=False)
