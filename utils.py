import os
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split


# PyTorch, TensorFlow.keras, SciKit
# -> 1: API low lvl, same performance of TensorFlow
# -> 2: API high lvl (easier, more concise)
# -> 3: exploit Model Selection + using another SW tool

# Un'altra rete per il MONK <- semplice, con Keras

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_tr(internal_test_set=False):
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TR.csv")
    train = loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)

    x = train[:, :-2]
    y = train[:, -2:]

    if internal_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        return x_train, y_train, x_test, y_test
    else:
        return x, y


def read_ts():
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TS.csv")
    test = loadtxt(file, delimiter=',', usecols=range(1, 11), dtype=np.float64)

    x = test[:, :-2]
    y = test[:, -2:]

    return x, y


if __name__ == '__main__':
    read_tr()
