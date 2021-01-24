import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

# TensorFlow INFO, WARNING and ERROR are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TEAM_NAME = "MARIO"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_TEST_FILENAME = f"{TEAM_NAME}_ML-CUP20-TS.csv"


# read development set
# if its is True, it splits the provided dataset into development and internal test set
def read_tr(its=False):
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TR.csv")
    train = loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)

    x = train[:, :-2]
    y = train[:, -2:]

    if its:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        return x_train, y_train, x_test, y_test
    else:
        return x, y


# read provided blind test set
def read_ts():
    file = os.path.join(ROOT_DIR, "ml_cup_data", "ML-CUP20-TS.csv")
    test = loadtxt(file, delimiter=',', usecols=range(1, 11), dtype=np.float64)

    return test


def save_figure(model_name, **params):
    name = ""
    for k, v in params.items():
        name += f"{k}{v}_"
    name += ".png"

    # create plot directory if it doesn't exist
    dir_path = os.path.join(ROOT_DIR, model_name, "plot")
    os.makedirs(dir_path, exist_ok=True)

    # save plot as figure
    fig_path = os.path.join(dir_path, name)
    plt.savefig(fig_path, dpi=600)


# save predicted results on a csv file
def write_blind_results(y_pred):

    assert len(y_pred) == 472, "Not enough data were predicted! 472 predictions expected!"

    file = os.path.join(ROOT_DIR, BLIND_TEST_FILENAME)
    with open(file, "w") as f:
        print("# Jacopo Massa \t Giulio Purgatorio", file=f)
        print("# MARIO", file=f)
        print("# ML-CUP20", file=f)
        print("# 25/01/2021", file=f)

        pred_id = 1
        for p in y_pred:
            print("{},{},{}".format(pred_id, p[0], p[1]), file=f)
            pred_id += 1

    f.close()


# loss function for Keras and SVM models
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# it retrieves the mean value of all the passed losses
def euclidean_distance_score(y_true, y_pred):
    return np.mean(euclidean_distance_loss(y_true, y_pred))


scorer = make_scorer(euclidean_distance_score, greater_is_better=False)
