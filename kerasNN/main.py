import time
import numpy as np
import os

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from utils import read_tr

# tensorflow INFO, WARNING and ERROR are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def create_model(layers=3, neurons=25, init_mode='glorot_normal', activation='relu', lmb=0.0005, eta=0.001, alpha=0.7):
    model = Sequential()
    regularizer = l2(lmb)

    for i in range(layers):
        model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation, kernel_regularizer=regularizer))

    # add output layer
    model.add(Dense(2, activation='linear', kernel_initializer=init_mode))

    # set Stochastic Gradient Descent optimizer
    optimizer = SGD(learning_rate=eta, momentum=alpha)

    model.compile(optimizer=optimizer, loss=euclidean_distance_loss)

    return model


def keras_model_selection(x, y):

    # fix random seed for reproducibility
    seed = 27
    np.random.seed(seed)

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=64, verbose=0)

    # define the grid search parameters
    batch_size = [8, 16, 32, 64, 128]
    eta = np.arange(start=0.0005, stop=0.0011, step=0.0001)
    alpha = np.arange(start=0.4, stop=1, step=0.1)
    lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    neurons = [1, 5, 10, 15, 20, 25, 30]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb)

    start_time = time.time()
    print("Starting Grid Search...")

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10, scoring='r2', verbose=1)
    grid_result = grid.fit(x, y)

    end_time = time.time() - start_time
    print(f"Ended Grid Search. ({end_time})")

    # summarize results
    print(f"Best: {grid.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print(f"{mean} ({std}) with: {param}")


if __name__ == '__main__':
    # read training set
    x, y, x_t, y_t = read_tr(internal_test_set=True)

    keras_model_selection(x, y)
