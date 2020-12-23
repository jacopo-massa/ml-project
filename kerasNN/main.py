import time
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from utils import *

# tensorflow INFO, WARNING and ERROR are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def create_model(layers=3, neurons=25, init_mode='glorot_normal', activation='tanh', lmb=0.0005, eta=0.001, alpha=0.7):
    model = Sequential()

    for i in range(layers):
        model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation, kernel_regularizer=l2(lmb)))

    # add output layer
    model.add(Dense(2, activation='linear', kernel_initializer=init_mode))

    # set Stochastic Gradient Descent optimizer
    optimizer = SGD(learning_rate=eta, momentum=alpha)

    model.compile(optimizer=optimizer, loss=rmse)

    return model


def model_selection(x, y):

    # fix random seed for reproducibility
    seed = 27
    np.random.seed(seed)

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=64, verbose=0)

    # define the grid search parameters
    eta = np.arange(start=0.0005, stop=0.0011, step=0.0002)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.2)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0002)
    lmb = [float(round(i, 4)) for i in list(lmb)]

    batch_size = [8, 16, 32, 64, 128]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    neurons = [1, 5, 10, 15, 20, 25, 30]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb)

    start_time = time.time()
    print("Starting Grid Search...")

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10,
                        return_train_score=True, scoring='neg_root_mean_squared_error', verbose=1, refit=True)

    grid_result = grid.fit(x, y)

    print(f"Best: {grid.best_score_} using {grid_result.best_params_}")
    hist = grid_result.best_estimator_.model.history.history
    plot_learning_curve(hist)

    end_time = time.time() - start_time
    print(f"Ended Grid Search. ({end_time})")

    return grid_result.best_params_


def predict(model, x_ts, x_its, y_its):

    y_ipred = model.predict(x_its)
    iloss = euclidean_distance_loss(y_its, y_ipred)

    y_pred = model.predict(x_ts)

    return y_pred, iloss


def plot_learning_curve(history, start_epoch=1, **kwargs):

    lgd = ['Loss TR']
    plt.plot(range(start_epoch, kwargs['epochs']), history['loss'][start_epoch:])

    if "val_loss" in history:
        plt.plot(history['val_loss'])
        lgd.append('Loss VL')
    plt.legend(lgd)
    plt.title(f'Learning Curve Keras \n {kwargs}')

    name = ""
    for k, v in kwargs.items():
        name += f"{k}{v}_"
    name += ".png"

    path = os.path.join(ROOT_DIR, "kerasNN", "plot", name)
    plt.savefig(path, dpi=600)
    plt.show()


def cross_validation(x, y, eta, alpha, lmb, n_splits=10, epochs=200, batch_size=64):

    model_cv = create_model(eta=eta, alpha=alpha, lmb=lmb)
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)

    cv_loss = []
    lgd = []
    fold_idx = 0
    for tr_idx, vl_idx in kfold.split(x):
        print(f"Starting fold {fold_idx}")
        res_cv = model_cv.fit(x[tr_idx], y[tr_idx], epochs=epochs, batch_size=batch_size,
                              validation_data=(x[vl_idx], y[vl_idx]), verbose=0)

        loss_tr = res_cv.history['loss']
        loss_vl = res_cv.history['val_loss']
        cv_loss.append(loss_tr[-1])
        cv_loss.append(loss_vl[-1])

        plt.plot(loss_tr)
        plt.plot(loss_vl)

        lgd.append(f'Loss TR {fold_idx}')
        lgd.append(f'Loss VL {fold_idx}')
        fold_idx += 1

        print(f"Ended fold {fold_idx}, with {loss_tr[-1]} - {loss_vl[-1]}")

    # plot and save cv results
    param = dict(alpha=alpha, eta=eta, lmb=lmb, epochs=epochs, batch_size=batch_size)

    plt.legend(lgd)
    plt.title(f"Keras Cross Validation \n {param}")

    name = "cv_"
    for k, v in param.items():
        name += f"{k}{v}_"
    name += ".png"

    path = os.path.join(ROOT_DIR, "kerasNN", "plot", name)
    plt.savefig(path, dpi=600)
    plt.show()

    # retrain model on the entire TR
    res_final = model_cv.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    plot_learning_curve(res_final.history, **param)


if __name__ == '__main__':
    # read training set
    x, y, x_t, y_t = read_tr(its=True)

    """# eta = 0.0009, alpha = 0.8, lmb = 0.0009, epochs = 200, batch_size = 64

    model = create_model(eta=params['eta'], alpha=params['alpha'], lmb=params['lmb'])

    res = model.fit(x, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=2)

    plot_learning_curve(res.history, 30, **params)"""

    params = dict(eta=0.01, alpha=0.55, lmb=0.0004, epochs=200, batch_size=64)
    cross_validation(x, y, **params, n_splits=4)




