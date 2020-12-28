import time
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold

from utils import *


def create_model(layers=3, n_units=30, init_mode='glorot_normal', activation='tanh', lmb=0.0005, eta=0.001, alpha=0.7):
    model = Sequential()

    for i in range(layers):
        model.add(Dense(n_units, kernel_initializer=init_mode, activation=activation, kernel_regularizer=l2(lmb)))

    # add output layer
    model.add(Dense(2, activation='linear', kernel_initializer=init_mode))

    # set Stochastic Gradient Descent optimizer
    optimizer = SGD(learning_rate=eta, momentum=alpha)

    model.compile(optimizer=optimizer, loss=euclidean_distance_loss)

    return model


def model_selection(x, y, epochs=200, batch_size=32):

    # fix random seed for reproducibility
    seed = 27
    np.random.seed(seed)

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

    """# define the grid search parameters
    eta = np.arange(start=0.0005, stop=0.0011, step=0.0002)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.2)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0002)
    lmb = [float(round(i, 4)) for i in list(lmb)]

    batch_size = [16, 32, 64, 128]
    neurons = [15, 20, 25, 30]"""


    eta = [0.1, 0.01]
    alpha = [0.5, 0.6]
    lmb = [0.0005]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb)

    start_time = time.time()
    print("Starting Grid Search...")

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,
                        return_train_score=True, scoring=scorer, verbose=2)

    grid_result = grid.fit(x, y)

    best_params = grid_result.best_params_
    best_params['epochs'] = epochs
    best_params['batch_size'] = batch_size

    end_time = time.time() - start_time
    print(f"Ended Grid Search. ({end_time})")

    print(f"Best: {abs(grid.best_score_)} using {grid_result.best_params_}")

    return best_params


def predict(model, x_ts, x_its, y_its):

    y_ipred = model.predict(x_its)
    iloss = rmse(y_its, y_ipred)

    y_pred = model.predict(x_ts)

    return y_pred, K.eval(iloss)


def plot_learning_curve(history, start_epoch=1, **kwargs):

    lgd = ['Loss TR']
    plt.plot(range(start_epoch, kwargs['epochs']), history['loss'][start_epoch:])
    if "val_loss" in history:
        plt.plot(range(start_epoch, kwargs['epochs']), history['val_loss'][start_epoch:])
        lgd.append('Loss VL')
    plt.legend(lgd)
    plt.title(f'Keras Learning Curve \n {kwargs}')

    name = ""
    for k, v in kwargs.items():
        name += f"{k}{v}_"
    name += ".png"

    path = os.path.join(ROOT_DIR, "kerasNN", "plot", name)
    plt.savefig(path, dpi=600)
    plt.show()


def cross_validation(x, y, eta, alpha, lmb, n_splits=10, epochs=200, batch_size=64):

    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    model_cv = create_model(eta=eta, alpha=alpha, lmb=lmb)

    cv_loss = []
    lgd = []
    fold_idx = 0
    for tr_idx, vl_idx in kfold.split(x, y):
        print(f"Starting fold {fold_idx}")
        res_cv = model_cv.fit(x[tr_idx], y[tr_idx], epochs=epochs, batch_size=batch_size,
                              validation_data=(x[vl_idx], y[vl_idx]), verbose=0)

        loss_tr = res_cv.history['loss']
        loss_vl = res_cv.history['val_loss']
        cv_loss.append([loss_tr[-1], loss_vl[-1]])

        plt.plot(loss_tr)
        plt.plot(loss_vl)

        lgd.append(f'Loss TR {fold_idx}')
        lgd.append(f'Loss VL {fold_idx}')
        fold_idx += 1

        print(f"Ended fold {fold_idx}, with {loss_tr[-1]} - {loss_vl[-1]}")

    # plot and save cv results
    param = dict(eta=eta, alpha=alpha, lmb=lmb, epochs=epochs, batch_size=batch_size)

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

    # mean of loss on TR and VL
    return model_cv, list(np.mean(cv_loss, axis=0))


if __name__ == '__main__':
    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    params = model_selection(x, y)

    model = create_model(eta=params['eta'], alpha=params['alpha'], lmb=params['lmb'])

    res = model.fit(x, y, validation_split=0.3, epochs=params['epochs'], batch_size=params['batch_size'], verbose=2)
    tr_losses = res.history['loss']
    val_losses = res.history['val_loss']

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))










