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


def model_selection(x, y, epochs=200):

    # fix random seed for reproducibility
    seed = 27
    np.random.seed(seed)

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=epochs, verbose=0)

    # define the grid search parameters
    eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.1)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    lmb = [float(round(i, 4)) for i in list(lmb)]

    batch_size = [16, 32, 64]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb, batch_size=batch_size)

    start_time = time.time()
    print("Starting Grid Search...\n")

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10,
                        return_train_score=True, scoring=scorer, verbose=1)

    grid_result = grid.fit(x, y)

    print("\nEnded Grid Search. ({:.4f})\n".format(time.time() - start_time))

    means_train = abs(grid_result.cv_results_['mean_train_score'])
    means_test = abs(grid_result.cv_results_['mean_test_score'])
    times_train = grid_result.cv_results_['mean_fit_time']
    times_test = grid_result.cv_results_['mean_score_time']
    params = grid_result.cv_results_['params']

    for m_ts, t_ts, m_tr, t_tr, p in sorted(zip(means_test, times_test, means_train, times_train, params)):
        print("{} \t TR {:.4f} (in {:.4f}) \t TS {:.4f} (in {:.4f})".format(p, m_tr, t_tr, m_ts, t_ts))

    print("\nBest: {:.4f} using {}\n".format(abs(grid.best_score_), grid_result.best_params_))

    best_params = grid_result.best_params_
    best_params['epochs'] = epochs

    return best_params


def predict(model, x_ts, x_its, y_its):

    y_ipred = model.predict(x_its)
    iloss = euclidean_distance_loss(y_its, y_ipred)

    y_pred = model.predict(x_ts)
    return y_pred, K.eval(iloss)


def plot_learning_curve(history, start_epoch=1, savefig=False, **kwargs):

    lgd = ['Loss TR']
    plt.plot(range(start_epoch, kwargs['epochs']), history['loss'][start_epoch:])
    if "val_loss" in history:
        plt.plot(range(start_epoch, kwargs['epochs']), history['val_loss'][start_epoch:])
        lgd.append('Loss VL')
    plt.legend(lgd)
    plt.title(f'Keras Learning Curve \n {kwargs}')

    if savefig:
        save_figure("kerasNN", **kwargs)

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


def keras_nn(ms=False):
    print("keras start")
    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    if ms:
        params = model_selection(x, y)
    else:
        # params = dict(eta=0.005, alpha=0.5, lmb=0.0005, epochs=200, batch_size=32)
        params = dict(eta=0.006, alpha=0.6, lmb=0.0007, epochs=200, batch_size=32)

    model = create_model(eta=params['eta'], alpha=params['alpha'], lmb=params['lmb'])

    res = model.fit(x, y, validation_split=0.3, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)
    tr_losses = res.history['loss']
    val_losses = res.history['val_loss']

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    print("keras end")

    plot_learning_curve(res.history, **params)


if __name__ == '__main__':
    keras_nn(ms=True)
