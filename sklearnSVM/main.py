from utils import *

from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


def model_selection(x, y):
    # fix random seed for reproducibility
    seed = 27
    np.random.seed(seed)

    svr = SVR()
    model = MultiOutputRegressor(svr)

    # define the grid search parameters
    epsilon = np.arange(start=0.1, stop=0.9, step=0.1)
    epsilon = [float(round(i, 4)) for i in list(epsilon)]

    param_grid = [{'estimator__kernel': ['rbf'],
                   'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4, 'auto', 'scale'],
                   'estimator__C': [5, 10, 15, 25],
                   'estimator__epsilon': epsilon}]

    start_time = time.time()
    print("Starting Grid Search...")

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

    return grid.best_params_


def predict(model, x_ts, x_its, y_its):
    # predict on internal test set
    iloss = K.eval(euclidean_distance_loss(y_its, model.predict(x_its)))
    # predict on blind test set
    y_pred = model.predict(x_ts)
    # return predicted target on blind test set,
    # and losses on internal test set
    return y_pred, iloss


def plot_learning_curve(model, x, y, savefig=False):

    # dictify model's parameters
    p = model.get_params()
    params = dict(kernel=p['estimator__kernel'], C=p['estimator__C'],
                  gamma=p['estimator__gamma'], eps=p['estimator__epsilon'])

    # plot learning curve by training and scoring the model for different train sizes
    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(model, x, y, train_sizes=np.linspace(0.1, 1, 50),
                       n_jobs=-1, scoring=scorer, cv=10, verbose=1)

    plt.plot(train_sizes, np.mean(np.abs(train_scores_svr), axis=1))
    plt.plot(train_sizes, np.mean(np.abs(test_scores_svr), axis=1))
    plt.xlabel("Train size")
    plt.ylabel("Loss")
    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'SVR Learning curve \n {params}')

    if savefig:
        save_figure("sklearnSVM", **params)

    plt.show()


def sklearn_svm(ms=False):
    print("sklearn start")

    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    # choose model selection or hand-given parameters
    if ms:
        params = model_selection(x, y)
    else:
        params = dict(estimator__kernel='rbf', estimator__C=8, estimator__epsilon=0.6, estimator__gamma='scale')

    # create model and fit the model
    svr = SVR(kernel=params['estimator__kernel'], C=params['estimator__C'],
              gamma=params['estimator__gamma'], epsilon=params['estimator__epsilon'])

    # we use MOR to perform the multi-output regression task
    model = MultiOutputRegressor(svr)

    # split development set into train and test set
    x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, test_size=0.3)
    model.fit(x_tr, y_tr)

    tr_losses = euclidean_distance_loss(y_tr, model.predict(x_tr))
    val_losses = euclidean_distance_loss(y_vl, model.predict(x_vl))

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", np.mean(tr_losses))
    print("VL Loss: ", np.mean(val_losses))
    print("TS Loss: ", np.mean(ts_losses))

    print("\nsklearn end")

    plot_learning_curve(model, x, y)


if __name__ == '__main__':
    sklearn_svm()
