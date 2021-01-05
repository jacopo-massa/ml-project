import matplotlib.pyplot as plt
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

    # Set the parameters by cross-validation
    epsilon = np.arange(start=0.1, stop=0.9, step=0.1)
    epsilon = [float(round(i, 4)) for i in list(epsilon)]

    param_grid = [{'estimator__kernel': ['rbf'],
                   'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4, 'auto', 'scale'],
                   'estimator__C': [5, 10, 15, 25],
                   'estimator__epsilon': epsilon}]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10,
                        return_train_score=True, scoring=scorer, verbose=2)

    grid_result = grid.fit(x, y)

    print(f"Best: {abs(grid.best_score_)} using {grid_result.best_params_}")

    return grid.best_params_


def predict(model, x_ts, x_its, y_its):

    iloss = K.eval(euclidean_distance_loss(y_its, model.predict(x_its)))

    y_pred = model.predict(x_ts)

    return y_pred, iloss


def plot_learning_curve(model, x, y):

    # dictify model's parameters
    p = model.get_params()
    params = dict(kernel=p['estimator__kernel'], C=p['estimator__C'],
                  gamma=p['estimator__gamma'], eps=p['estimator__epsilon'])

    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(model, x, y, train_sizes=np.linspace(0.1, 1, 10),
                       n_jobs=-1, scoring=scorer, cv=10, verbose=2)

    plt.plot(train_sizes, np.mean(np.abs(train_scores_svr), axis=1))
    plt.plot(train_sizes, np.mean(np.abs(test_scores_svr), axis=1))
    plt.xlabel("Train size")
    plt.ylabel("Mean Euclidean Error (MEE)")
    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'SVR Learning curve \n {params}')

    name = ""
    for k, v in params.items():
        name += f"{k}{v}_"
    name += ".png"

    path = os.path.join(ROOT_DIR, "sklearnSVM", "plot", name)
    plt.savefig(path, dpi=600)
    plt.show()


def sklearn_svm():
    print("sklearn start")
    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    params = model_selection(x, y)

    # create model with best params found by GridSearch
    svr = SVR(kernel=params['estimator__kernel'], C=params['estimator__C'],
              gamma=params['estimator__gamma'], epsilon=params['estimator__epsilon'])

    model = MultiOutputRegressor(svr)

    # fit model on the entire TR
    x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, test_size=0.3)
    model.fit(x_tr, y_tr)

    tr_losses = euclidean_distance_loss(y_tr, model.predict(x_tr))
    val_losses = euclidean_distance_loss(y_vl, model.predict(x_vl))

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", np.mean(tr_losses))
    print("VL Loss: ", np.mean(val_losses))
    print("TS Loss: ", np.mean(ts_losses))

    print("sklearn end")

    # plot_learning_curve(model, x, y)
