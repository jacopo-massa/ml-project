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
                   'estimator__C': [1, 10, 100, 1000],
                   'estimator__epsilon': epsilon}]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10,
                        return_train_score=True, scoring=scorer, verbose=2, refit=True)

    grid_result = grid.fit(x, y)

    print(f"Best: {grid.best_score_} using {grid_result.best_params_}")

    return grid.best_estimator_, grid_result


def predict(model, x_ts, x_its, y_its):
    y_ipred = model.predict(x_its)
    iloss = K.eval(euclidean_distance_loss(y_its, y_ipred))

    y_pred = model.predict(x_ts)

    return y_pred, iloss


def plot_learning_curve(model, x, y):

    # dictify model's parameters
    p = model.get_params()
    params = dict(C=p['estimator__C'], gamma=p['estimator__gamma'], eps=p['estimator__epsilon'])

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


if __name__ == '__main__':
    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    model, _ = model_selection(x, y)

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TS Loss: ", np.mean(ts_losses))
