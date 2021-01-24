import torch
import torch.nn.functional as F
import multiprocessing as mp

from utils import *
from torch import FloatTensor, tanh
from torch.optim import SGD
from torch.nn import Module, Linear, Parameter
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold

ms_result = []


class Net(Module):

    def __init__(self, n_units=30, in_features=10, out_features=2):
        super(Net, self).__init__()

        # set up model parameters
        self.n_units = Parameter(FloatTensor(n_units), requires_grad=False)
        self.in_features = Parameter(FloatTensor(in_features), requires_grad=False)
        self.out_features = Parameter(FloatTensor(out_features), requires_grad=False)

        # input layer
        self.l_in = Linear(in_features=in_features, out_features=n_units)

        # hidden layers
        self.l2 = Linear(in_features=n_units, out_features=n_units)
        self.l3 = Linear(in_features=n_units, out_features=n_units)

        # output layer
        self.l_out = Linear(in_features=n_units, out_features=out_features)

    def forward(self, x):
        # input layer
        x = tanh(self.l_in(x))

        # hidden layers
        x = tanh(self.l2(x))
        x = tanh(self.l3(x))

        # output layer
        x = self.l_out(x)

        return x


# weights initialization function, as Glorot initialization (also called Xavier initialization)
def init_weights(m):
    if type(m) == Linear:
        xavier_normal_(m.weight)


# re-written loss function for PyTorch (with tensors)
def mean_euclidean_error(y_real, y_pred):
    return torch.mean(F.pairwise_distance(y_real, y_pred, p=2))


def make_train_step(model, loss_fn, optimizer):

    # Builds function that performs a step in the train loop
    def train_step(x, y):

        # set model to TRAIN mode
        model.train()
        # make predictions
        y_hat = model(x)
        # compute loss
        loss = loss_fn(y, y_hat)
        # compute gradients
        loss.backward()
        # update parameter and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # return the loss
        return loss.item()

    # return the function that will be called inside the train loop
    return train_step


def plot_learning_curve(losses, val_losses, start_epoch=1, savefig=False, **kwargs):
    plt.plot(range(start_epoch, kwargs['epochs']), losses[start_epoch:])
    plt.plot(range(start_epoch, kwargs['epochs']), val_losses[start_epoch:])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'PyTorch Learning Curve \n {kwargs}')

    if savefig:
        save_figure("pytorchNN", **kwargs)

    plt.show()


def fit(x, y, model, optimizer, loss_fn=mean_euclidean_error, epochs=200, batch_size=64, val_data=None):

    # create the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    val_losses = []

    # change our data into tensors to work with PyTorch
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    # if validation data are given, use them. Else split the given development set
    if val_data:
        x_val, y_val = val_data
        x_val_tensor = torch.from_numpy(x_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()

        val_data = TensorDataset(x_val_tensor, y_val_tensor)
        train_data = dataset

    else:
        # split dataset into train set and validation set (70% - 30%)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size

        train_data, val_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    # for each epoch...
    for epoch in range(epochs):
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            # ...perform one train step and return the corresponding loss
            loss = train_step(x_batch, y_batch)
            epoch_losses.append(loss)

        losses.append(np.mean(epoch_losses))

        epoch_val_losses = []
        # perform validation
        with torch.no_grad():
            for x_val, y_val in val_loader:
                # set model to VALIDATION mode
                model.eval()
                # makes predictions
                y_hat = model(x_val)
                # computes loss
                val_loss = loss_fn(y_val, y_hat)
                epoch_val_losses.append(val_loss.item())

            val_losses.append(np.mean(epoch_val_losses))
    return losses, val_losses


def cross_validation(x, y, n_splits=10, epochs=200, batch_size=64, eta=0.003, alpha=0.85, lmb=0.0002):
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    cv_loss = []
    fit_times = []
    fold_idx = 1

    # for each fold (whose number is defined by n_splits) ...
    for tr_idx, vl_idx in kfold.split(x, y):

        # ... create and fit a different model ...
        model = Net()
        model.apply(init_weights)
        optimizer = SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

        fit_time = time.time()
        loss_tr, loss_vl = fit(x[tr_idx], y[tr_idx], model=model, optimizer=optimizer, epochs=epochs,
                               batch_size=batch_size, val_data=(x[vl_idx], y[vl_idx]))

        fit_time = time.time() - fit_time
        fit_times.append(fit_time)

        # ... and save results ...
        cv_loss.append([loss_tr[-1], loss_vl[-1]])
        fold_idx += 1

    params = dict(eta=eta, alpha=alpha, lmb=lmb, epochs=epochs, batch_size=batch_size)

    # calculate average time to make the entire cross validation process
    mean_fit_time = np.mean(fit_times)
    return params, cv_loss, mean_fit_time


# callback function for the multiprocessing task
def log_ms_result(result):
    ms_result.append(result)


def model_selection(x, y):
    # define a pool of tasks, for multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())

    # define the grid search parameters
    eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.2)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0002)
    lmb = [float(round(i, 4)) for i in list(lmb)]

    batch_size = [16, 32, 64]

    gs_size = len(batch_size) * len(eta) * len(alpha) * len(lmb)

    ms_time = time.time()
    print(f"Starting Grid Search, for a total of {gs_size} fits.")

    for e in eta:
        for a in alpha:
            for l in lmb:
                for b in batch_size:
                    pool.apply_async(cross_validation, (x, y), dict(eta=e, alpha=a, lmb=l, batch_size=b),
                                     callback=log_ms_result)

    pool.close()
    pool.join()

    print("\nEnded Grid Search. ({:.4f})\n".format(time.time() - ms_time))

    # print model selection results
    sorted_res = sorted(ms_result, key=lambda tup: (np.mean(tup[1], axis=0))[1])
    for (p, l, t) in sorted_res:
        scores = np.mean(l, axis=0)
        print("{} \t TR {:.4f} \t TS {:.4f} (Fit Time: {:.4f})".format(p, scores[0], scores[1], t))

    min_loss = (np.mean(sorted_res[0][1], axis=0))[1]
    best_params = sorted_res[0][0]

    print("\nBest score {:.4f} with {}\n".format(min_loss, best_params))
    return best_params


def predict(model, x_ts, x_its, y_its):
    # change our data into tensors to work with PyTorch
    x_ts = torch.from_numpy(x_ts).float()
    x_its = torch.from_numpy(x_its).float()
    y_its = torch.from_numpy(y_its).float()

    # predict on internal test set
    y_ipred = model(x_its)
    iloss = mean_euclidean_error(y_its, y_ipred)

    # predict on blind test set
    y_pred = model(x_ts)

    # return predicted target on blind test set,
    # and losses on internal test set
    return y_pred.detach().numpy(), iloss.item()


def pytorch_nn(ms=False):
    print("pytorch start\n")

    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    # choose model selection or hand-given parameters
    if ms:
        params = model_selection(x, y)
    else:
        params = dict(eta=0.003, alpha=0.85, lmb=0.0002, epochs=80, batch_size=64)

    # create and fit the model
    model = Net()
    model.apply(init_weights)
    optimizer = SGD(model.parameters(), lr=params['eta'], momentum=params['alpha'], weight_decay=params['lmb'])

    tr_losses, val_losses = fit(x, y, model=model, optimizer=optimizer,
                                batch_size=params['batch_size'], epochs=params['epochs'])

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    print("\npytorch end")

    plot_learning_curve(tr_losses, val_losses, savefig=True, **params)

    # generate csv file for MLCUP
    write_blind_results(y_pred)


if __name__ == '__main__':
    pytorch_nn()
