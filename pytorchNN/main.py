import torch
import torch.nn.functional as F
import multiprocessing as mp

from utils import *
from torch import FloatTensor, tanh
from torch.optim import SGD
from torch.nn import Module, Linear, Parameter
from torch.nn.init import xavier_uniform
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold

ms_result = []
cv_result = []


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


def init_weights(m):
    if type(m) == Linear:
        xavier_uniform(m.weight)


def rmse(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def mean_euclidean_error(y_real, y_pred):
    return torch.mean(F.pairwise_distance(y_real, y_pred, p=2))


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        y_hat = model(x)
        # Computes loss
        loss = loss_fn(y, y_hat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def plot_learning_curve(losses, val_losses, start_epoch=1, savefig=False, **kwargs):
    plt.plot(range(start_epoch, kwargs['epochs']), losses[start_epoch:])
    plt.plot(range(start_epoch, kwargs['epochs']), val_losses[start_epoch:])

    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'PyTorch Learning Curve \n {kwargs}')

    if savefig:
        save_figure("pytorchNN", **kwargs)

    plt.show()


def fit(x, y, model, optimizer, loss_fn=mean_euclidean_error, epochs=200, batch_size=32, val_data=None):
    # Creates the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    val_losses = []

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

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

    # For each epoch...
    for epoch in range(epochs):
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            # ...Performs one train step and returns the corresponding loss
            loss = train_step(x_batch, y_batch)
            epoch_losses.append(loss)

        losses.append(np.mean(epoch_losses))

        epoch_val_losses = []
        # Perform validation
        with torch.no_grad():
            for x_val, y_val in val_loader:
                # Sets model to VALIDATION mode
                model.eval()

                # Makes predictions
                y_hat = model(x_val)
                # Computes loss
                val_loss = loss_fn(y_val, y_hat)
                epoch_val_losses.append(val_loss.item())

            val_losses.append(np.mean(epoch_val_losses))
    return losses, val_losses


def log_cv_result(result):
    loss_tr, loss_vl = result
    cv_result.append([loss_tr[-1], loss_vl[-1]])


def cross_validation(x, y, n_splits=2, epochs=200, batch_size=32, alpha=0.9, eta=0.001, lmb=0.0005):
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    cv_loss = []
    fold_idx = 1
    for tr_idx, vl_idx in kfold.split(x, y):
        print(f"Starting fold {fold_idx}")

        model = Net()
        optimizer = SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

        loss_tr, loss_vl = fit(x[tr_idx], y[tr_idx], model=model, optimizer=optimizer, epochs=epochs,
                               batch_size=batch_size, val_data=(x[vl_idx], y[vl_idx]))

        cv_loss.append([loss_tr[-1], loss_vl[-1]])
        print(f"Ended fold {fold_idx}, with {loss_tr[-1]} - {loss_vl[-1]}")
        fold_idx += 1

    params = dict(eta=eta, alpha=alpha, lmb=lmb, epochs=epochs, batch_size=batch_size)
    return params, cv_loss


def log_ms_result(result):
    # cv_res, params = result
    ms_result.append(result)


def model_selection(x, y):
    pool = mp.Pool(processes=mp.cpu_count())

    batch_size = [16, 32, 64]
    # eta = np.arange(start=0.0001, stop=0.1, step=0.0002)
    eta = [0.0005, 0.005, 0.05]
    alpha = np.arange(start=0.4, stop=1, step=0.2)
    lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)

    for e in eta:
        for a in alpha:
            for l in lmb:
                for b in batch_size:
                    pool.apply_async(cross_validation, (x, y), dict(eta=e, alpha=a, lmb=l, batch_size=b),
                                     callback=log_ms_result)

    pool.close()
    pool.join()

    # min_vl_loss = np.amin([np.mean(r, axis=0) for _, (_, r) in enumerate(ms_result)], axis=0)[1]

    min_loss = None
    min_idx = 0
    for idx, (d, r) in enumerate(ms_result):
        mean = (np.mean(r, axis=0))[1]
        if min_loss is None or min_loss > mean:
            min_loss = mean
            min_idx = idx

    best_params = ms_result[min_idx][0]

    print(f"Best score {min_loss} with {best_params}")
    return best_params


def predict(model, x_ts, x_its, y_its):
    x_ts = torch.from_numpy(x_ts).float()
    x_its = torch.from_numpy(x_its).float()
    y_its = torch.from_numpy(y_its).float()

    y_ipred = model(x_its)
    iloss = rmse(y_its, y_ipred)

    y_pred = model(x_ts)

    return y_pred.detach().numpy(), iloss.item()


def pytorch_nn(ms=False):
    print("pytorch start")
    # read training set
    x, y, x_its, y_its = read_tr(its=True)

    if ms:
        params = model_selection(x, y)
    else:
        # params = dict(eta=0.005, alpha=0.8, lmb=0.0006, epochs=200, batch_size=64)
        params = dict(eta=0.005, alpha=0.7, lmb=0.0002, epochs=160, batch_size=64)

    model = Net()
    optimizer = SGD(model.parameters(), lr=params['eta'], momentum=params['alpha'], weight_decay=params['lmb'])

    tr_losses, val_losses = fit(x, y, model=model, optimizer=optimizer,
                                batch_size=params['batch_size'], epochs=params['epochs'])

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    print("pytorch end")

    plot_learning_curve(tr_losses, val_losses, start_epoch=20, savefig=True, **params)


if __name__ == '__main__':
    pytorch_nn()
