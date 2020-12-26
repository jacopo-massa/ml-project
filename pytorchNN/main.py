import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *
from torch import FloatTensor, tanh
from torch.optim import SGD
from torch.nn import Module, Linear, Parameter
from torch.nn.init import xavier_uniform
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold


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


def plot_learning_curve(losses, val_losses):
    plt.plot(losses)
    plt.plot(val_losses)

    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'PyTorch Learning Curve')
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


def cross_validation(x, y, n_splits=10, epochs=200, batch_size=32):

    eta = 0.001
    alpha = 0.9
    lmb = 0.0006
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    cv_loss = []
    fold_idx = 1
    for tr_idx, vl_idx in kfold.split(x, y):
        print(f"Starting fold {fold_idx}")

        model = Net()
        optimizer = SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

        loss_tr, loss_vl = fit(x[tr_idx], y[tr_idx], model=model, optimizer=optimizer, epochs=epochs,
                               batch_size=batch_size, val_data=(x[vl_idx], y[vl_idx]))

        cv_loss.append({f"Fold {fold_idx}": [loss_tr[-1], loss_vl[-1]]})
        print(f"Ended fold {fold_idx}, with {loss_tr[-1]} - {loss_vl[-1]}")
        fold_idx += 1

    return cv_loss


if __name__ == '__main__':
    # read training set
    x, y, x_t, y_t = read_tr(its=True)

    """tr_losses, val_losses = fit(x, y, model=model, optimizer=optimizer, batch_size=batch_size)

    print(len(tr_losses), len(val_losses))
    print("TR Loss: ", np.mean(tr_losses))
    print("VL Loss: ", np.mean(val_losses))

    plot_learning_curve(tr_losses, val_losses)"""

    cross_validation(x, y)

