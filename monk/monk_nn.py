import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from utils import *


def scaled(data):

    scaled_data = []
    for i in range(1, 7):
        col = data[:, i]

        data[:, i] = np.interp(col, (col.min(), col.max()), (0, len(np.unique(col)) - 1))

        col_cat = to_categorical(data[:, i])

        scaled_data = np.concatenate((scaled_data, col_cat), axis=1) if i != 1 else col_cat

    return scaled_data


def flattened(data):
    flat = []
    for d in data:
        flat.append(np.ravel(d))

    return np.array([np.array(f) for f in flat])


def get_one_hot_encoded(monk_number):

    train_file = "./dataset/monks-{}.train".format(monk_number)
    test_file = "./dataset/monks-{}.test".format(monk_number)

    # range up to 8 because there's a first blank space to be skipped
    train = loadtxt(train_file, delimiter=' ', usecols=range(1, 8))
    test = loadtxt(test_file, delimiter=' ', usecols=range(1, 8))

    # get target values
    y = train[:, 0]
    y_test = test[:, 0]

    # scale other values per column between 0 and # unique values for that column
    x = scaled(train)
    x_test = scaled(test)

    """# one-hot encode input
    x = to_categorical(train[:, 1:7])
    x_test = to_categorical(test[:, 1:7])

    # flatten the input
    x = flattened(x)
    x_test = flattened(x_test)"""

    return x, y, x_test, y_test


def monk_solver(monk_number, n_unit, eta, alpha, epochs, lmb=None, batch_size=None):

    # get data
    x, y, x_test, y_test = get_one_hot_encoded(monk_number)

    # create the model
    regularizer = l2(lmb) if lmb else None

    model = Sequential([
        Dense(n_unit, activation='tanh', kernel_regularizer=regularizer, input_dim=17),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[BinaryAccuracy(name='accuracy')])

    res = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

    # plot results for training set
    plt.plot(res.history['loss'])
    plt.plot(res.history['val_loss'])
    plt.legend(['Loss TR', 'Loss TS'], loc='center right')
    plt.title(f'MONK {monk_number} (eta = {eta}, alpha = {alpha}, lambda = {lmb}) - Loss')
    plt.show()

    # plot results for "test" (validation) set
    plt.plot(res.history['accuracy'])
    plt.plot(res.history['val_accuracy'])
    plt.legend(['Accuracy TR', 'Accuracy TS'], loc='center right')
    plt.title(f'MONK {monk_number} (eta = {eta}, alpha = {alpha}, lambda = {lmb}) - Accuracy')
    plt.show()

    # "don't specify batch_size if your data is in the form of datasets, generators,
    # or keras' sequences (since they generate batches)."

    # batch size sicuro <= 32, può spaziare da 1 (piccoli step poco costosi che portano
    # anche in direzioni errate, ma in media verso un local minima) a 32 (32 alla volta).
    # Semplicemente se hai 1000 batch samples e metti 500 a botta, in 2 iterazioni hai finito

    # Riguardo le epoch,ogni epoca migliora il modello "fino a 'na certa". Arrivi ad un
    # plateau e lo guardi tramite plot (x -> #epoche, y -> accuracy) [random value tipo 50]


if __name__ == '__main__':

    monk_solver(monk_number=1, n_unit=4, eta=0.25, alpha=0.85, epochs=90, batch_size=25)
    #monk_solver(monk_number=2, n_unit=4, eta=0.25, alpha=0.85, epochs=150)
    #monk_solver(monk_number=3, n_unit=4, eta=0.15, alpha=0.5, epochs=100)
    #monk_solver(monk_number=3, n_unit=4, eta=0.1, alpha=0.45, epochs=170, lmb=0.001)
