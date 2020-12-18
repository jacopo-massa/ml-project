import matplotlib.pyplot as plt
import numpy as np


from numpy import loadtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryAccuracy


# One Hot Encoding through Keras
def get_one_hot_encoded(monk_number):
    # range up to 8 because there's a first blank space to be skipped

    train_file = "./dataset/monks-{}.train".format(monk_number)
    test_file = "./dataset/monks-{}.test".format(monk_number)

    train = loadtxt(train_file, delimiter=' ', usecols=range(1, 8))
    test = loadtxt(test_file, delimiter=' ', usecols=range(1, 8))

    x = to_categorical(train[:, 1:7])
    x_test = to_categorical(test[:, 1:7])

    flat_x = []
    for i in range(len(x)):
        flat_x.append(np.ravel(x[i]))

    x = np.array([np.array(xi) for xi in flat_x])

    flat_x_test = []
    for i in range(len(x_test)):
        flat_x_test.append(np.ravel(x_test[i]))

    x_test = np.array([np.array(xi) for xi in flat_x_test])

    y = train[:, 0]
    y_test = test[:, 0]

    return x, y, x_test, y_test


def monk_solver(monk_number, n_unit, eta, alpha, epochs, batch_size=None):

    # get data
    x, y, x_test, y_test = get_one_hot_encoded(monk_number)

    # create the model
    model = Sequential([
        Dense(n_unit, activation='tanh', input_dim=30),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[BinaryAccuracy(name='accuracy')])

    res = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

    plt.plot(res.history['loss'])
    plt.plot(res.history['accuracy'])
    plt.legend(['Loss', 'Accuracy'])
    plt.show()

    # "don't specify batch_size if your data is in the form of datasets, generators,
    # or keras' sequences (since they generate batches)."

    # batch size sicuro <= 32, può spaziare da 1 (piccoli step poco costosi che portano
    # anche in direzioni errate, ma in media verso un local minima) a 32 (32 alla volta).
    # Semplicemente se hai 1000 batch samples e metti 500 a botta, in 2 iterazioni hai finito

    # Riguardo le epoch,ogni epoca migliora il modello "fino a 'na certa". Arrivi ad un
    # plateau e lo guardi tramite plot (x -> #epoche, y -> accuracy) [random value tipo 50]


monk_solver(monk_number=1, n_unit=4, eta=0.25, alpha=0.85, epochs=100)

# get_one_hot_encoded(1)

