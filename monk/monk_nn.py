from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from utils import *


# input encoding for monk's problems
def scaled(data):

    scaled_data = []
    for i in range(1, 7):
        col = data[:, i]

        # interpolate each column, from 0 to the number of unique items in that column...
        data[:, i] = np.interp(col, (col.min(), col.max()), (0, len(np.unique(col)) - 1))

        # ... so that we have the minimum number of bits when we 1-hot-encode data
        col_cat = to_categorical(data[:, i])

        # create numpy matrix
        scaled_data = np.concatenate((scaled_data, col_cat), axis=1) if i != 1 else col_cat

    return scaled_data


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

    return x, y, x_test, y_test


def save_monk_fig(monk_number, eta, alpha, lmb, plt_type='loss'):

    name = f"monk{monk_number}_{plt_type}_eta{eta}_alpha{alpha}_lmb{lmb if lmb else 0}.png"

    # create plot directory if it doesn't exist
    dir_path = os.path.join(ROOT_DIR, "monk", "plot")
    os.makedirs(dir_path, exist_ok=True)

    # save plot as figure
    fig_path = os.path.join(dir_path, name)
    plt.savefig(fig_path, dpi=600)


def monk_solver(monk_number, eta, alpha, epochs, lmb=None, batch_size=25, n_unit=4):

    # get data
    x, y, x_test, y_test = get_one_hot_encoded(monk_number)

    # create the model
    regularizer = l2(lmb) if lmb else None
    optimizer = SGD(learning_rate=eta, momentum=alpha)
    model = Sequential([
        Dense(n_unit, activation='tanh', kernel_regularizer=regularizer, input_dim=17),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[BinaryAccuracy(name='accuracy')])

    res = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

    # plot results for training set
    plt.plot(res.history['loss'])
    plt.plot(res.history['val_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(['Loss TR', 'Loss TS'], loc='center right')
    plt.title(f'MONK {monk_number} (eta = {eta}, alpha = {alpha}, lambda = {lmb}) - Loss')
    save_monk_fig(monk_number, eta, alpha, lmb)
    plt.show()

    # plot results for "test" (validation) set
    plt.plot(res.history['accuracy'])
    plt.plot(res.history['val_accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Accuracy TR', 'Accuracy TS'], loc='center right')
    plt.title(f'MONK {monk_number} (eta = {eta}, alpha = {alpha}, lambda = {lmb}) - Accuracy')
    save_monk_fig(monk_number, eta, alpha, lmb, plt_type='acc')
    plt.show()


if __name__ == '__main__':

    monk_solver(monk_number=1, eta=0.22, alpha=0.85, epochs=70)
    monk_solver(monk_number=2, eta=0.21, alpha=0.77, epochs=100)
    monk_solver(monk_number=3, eta=0.2, alpha=0.76, epochs=120)
    monk_solver(monk_number=3, eta=0.3, alpha=0.6, lmb=0.0001, epochs=200)
