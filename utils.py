import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def ReLU(x):
    return np.where(x > 0, x, 0)


def d_ReLU(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1 - tanh(x) ** 2


def L2Loss(out, y):
    return 0.5 * np.linalg.norm(out - y) ** 2


def d_L2Loss(out, y):
    return out - y


def generate_data(f, x_dim, num_of_samples, test=False):
    """

    :param f: function to evaluate samples on
    :param x_dim: input dimension
    :param num_of_samples: number of samples
    :param test: train test split
    :return: data set with labels
    """
    x = np.random.uniform(-np.pi, np.pi, size=(num_of_samples, x_dim))
    f_x = np.apply_along_axis(f, 1, x)
    data = [(x[i, :], f_x[i]) for i in range(num_of_samples)]
    if test:
        test_split = num_of_samples // 10
        return {"train": data[:-test_split], "test": data[-test_split:]}
    return {"train": data}


def plot_data(data, input_dim, name):
    x = np.array([data_point[0][0] for data_point in data])
    f = np.array([data_point[1] for data_point in data])
    if input_dim > 1:
        y = np.array([data_point[0][1] for data_point in data])
        if f.shape != x.shape:
            f = f.squeeze(axis=-1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x, y, f, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f')
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(x, f)
        ax.set_xlabel('x')
        ax.set_ylabel('f')
    plt.savefig(name)


def plot_loss(epochs, train_losses, test_losses, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, test_losses, label="Test loss")
    ax.legend()
    plt.savefig(name)


def plot_average_weight(epochs, weights, rows, cols, name):
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            ws = [w[cols * row + col][0] for w in weights]
            bs = [w[cols * row + col][1] for w in weights]
            axs[row].plot(epochs, ws,
                          label="weight")
            axs[row].plot(epochs, bs,
                          label="bias")
    plt.legend()
    plt.savefig(name)


def plot_predictions(nn, x_dim, num_of_samples, name):
    xs = np.random.uniform(-np.pi, np.pi, size=(num_of_samples, x_dim))
    preds = nn.predict(xs)
    plot_data([(xs[i, :], preds[i]) for i in range(num_of_samples)], x_dim, name)
