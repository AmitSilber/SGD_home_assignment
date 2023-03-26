import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

a = 2
width = np.pi


def ReLU(x):
    return np.where(x > 0, x, 0)


def d_ReLU(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1 - tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def swish(x):
    return x * sigmoid(x)


def d_swish(x):
    return sigmoid(x) + x * d_sigmoid(x)


def snake(x):
    return x + np.power(np.sin(a * x), 2) / a


def d_snake(x):
    return 1 + 2 * np.cos(a * x) * np.sin(a * x)


relu_kit = {"func": ReLU, "d_func": d_ReLU}
tanh_kit = {"func": tanh, "d_func": d_tanh}
sigmoid_kit = {"func": sigmoid, "d_func": d_sigmoid}
swish_kit = {"func": swish, "d_func": d_swish}
snake_kit = {"func": snake, "d_func": d_snake}

#####################
# experiment dict format: {
#                         name: experiment_params: {
# "layers" : num of layers
# "activation" : {
#                                                                    "func": activation function
#                                                                    "d_func": derivative
#                                                                    }
# "batch_size" : <int>
# "sgd_params" : {
#                                                                    "lr": learning rate
#                                                                    "momentum": <float> (optional)
#                                                                    }

num_of_samples = 1000
num_of_predictions = 1000
num_of_total_steps = 1000

vanilla_sgd_tanh = {"layers": [1, 50, 1],
                    "activation": tanh_kit,
                    "batch_size": num_of_samples,
                    "epochs": num_of_total_steps,
                    "sgd_params": {"lr": 0.008}}
momentum_sgd_tanh = {"layers": [1, 50, 1],
                     "activation": tanh_kit,
                     "batch_size": num_of_samples,
                     "epochs": num_of_total_steps,
                     "sgd_params": {"lr": 0.008, "momentum": 0.8}}

vanilla_vs_momentum_experiment = {"vanilla sgd": vanilla_sgd_tanh,
                                  "momentum sgd": momentum_sgd_tanh
                                  }


def L2Loss(out, y):
    return 0.5 * np.linalg.norm(out - y) ** 2


def d_L2Loss(out, y):
    return out - y


def generate_data(f, x_dim, num_of_samples):
    """

    :param f: function to evaluate samples on
    :param x_dim: input dimension
    :param num_of_samples: number of samples
    :return: data set with labels
    """
    x = np.random.uniform(-width, width, size=(num_of_samples, x_dim))
    f_x = np.apply_along_axis(f, 1, x)
    data = [(x[i, :], f_x[i]) for i in range(num_of_samples)]
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
    plt.close()


def plot_loss(epochs, train_losses, file_name):
    fig = plt.figure()
    ax = fig.add_subplot()
    for name in train_losses.keys():
        ax.plot(epochs, train_losses[name]["stats"]["train_loss"], label=f"{name} training loss")
        ax.legend()
    plt.savefig(file_name)
    plt.close()


def plot_average_weight(epochs, weights, rows, cols, file_name):
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            for activation in weights:
                ws = [w[cols * row + col][0] for w in weights[activation]["stats"]["avg_weight"]]
                axs[row].plot(epochs, ws,
                              label=f"{activation} network weights for layer {row + 1}")
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def plot_predictions(nn, x_dim, num_of_samples, file_name):
    xs = np.random.uniform(-width, width, size=(num_of_samples, x_dim))
    for name in nn.keys():
        preds = nn[name]["net"].predict(xs)
        plot_data([(xs[i, :], preds[i]) for i in range(num_of_samples)], x_dim, f"{name} {file_name}")
