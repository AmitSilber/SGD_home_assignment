from Network import Network
import numpy as np
from functools import partial
from utils import generate_data, plot_predictions, plot_loss, plot_average_weight, plot_data


def q1(a, x):
    return np.sin(x) * np.cos(a * x)


def q2(a, x):
    return np.sin(a[0] * x[0]) * np.cos(a[1] * x[1])


def run_experiment(experiment_name, f, f_parameters, layers, num_of_samples=10000, epochs=100, lr=0.001, batch_size=90,
                   num_of_predictions=0,
                   test=False):
    f = partial(f, f_parameters)
    data = generate_data(f, x_dim=layers[0], num_of_samples=num_of_samples, test=test)
    net = Network(layers)
    plot_data(data["train"], input_dim=layers[0], name=experiment_name + "_generated_data.pdf")

    stats = net.SGD(epochs=epochs, lr=lr, batch_size=batch_size, train=data["train"], test=data["test"])
    plot_predictions(net, x_dim=layers[0], num_of_samples=num_of_predictions, name=experiment_name + "_predictions.pdf")
    plot_loss(range(1, epochs + 1), stats["train_loss"], stats["test_loss"], name=experiment_name + "_loss_plots.pdf")
    plot_average_weight(range(1, epochs + 1), stats["avg_weight"], rows=2, cols=1,
                        name=experiment_name + "_avg_weight_plot.pdf")


if __name__ == '__main__':
    run_experiment(experiment_name="1d_a=2_n=100_lr=0.001", f=q1, f_parameters=2, layers=[1,100, 1],
                   num_of_predictions=10000,
                   test=True)
