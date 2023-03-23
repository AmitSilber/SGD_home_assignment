import utils
from Network import Network
import numpy as np
from functools import partial
from utils import generate_data, plot_predictions, plot_loss, plot_average_weight, plot_data, activation_func_dict


def q1(a, x):
    return np.sin(x) * np.cos(a * x)


def q2(a, x):
    return np.sin(a[0] * x[0]) * np.cos(a[1] * x[1])


def run_experiment(experiment_name, f, f_parameters, layers, fine_tuning_params, num_of_samples, epochs, lr,
                   num_of_predictions):
    f = partial(f, f_parameters)
    data = generate_data(f, x_dim=layers[0], num_of_samples=num_of_samples)
    plot_data(data["train"], input_dim=layers[0], name=experiment_name + "_generated_data.pdf")

    nets_dict = dict()
    for net_name in fine_tuning_params.keys():
        net = Network(layers, fine_tuning_params[net_name]["activation"])
        stats = net.SGD(epochs=epochs, lr=lr, train=data["train"],
                        batch_size=fine_tuning_params[net_name]["batch_size"])
        nets_dict[net_name] = {"net": net, "stats": stats}

    make_plots(nets_dict, layers[0], num_of_predictions, experiment_name, epochs)


def make_plots(nets_dict, input_dim, num_of_predictions, experiment_name, epochs):
    plot_predictions(nets_dict, x_dim=input_dim, num_of_samples=num_of_predictions,
                     name=experiment_name + "_predictions.pdf")

    plot_loss(range(1, epochs + 1), nets_dict, name=experiment_name + "_loss_plots.pdf")

    plot_average_weight(range(1, epochs + 1), nets_dict, rows=2, cols=1,
                        name=experiment_name + "_avg_weight_plot.pdf")


if __name__ == '__main__':
    file_name = "tanh_l=100_a=2"
    run_experiment(experiment_name=file_name,
                   f=q1,
                   f_parameters=2,
                   layers=[1, 100, 1],
                   fine_tuning_params=utils.tanh_one_batch_dict,
                   num_of_samples=10000,
                   epochs=200,
                   lr=0.008,
                   num_of_predictions=10000,
                   )
