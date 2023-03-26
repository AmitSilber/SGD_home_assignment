import utils
from Network import Network
import numpy as np
from functools import partial
from utils import generate_data, plot_predictions, plot_loss, plot_average_weight, plot_data


def q1(a, x):
    return np.sin(x) * np.cos(a * x)


def q2(a, x):
    return np.sin(a[0] * x[0]) * np.cos(a[1] * x[1])


def run_experiment(data, layers, activation, batch_size, epochs, sgd_params):
    net = Network(layers, activation)
    stats = net.SGD(epochs=epochs, parameters=sgd_params, train=data["train"],
                    batch_size=batch_size)
    return {"net": net, "stats": stats}


def make_plots(nets_dict, input_dim, num_of_predictions, experiment_name, epochs):
    plot_predictions(nets_dict, x_dim=input_dim, num_of_samples=num_of_predictions,
                     file_name=f"{experiment_name} predictions.pdf")

    plot_loss(range(1, epochs + 1), nets_dict, file_name=f"{experiment_name} loss_plots.pdf")

    plot_average_weight(range(1, epochs + 1), nets_dict, rows=2, cols=1,
                        file_name=f"{experiment_name} avg_weight_plot.pdf")


def multiple_experiments(f, f_parameters, input_dim, num_of_samples, num_of_predictions, num_of_total_steps,
                         experiments_parameters, name):
    f = partial(f, f_parameters)
    data = generate_data(f, x_dim=input_dim, num_of_samples=num_of_samples)
    plot_data(data["train"], input_dim=input_dim, name=f"{name} generated_data.pdf")
    nets_dict = {name: run_experiment(data,
                                      layers=experiments_parameters[name]["layers"],
                                      activation=experiments_parameters[name]["activation"],
                                      batch_size=experiments_parameters[name]["batch_size"],
                                      epochs=experiments_parameters[name]["epochs"],
                                      sgd_params=experiments_parameters[name]["sgd_params"]
                                      ) for name in experiments_parameters.keys()
                 }
    make_plots(nets_dict, input_dim, num_of_predictions, name, num_of_total_steps)


if __name__ == '__main__':
    file_name = "tanh, hidden layer=50_a=2,"
    multiple_experiments(f=q1,
                         f_parameters=2,
                         input_dim=1,
                         num_of_samples=utils.num_of_samples,
                         num_of_predictions=utils.num_of_predictions,
                         num_of_total_steps=utils.num_of_total_steps,
                         experiments_parameters=utils.vanilla_vs_momentum_experiment,
                         name=file_name

                         )
