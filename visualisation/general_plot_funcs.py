
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re

from torch.nn import Module
from torch import Tensor

from pathlib import Path
from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_matrix import PlotMatrix, PlotMosaic
from .components.line_components import SingleTrajectoryPlot, TrajectoryPlotFill


"""
Plotting Functions - plot_loss_tensor uniform
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_agg_training_losses(
        training_losses: dict[str, Tensor],
        epochs: int,
        loss_names: dict[str, str] = None,
        title: str = None, 
        save_path: Path = None,
    ):

    ###--- Plotting Matrix ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}

    for n, (name, losses) in enumerate(training_losses.items(), 0):

        #loss_name = loss_names[name]
        n_iterations = len(losses)
        iterations = range(1, n_iterations + 1)

        n_iter_e = n_iterations // epochs
        separator_x = [n_iter_e * e for e in range(1, epochs + 1)]

        plot_dict[(n,0)] = SingleTrajectoryPlot(
            x_data = iterations,
            y_data = losses,
            separator_x = separator_x,
            x_label = 'Iterations',
            y_label = f'{name} Loss',
        )

    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 10, figsize = (14, 6))



"""
Plotting Functions - plot_loss_tensor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_loss_tensor(
        observed_losses: Tensor,
        epochs: int,
        loss_name: str = 'Loss',
        title: str = None, 
        save_path: Path = None,
    ):
    """
    Assumes shape (e, n_iter), where e is the number of epochs and n_iter the number of iterations per epoch.
    """
    epochs, n_iter_e = observed_losses.shape

    losses = observed_losses.flatten()
    
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    n_iterations = len(losses)
    iterations = range(1, n_iterations + 1)
    separator_x = [n_iter_e * e for e in range(1, epochs + 1)]

    plot_dict = {
        (0,0): SingleTrajectoryPlot(
            x_data = iterations,
            y_data = losses,
            separator_x = separator_x,
            x_label = 'Iterations',
            y_label = f'{loss_name} Loss',
        )
    }
    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 14, figsize = (14, 6))






"""
Plotting Functions - Plotting a 3D latent space with error or attributes
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_3Dlatent_with_error(latent_tensor: Tensor, loss_tensor: Tensor, title: str, save: bool = False):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(latent_tensor[:, 0], latent_tensor[:, 1], latent_tensor[:, 2], c = loss_tensor, cmap = 'RdYlGn_r')

    # Add colorbar
    colorbar = fig.colorbar(scatter)
    colorbar.set_label('Error')

    # Set plot labels and title
    ax.set_xlabel('$x_l$')
    ax.set_ylabel('$y_l$')
    ax.set_zlabel('$z_l$')
    plt.title(title)

    plt.show()

    if save:
        file_name = re.sub(r'\s+', '_', title).lower()
        plt.savefig(f"./results/figures/{file_name}.png", format='png')




def plot_3Dlatent_with_attribute(latent_tensor: Tensor, color_attr: Tensor | pd.Series, title: str, save: bool = False):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(latent_tensor[:, 0], latent_tensor[:, 1], latent_tensor[:, 2], c = color_attr, cmap = 'RdYlGn_r')

    # Add colorbar
    colorbar = fig.colorbar(scatter)
    colorbar.set_label('Reconstruction Error')

    # Set plot labels and title
    ax.set_xlabel('$x_l$')
    ax.set_ylabel('$y_l$')
    ax.set_zlabel('$z_l$')
    plt.title(title)

    plt.show()

    if save:
        file_name = re.sub(r'\s+', '_', title).lower()
        plt.savefig(f"./results/figures/{file_name}.png", format='png')