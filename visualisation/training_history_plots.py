
import torch
import pandas as pd
import numpy as np
import re

from torch import Tensor

from pathlib import Path
from itertools import product
from functools import wraps
from typing import Callable

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from observers.observations_converter import TrainingObsConverter

from .plot_matrix import PlotMatrix, PlotMosaic
from .components.line_components import SingleTrajectoryPlot, TrajectoryPlotFill
from .components.scatter_components import MultiScatterPlot

"""
Plotting Functions - Plot iteration loss development for (multiple) loss(es)
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_agg_training_losses(
        observed_losses: dict[str, Tensor],
        epochs: int = None,
        title: str = None, 
        save_path: Path = None,
    ):
    """
    Plots training iteration loss values for multiple losses.

    Args:
        training_losses: dict[str, Tensor]
            Keys represent loss names, values are loss tensors
                i) flattened of shape (epochs * iterations,)
                ii) stacked (epochs, iterations)
            if epochs is None, ii) is assumed.
        epochs: int
            Number of epochs in training process.
        title: str
            Title of the plot.
        save_path: Path
            Optional Path to save figure to, if None figure is just shown, not saved.
    """

    ###--- Plotting Matrix ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}

    for n, (name, losses) in enumerate(observed_losses.items(), 0):

        #loss_name = loss_names[name]
        total_iterations = len(losses)
        iterations = range(1, total_iterations + 1)

        if epochs is None:
            epochs, n_iter_e = losses.shape
            losses = losses.flatten()
        else:
            n_iter_e = total_iterations // epochs

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
Plotting Functions - Plot iteration loss development for (multiple) loss(es)
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_2Dlatent_by_epoch(latent_observations: dict[int,Tensor], title: str = None, save_path: Path = None):
    """
    Args:
        latent_observations: dict[int, Tensor]
            Assumes batched tensor values of shape (b, z_1, z_2)
    """
    

    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    latent_scatter = MultiScatterPlot(
        tensor_data = latent_observations,
        x_label = '$z_1$',
        y_label = '$z_2$',
        cmap = 'viridis',
    )

    plot_matrix.add_plot_dict({
        (0,0): latent_scatter,
    })

    plot_matrix.draw(fontsize = 14, figsize = (10, 8))




"""
Plotting Functions - Distribution parameter history
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_dist_params(dist_params_tensor: Tensor, functional = torch.max):
    """
    Expects dist_params_tensor of shape (n_epochs, dataset_size, latent_dim, n_dist_params)
    """
    dist_params = dist_params_tensor.unbind(dim = -1)

    n_params = len(dist_params)
    n_rows = int(n_params**0.5)  # Calculate the number of rows for the plot matrix
    n_cols = n_params // n_rows + (n_params % n_rows > 0)  # Calculate the number of columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    fig.suptitle(f"Distribution Parameter Development", fontsize=16)

    for idx, param_tensor in enumerate(dist_params):

        ax = axes.flatten()[idx] if n_params > 1 else axes  # Handle single parameter case

        n_iter_total = param_tensor.shape[0] * param_tensor.shape[1]
        flattened_param_tensor = param_tensor.flatten(start_dim = 0, end_dim = 1)

        param_values = torch.tensor([
            functional(param).item() 
            for param in flattened_param_tensor
        ])

        iterations = len(param_values)
        ax.plot(range(iterations), param_values)

        # Add vertical lines for each epoch
        epochs = param_tensor.shape[0]
        iterations_per_epoch = param_tensor.shape[1]
        for epoch in range(1, epochs):
            ax.axvline(x = epoch * iterations_per_epoch, color = 'r', linestyle = '--')

        ax.set_title(f'Params {idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Functional Value')

    # Hide any unused subplots
    if n_params > 1:
        for idx in range(n_params, n_rows * n_cols):
            axes.flatten()[idx].axis('off')

    plt.tight_layout()
    plt.show()




def plot_dist_params_batch(dist_params_tensor: Tensor, batch_size: int, functional: Callable[[Tensor], Tensor] = torch.max):
    """
    Expects dist_params_tensor of shape (n_epochs, dataset_size, latent_dim, n_dist_params)
    """
    dist_params = dist_params_tensor.unbind(dim = -1)

    n_params = len(dist_params)
    n_rows = int(n_params**0.5)  # Calculate the number of rows for the plot matrix
    n_cols = n_params // n_rows + (n_params % n_rows > 0)  # Calculate the number of columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    fig.suptitle(f"Distribution Parameter Development", fontsize=16)

    for idx, param_tensor in enumerate(dist_params):

        ax: Axes = axes.flatten()[idx] if n_params > 1 else axes  # Handle single parameter case

        n_epochs = param_tensor.shape[0]
        size_dataset = param_tensor.shape[1]
        max_batch_idx = n_epochs * size_dataset // batch_size

        flattened_param_tensor = param_tensor.flatten(start_dim = 0, end_dim = 1)
        
        sample_param_values = torch.tensor([
            functional(param).item() 
            for param in flattened_param_tensor
        ])

        batched_values = [
            sample_param_values[i*batch_size : (i+1)*batch_size] 
            for i in range(max_batch_idx)
        ]
        batched_values.append(sample_param_values[max_batch_idx * batch_size: -1])

        batch_values_mean = torch.tensor([batch_values.mean() for batch_values in batched_values])
        batch_values_std = torch.tensor([batch_values.std() for batch_values in batched_values])

        total_iterations = len(batched_values)
        ax.plot(range(total_iterations), batch_values_mean)

        # Add vertical lines for each epoch
        epochs = n_epochs
        iterations_per_epoch = len(batched_values) / epochs
        for epoch in range(1, epochs):
            ax.axvline(x = epoch * iterations_per_epoch, color = 'r', linestyle = '--')

        ax.fill_between(
            range(total_iterations), 
            batch_values_mean - batch_values_std, 
            batch_values_mean + batch_values_std, 
            alpha=0.2, 
            color='gray',
        )

        ax.set_title(f'Params {idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Functional Value')

    # Hide any unused subplots
    if n_params > 1:
        for idx in range(n_params, n_rows * n_cols):
            axes.flatten()[idx].axis('off')

    plt.tight_layout()
    plt.show()