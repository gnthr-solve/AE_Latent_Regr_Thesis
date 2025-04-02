
import torch
import pandas as pd
import numpy as np
import re

from torch import Tensor

from pathlib import Path
from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from helper_tools.transform_observations import transform_to_epoch_sample

from .plot_matrix import PlotMatrix, PlotMosaic
from .components.line_components import SingleTrajectoryPlot, TrajectoryPlotFill


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
def plot_2Dlatent_by_epoch(latent_observations: Tensor, n_epochs: int, title: str = None, save_path: Path = None):
    """
    Args:
        latent_observations: Tensor
            Assumed shape (n_epochs * n_samples, z1, z2)
    """
    

    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    dim_red_scatter = None

    plot_matrix.add_plot_dict({
        (0,0): dim_red_scatter,
    })

    plot_matrix.draw(fontsize = 14, figsize = (10, 8))