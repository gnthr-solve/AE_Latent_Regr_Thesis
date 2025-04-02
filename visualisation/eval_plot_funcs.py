
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