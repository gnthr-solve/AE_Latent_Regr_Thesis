import torch
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from torch import Tensor
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Callable

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MaxNLocator, MultipleLocator, AutoMinorLocator

from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from ..axes_components import AxesComponent



class DFScatterPlot(AxesComponent):
    def __init__(
            self, 
            df: pd.DataFrame, 
            x_col: str, 
            y_col: str,
            x_label: str = None, 
            y_label: str = None,
            color: str = 'blue',
            title: str = None,
        ):
        """
        Initializes the SimpleScatterPlot.

        Args:
            df (pd.DataFrame): The aggregated dataset DataFrame.
            x_col (str): The column name for the X-axis.
            y_col (str): The column name for the Y-axis.
            title (str, optional): The title of the plot. Defaults to None.
        """
        self.df = df
        self.x_col = x_col
        self.y_col = y_col

        self.x_label = x_label if x_label else x_col
        self.y_label = y_label if y_label else y_col

        self.color = color

        self.title = title# if title else f'{y_col} vs {x_col}'


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        """
        Draws the simple scatterplot on the given Axes.

        Args:
            ax (plt.Axes): The matplotlib Axes to draw on.
            fontsize (int, optional): Font size for labels and title. Defaults to 10.
        """
        # Plot scatter points
        ax.scatter(self.df[self.x_col], self.df[self.y_col], alpha=0.7, c = self.color, edgecolors='w', s=50)
        
        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)

        if self.title:
            ax.set_title(self.title, fontsize=fontsize + 2)
        
        # Enable grid for better readability
        ax.grid(True)




class ScatterPlot(AxesComponent):
    def __init__(
            self, 
            x_data: Any, 
            y_data: Iterable, 
            x_label: str = None, 
            y_label: str = None, 
            color: str = 'blue',
            title: str = None,
        ):

        """
        Initializes the ColumnPairScatterPlot.

        Args:
            x_data (Any): The data for the X-axis (e.g., DataFrame column, list, numpy array).
            y_data (Iterable): The data for the Y-axis. Must be an iterable of the same length as x_data.
            x_label (str, optional): Label for the X-axis. Defaults to None.
            y_label (str, optional): Label for the Y-axis. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
        """
        self.x_data = x_data
        self.y_data = y_data

        self.x_label = x_label or 'X-axis'
        self.y_label = y_label or 'Y-axis'

        self.color = color
        
        self.title = title if title else 'Scatter Plot'


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        """
        Draws the scatter plot on the given Axes.

        Args:
            ax (plt.Axes): The matplotlib Axes to draw on.
            fontsize (int, optional): Font size for labels and title. Defaults to 10.
        """
        # Plot scatter points
        ax.scatter(self.x_data, self.y_data, alpha=0.7, c = self.color, edgecolors='w', s=50, color='blue')

        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
        
        ax.set_title(self.title, fontsize=fontsize + 2)

        # Enable grid for better readability
        ax.grid(True)




class ColoredScatterPlot(AxesComponent):
    def __init__(self, 
                 x_data: Iterable, 
                 y_data: Iterable, 
                 color_data: Iterable, 
                 x_label: str = None,
                 y_label: str = None,
                 color_label: str = None,
                 x_scale: str = 'linear',
                 y_scale: str = 'linear',
                 title: str = None,
                 cmap: str = 'viridis'):
        """
        Initializes the ColoredScatterPlot.

        Args:
            x_data (Any): The data for the X-axis.
            y_data (Iterable): The data for the Y-axis.
            color_data (Iterable): The data used for color-coding the points.
            x_label (str, optional): Label for the X-axis. Defaults to None.
            y_label (str, optional): Label for the Y-axis. Defaults to None.
            color_label (str, optional): Label for the color bar. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.color_data = color_data
        self.x_label = x_label if x_label else 'x'
        self.y_label = y_label if y_label else 'y'
        self.color_label = color_label

        self.x_scale = x_scale
        self.y_scale = y_scale

        self.title = title
        self.cmap = cmap


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        """
        Draws the colored scatter plot on the given Axes.

        Args:
            ax (plt.Axes): The matplotlib Axes to draw on.
            fontsize (int, optional): Font size for labels and title. Defaults to 10.
        """
        # Create scatter plot with color mapping
        scatter = ax.scatter(
            self.x_data, 
            self.y_data, 
            c=self.color_data, 
            cmap=self.cmap,
            alpha=0.7, 
            edgecolors='w', 
            s=50
        )

        # Add colorbar with label
        cbar = plt.colorbar(scatter, ax=ax)
        if self.color_label:
            cbar.set_label(self.color_label, fontsize=fontsize)

        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
        
        ax.set_xscale(self.x_scale)
        ax.set_yscale(self.y_scale)

        ax.set_title(self.title, fontsize=fontsize + 2)

        # Enable grid for better readability
        ax.grid(True)




        
class LatentSpaceScatterPlot(AxesComponent):

    def __init__(
            self, 
            latent_tensor: Tensor, 
            error_tensor: list[float] | Tensor, 
            x_label: str,
            y_label: str,
            c_label: str,
            title: str = None
        ):
        
        self.latent_tensor = latent_tensor
        self.error_tensor = error_tensor

        self.x_label = x_label
        self.y_label = y_label
        self.c_label = c_label

        self.title = title


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        """
        Draws the latent space scatterplot with color-coded errors on the given Axes.

        Args:
            ax (plt.Axes): The matplotlib Axes to draw on.
            fontsize (int, optional): Font size for title. Defaults to 10.
        """
        if len(self.error_tensor) != len(self.latent_tensor):
            raise ValueError("Length of error_tensor must match the number of rows in latent_space.")
        
        # Create a scatter plot with color mapping based on error
        scatter = ax.scatter(
            self.latent_tensor[:, 0],  
            self.latent_tensor[:, 1],
            c=self.error_tensor,
            cmap='viridis',
            alpha=0.7,
            edgecolors='w',
            s=50
        )
        
        # Add colorbar to indicate error magnitude
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(self.c_label, fontsize=fontsize)
        
        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
        ax.set_title(self.title, fontsize=fontsize + 2)
        
        ax.grid(True)





class DimReducedScatterPlot(AxesComponent):
    def __init__(self,
                 feature_tensor: torch.Tensor | np.ndarray,
                 color_data: torch.Tensor | np.ndarray,
                 perplexity: float = 30.0,
                 n_iter: int = 1000,
                 random_state: int = 42,
                 x_label: str = 't-SNE 1',
                 y_label: str = 't-SNE 2',
                 color_label: str = None,
                 title: str = None,
                 cmap: str = 'viridis'):
        """
        Initializes a scatter plot of high-dimensional data reduced to 2D using t-SNE.

        Args:
            feature_tensor: High-dimensional features to be reduced and plotted
            color_values: Values used for coloring the scatter points
            perplexity: t-SNE perplexity parameter (balances local and global structure)
            n_iter: Number of iterations for t-SNE optimization
            random_state: Random seed for reproducibility
            x_label: Label for x-axis
            y_label: Label for y-axis
            color_label: Label for the colorbar
            title: Plot title
            cmap: Colormap to use
        """

        self.feature_tensor = feature_tensor
        self.color_data = color_data

        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.x_label = x_label
        self.y_label = y_label
        self.color_label = color_label

        self.title = title
        self.cmap = cmap


    def _reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensionality of the input data to 2D using t-SNE.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Reduced data of shape (n_samples, 2)
        """
        tsne = TSNE(
            n_components = 2,
            perplexity = self.perplexity,
            n_iter = self.n_iter,
            random_state = self.random_state
        )
        return tsne.fit_transform(data)


    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares the input data for plotting by converting tensors to numpy
        and reducing dimensions if necessary.
        
        Returns:
            Tuple of (reduced_features, color_data) as numpy arrays
        """
        # Convert to numpy if necessary
        features = (self.feature_tensor.detach().cpu().numpy() 
                   if torch.is_tensor(self.feature_tensor) 
                   else self.feature_tensor)
        
        colors = (self.color_data.detach().cpu().numpy() 
                 if torch.is_tensor(self.color_data) 
                 else self.color_data)
        
        # Reduce dimensions if input is higher than 2D
        if features.shape[1] > 2:
            features = self._reduce_dimensions(features)
            
        return features, colors.flatten()


    def draw(self, ax: Axes, fontsize: int = 10):
        
        reduced_features, colors = self._prepare_data()
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=colors,
            cmap=self.cmap,
            alpha=0.7,
            edgecolors='w',
            s=50
        )

        cbar = plt.colorbar(scatter, ax=ax)
        if self.color_label:
            cbar.set_label(self.color_label, fontsize=fontsize)

        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)

        if self.title:
            ax.set_title(self.title, fontsize=fontsize + 2)

        ax.grid(True)




class MultiScatterPlot(AxesComponent):

    def __init__(
            self, 
            tensor_data: dict[str | int, Tensor], 
            x_label: str,
            y_label: str,
            title: str = None
        ):
        
        self.tensor_data = tensor_data

        self.x_label = x_label
        self.y_label = y_label

        self.title = title


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        """
        Draws the latent space scatterplot with color-coded errors on the given Axes.

        Args:
            ax (plt.Axes): The matplotlib Axes to draw on.
            fontsize (int, optional): Font size for title. Defaults to 10.
        """
        
        for i, (label, tensor) in enumerate(self.tensor_data.items()):

            ax.scatter(
                tensor[:, 0],  
                tensor[:, 1],
                c=self.error_tensor,
                alpha=0.7,
                edgecolors='w',
                s=50,
                label = label,
            )
        
        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
        ax.set_title(self.title, fontsize=fontsize + 2)
        
        ax.grid(True)