
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Iterable, Callable

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.tri as tri


from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import AutoLocator, MaxNLocator, MultipleLocator, AutoMinorLocator

from ..axes_components import AxesComponent




class ContourPlot(AxesComponent):
    def __init__(self, 
                 x_data: Iterable, 
                 y_data: Iterable, 
                 color_data: Iterable, 
                 x_label: str = None, 
                 y_label: str = None,
                 color_label: str = None,
                 levels: int = 10,
                 title: str = None,
                 cmap: str = 'viridis'):
        """
        Initializes the ContourPlot.

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

        self.levels = levels
        self.title = title
        self.cmap = cmap


    def draw(self, ax: plt.Axes, fontsize: int = 10):
        
        ax.tricontour(
            self.x_data, 
            self.y_data,
            self.color_data, 
            levels = self.levels, 
            linewidths = 0.5, 
            colors = 'k'
        )

        cntr = ax.tricontourf(
            self.x_data, 
            self.y_data,
            self.color_data, 
            levels = self.levels,
            extend = 'both',
            cmap = "RdBu_r",
        )

        cbar = plt.colorbar(cntr, ax=ax)
        ax.plot(self.x_data, self.y_data, 'ko', ms=3)

        #ax.set(xlim=(-2, 2), ylim=(-2, 2))
        ax.set_title(self.title)

        if self.color_label:
            cbar.set_label(self.color_label, fontsize=fontsize)

        # Set labels and title
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
    
        #plt.subplots_adjust(hspace=0.5)
    