

import torch
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MaxNLocator, MultipleLocator, AutoMinorLocator

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#from helper_tools import remove_duplicate_plot_descriptors

from .axes_components import AxesComponent, AxesComponentDecorator


"""
LineMarkDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class LineMarkDecorator(AxesComponentDecorator):

    def __init__(
            self, 
            axis_component: AxesComponent, 
            horizontal_value: float = None, 
            vertical_value: float = None,
            **style_kwargs,
            ):
        
        super().__init__(axis_component)

        self.horizontal_value = horizontal_value
        self.vertical_value = vertical_value
        self.style_kwargs = style_kwargs

    
    def draw(self, ax: Axes, **kwargs):

        color = self.style_kwargs.get('color', 'red')
        ax.axhline(y=self.horizontal_value, color = color, linestyle='dashed')
        ax.axvline(x=self.vertical_value, color = color, linestyle='dashed')

        super().draw(ax, **kwargs)

