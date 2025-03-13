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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from matplotlib import cm
from matplotlib.colors import to_rgba

from ..axes_components import AxesComponent


"""
SingleTrajectory
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SingleTrajectoryPlotDF(AxesComponent):
    def __init__(
            self, 
            df: pd.DataFrame, 
            time_col: str, 
            value_col: str,
            separator_col: str,
            time_label: str = 'Time',
            value_label: str = 'Value',
            color: str = None,
            title: str = None,

        ):
        """
        Initializes the SingleTrajectoryPlot.

        Args:
            df (pd.DataFrame): The timeseries DataFrame.
            time_col (str): The column name for time.
            value_col (str): The column name for the value to plot.
            separator_col (str): The column name for horizontal separators.
        """
        self.df = df
        self.time_col = time_col
        self.value_col = value_col
        self.separator_col = separator_col

        self.time_label = time_label
        self.value_label = value_label
        self.color = color

        self.title = title


    def draw(self, ax: Axes, fontsize: int = 10):
        
        ax.plot(self.df[self.time_col], self.df[self.value_col], label=self.value_col, color=self.color)
        
        separators = self.df[self.separator_col].unique()
        for separator in separators:

            separator_df = self.df[self.df[self.separator_col] == separator]

            ax.axvline(x=separator_df[self.time_col].max(), color='red', linestyle='--', linewidth=0.8)
        
        # Set labels and title
        ax.set_xlabel(self.time_label, fontsize=fontsize)
        ax.set_ylabel(self.value_label, fontsize=fontsize)

        ax.set_title(self.title, fontsize=fontsize + 2)

        ax.legend()
        ax.grid(True)




class SingleTrajectoryPlotFillDF(AxesComponent):
    def __init__(
            self,
            df: pd.DataFrame,
            time_col: str,
            value_col: str,
            separator_col: str,
            time_label: str = 'Time',
            value_label: str = 'Value',
            color: str = None,
            title: str = None,
            fill_style: str = 'gradient',  # 'gradient', 'alternating', or 'single'
            base_fill_color: str = 'purple'
        ):

        self.df = df
        self.time_col = time_col
        self.value_col = value_col
        self.separator_col = separator_col

        self.time_label = time_label
        self.value_label = value_label
        self.color = color
        
        self.fill_style = fill_style
        self.base_fill_color = base_fill_color

        self.title = title


    def get_fill_colors(self, n_sections: int) -> list:
        """Generate colors for filling based on the chosen style."""
        if self.fill_style == 'gradient':
            # Create a gradient from very light to slightly darker
            base_color = to_rgba(self.base_fill_color)
            alphas = np.linspace(0.05, 0.15, n_sections)
            return [(*base_color[:3], alpha) for alpha in alphas]
        
        elif self.fill_style == 'alternating':
            # Alternate between two very light colors
            color1 = to_rgba(self.base_fill_color, alpha=0.1)
            color2 = to_rgba(self.base_fill_color, alpha=0.05)
            return [color1 if i % 2 == 0 else color2 for i in range(n_sections)]
        
        else:  # 'single'
            return [to_rgba(self.base_fill_color, alpha=0.1)] * n_sections


    def draw(self, ax: Axes, fontsize: int = 10):
        # Plot the main trajectory
        ax.plot(self.df[self.time_col], self.df[self.value_col], 
                label=self.value_col, color=self.color)
        
        # Get unique separators and sort them
        separators = sorted(self.df[self.separator_col].unique())
        
        # Get fill colors
        fill_colors = self.get_fill_colors(len(separators))
        
        # Get the y-axis limits for filling
        y_min, y_max = ax.get_ylim()
        
        # Fill areas between separators
        for i, (sep1, sep2) in enumerate(zip(separators[:-1], separators[1:])):
            # Get x coordinates for current section
            x1 = self.df[self.df[self.separator_col] == sep1][self.time_col].max()
            x2 = self.df[self.df[self.separator_col] == sep2][self.time_col].max()
            
            # Fill the area
            ax.fill_between([x1, x2], y_min, y_max, 
                          color=fill_colors[i], alpha=1.0)
            
            # Add separator line
            ax.axvline(x=x2, color='red', linestyle='--', linewidth=0.8)
        
        # Handle the first section
        first_x = self.df[self.time_col].min()
        first_sep = self.df[self.df[self.separator_col] == separators[0]][self.time_col].max()
        ax.fill_between([first_x, first_sep], y_min, y_max, 
                       color=fill_colors[0], alpha=1.0)
        ax.axvline(x=first_sep, color='red', linestyle='--', linewidth=0.8)
        
        ax.set_xlabel(self.time_label, fontsize=fontsize)
        ax.set_ylabel(self.value_label, fontsize=fontsize)
        ax.set_title(self.title, fontsize=fontsize + 2)
        ax.legend()
        ax.grid(True)



"""
MultiTrajectory
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MultiTrajectoryPlotDF(AxesComponent):

    def __init__(
            self, 
            df: pd.DataFrame, 
            time_col: str, 
            value_cols: list[str], 
            separator_col: str,
            time_label: str = 'Time',
            value_label: str = 'Value',
            separator_color: str = 'black',
            title: str = None,
        ):

        self.df = df

        self.time_col = time_col
        self.value_cols = value_cols

        self.time_label = time_label
        self.value_label = value_label

        self.separator_col = separator_col
        self.separator_color = separator_color

        self.title = title


    def draw(self, ax: Axes, fontsize: int = 10):
    
        for col in self.value_cols:
            ax.plot(self.df[self.time_col], self.df[col], label=col)
        
        
        separators = self.df[self.separator_col].unique()
        for separator in separators:

            separator_df = self.df[self.df[self.separator_col] == separator]

            ax.axvline(x=separator_df[self.time_col].max(), color = self.separator_color, linestyle='--', linewidth=1)
        

        # Set labels and title
        ax.set_xlabel(self.time_label, fontsize=fontsize)
        ax.set_ylabel(self.value_label, fontsize=fontsize)
        ax.set_title(self.title, fontsize=fontsize + 2)
        ax.legend()
        ax.grid(True)



"""
Trajectory Comparison
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TrajectoriesComparisonPlotDF(AxesComponent):
    def __init__(
            self, 
            dfs: dict[str, pd.DataFrame], 
            time_col: str, 
            value_col: str,
            time_label: str = 'Time',
            value_label: str = 'Value',
            title: str = None,
        ):
        
        self.dfs = dfs

        self.time_col = time_col
        self.value_col = value_col

        self.time_label = time_label
        self.value_label = value_label

        self.title = title


    def draw(self, ax: Axes, fontsize: int = 10):
        
        # Plot each trajectory
        for name, df in self.dfs.items():
            ax.plot(df[self.time_col], df[self.value_col], label=name)
        
        # Set labels and title
        ax.set_xlabel(self.time_label, fontsize=fontsize)
        ax.set_ylabel(self.value_label, fontsize=fontsize)

        ax.set_title(self.title, fontsize=fontsize + 2)

        ax.legend()
        ax.grid(True)