import torch
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Iterable, Callable, Optional

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



class LinePlot(AxesComponent):
    def __init__(
            self, 
            x_data: Iterable, 
            y_data: Iterable, 
            x_label: str = None,
            y_label: str = None,
            x_scale: str = 'linear',
            y_scale: str = 'linear',
            color: str = 'blue',
            title: str = None,
        ):
        
        self.x_data = x_data
        self.y_data = y_data
        
        self.x_label = x_label if x_label else 'x'
        self.y_label = y_label if y_label else 'y'

        self.x_scale = x_scale
        self.y_scale = y_scale

        self.color = color
        self.title = title


    def draw(self, ax: Axes, fontsize: int = 10):
        
        ax.plot(self.x_data, self.y_data, color=self.color)
        
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)

        ax.set_xscale(self.x_scale)
        ax.set_yscale(self.y_scale)
        
        ax.set_title(self.title, fontsize=fontsize + 2)

        #ax.legend()
        ax.grid(True)




class SingleTrajectoryPlot(AxesComponent):
    def __init__(
        self,
        x_data: Iterable,
        y_data: Iterable,
        separator_x: Optional[Iterable] = None,  
        x_label: str = 'X',
        y_label: str = 'Y',
        color: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        Initializes a trajectory plot that works on generic x/y data.

        Args:
            x_data: Iterable of values for the x-axis.
            y_data: Iterable of values for the y-axis.
            separator_x: Optional iterable of x values at which vertical separators are drawn.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            color: Plot color.
            title: Plot title.
        """
        self.x_data = self._to_numpy(x_data)
        self.y_data = self._to_numpy(y_data)
        self.separator_x = self._to_numpy(separator_x) if separator_x is not None else None

        self.x_label = x_label
        self.y_label = y_label
        self.color = color
        self.title = title


    def _to_numpy(self, data: Iterable) -> np.ndarray:
        """Convert data to a numpy array if necessary."""
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)


    def draw(self, ax: Axes, fontsize: int = 10):
        """
        Draws the trajectory plot with optional vertical separators.

        Args:
            ax: Matplotlib Axes to draw on.
            fontsize: Font size for labels and title.
        """
        ax.plot(self.x_data, self.y_data, color = self.color)
        
        # Draw vertical separators if provided.
        if self.separator_x is not None:
            for sep in np.sort(self.separator_x):
                ax.axvline(x=sep, color='red', linestyle='--', linewidth=0.8)

        # Labels, title, grid, and legend.
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)

        if self.title:
            ax.set_title(self.title, fontsize=fontsize + 2)
        
        ax.grid(True)
        


class TrajectoryPlotFill(AxesComponent):
    def __init__(
        self,
        x_data: Iterable,
        y_data: Iterable,
        separator_x: Optional[Iterable] = None,  
        x_label: str = 'X',
        y_label: str = 'Y',
        color: Optional[str] = None,
        title: Optional[str] = None,
        fill_style: str = 'gradient',   # "gradient", "alternating", or "single"
        base_fill_color: str = 'purple'
    ):
        """
        Initializes a trajectory plot with filled sections.

        Args:
            x_data: Iterable of x-axis values.
            y_data: Iterable of y-axis values.
            separator_x: Optional iterable of x values where sections change.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            color: Color for the main trajectory line.
            title: Plot title.
            fill_style: Style for filling sections ("gradient", "alternating", or "single").
            base_fill_color: Base color used for filling.
        """
        self.x_data = self._to_numpy(x_data)
        self.y_data = self._to_numpy(y_data)
        self.separator_x = np.sort(self._to_numpy(separator_x)) if separator_x is not None else None

        self.x_label = x_label
        self.y_label = y_label
        self.color = color
        self.title = title

        self.fill_style = fill_style
        self.base_fill_color = base_fill_color


    def _to_numpy(self, data: Iterable) -> np.ndarray:
        """
        Convert data to numpy array.
        """
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)


    def get_fill_colors(self, n_sections: int) -> list:
        """
        Generate a list of RGBA colors for filling based on the fill_style.
        """
        if self.fill_style == 'gradient':
            base_color = to_rgba(self.base_fill_color)
            alphas = np.linspace(0.05, 0.15, n_sections)
            return [(*base_color[:3], alpha) for alpha in alphas]
        
        elif self.fill_style == 'alternating':
            color1 = to_rgba(self.base_fill_color, alpha=0.1)
            color2 = to_rgba(self.base_fill_color, alpha=0.05)
            return [color1 if i % 2 == 0 else color2 for i in range(n_sections)]
        
        else:  # "single"
            return [to_rgba(self.base_fill_color, alpha=0.1)] * n_sections


    def draw(self, ax: Axes, fontsize: int = 10):
        """
        Draws the trajectory plot with filled sections defined by separators.

        Args:
            ax: Matplotlib Axes to draw on.
            fontsize: Font size for labels and title.
        """
        # Plot the main trajectory line.
        ax.plot(self.x_data, self.y_data, color=self.color)

        # Determine fill regions.
        # If separators are provided, use them as boundaries.
        if self.separator_x is not None and len(self.separator_x) > 0:
            # Get fill colors for each section.
            n_sections = len(self.separator_x) + 1  # sections are one more than number of boundaries
            fill_colors = self.get_fill_colors(n_sections)
            
            # Get overall x-range.
            x_start = np.min(self.x_data)
            x_end = np.max(self.x_data)
            
            # Create list of boundaries including the start and end.
            boundaries = [x_start] + list(self.separator_x) + [x_end]
            
            # Before filling, get the current y-limits.
            # This is useful because it ensures that the fills cover the visible area.
            y_min, y_max = ax.get_ylim()
            
            # Fill each region and draw a vertical line at the boundary (except at the very start)
            for i in range(len(boundaries) - 1):
                x0, x1 = boundaries[i], boundaries[i+1]
                ax.fill_between([x0, x1], y_min, y_max, color=fill_colors[i], alpha=1.0)
                if i > 0:  # For all boundaries except the very first, draw a separator line at the left boundary of the section.
                    ax.axvline(x=x0, color='red', linestyle='--', linewidth=0.8)
                    
            # Optionally, you can draw a separator at the very last boundary
            ax.axvline(x=boundaries[-1], color='red', linestyle='--', linewidth=0.8)

        else:
            # If no separators are provided, nothing extra is done.
            pass

        # Set labels and title, add legend and grid.
        ax.set_xlabel(self.x_label, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)

        if self.title:
            ax.set_title(self.title, fontsize=fontsize + 2)

        ax.grid(True)
        

