
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Union

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MaxNLocator, MultipleLocator, AutoMinorLocator

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from ..axes_components import AxesComponent



"""
HistogramPlot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class DataFrameHistogramPlot(AxesComponent):
    def __init__(
            self, 
            df: pd.DataFrame, 
            value_column: str,
            bins: int = 20,
            density: bool = False,
            color: str = 'b',
            xlabel: str = None,
            ylabel: str = 'Frequency',
            title: str = None
        ):
        
        self.df = df
        self.value_column = value_column
        self.bins = bins
        self.density = density
        self.color = color
        self.xlabel = xlabel or value_column
        self.ylabel = ylabel
        self.title = title


    def draw(self, ax: Axes, **kwargs):

        ax.hist(
            x = self.df[self.value_column],
            bins = self.bins,
            density = self.density,
            color = self.color,
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True)

        if self.title:
            ax.set_title(self.title)




class TensorHistogramPlot(AxesComponent):
    def __init__(
            self, 
            data: Union[torch.Tensor, np.ndarray],
            bins: int = 20,
            density: bool = False,
            color: str = 'b',
            xlabel: str = None,
            ylabel: str = 'Frequency',
            title: str = None
        ):
        """
        Tensor/Array-based histogram plot.
        
        Args:
            data: Input tensor or array
            bins: Number of histogram bins
            density: Whether to normalise the histogram
            color: Bar color
            title: Plot title
            xlabel: X-axis label
        """
        self.data = data
        self.bins = bins
        self.density = density
        self.color = color
        self.title = title
        self.xlabel = xlabel


    def prepare_data(self):

        if isinstance(self.data, torch.Tensor):
            return self.data.detach().cpu().numpy()
        
        return self.data


    def draw(self, ax: Axes, **kwargs):

        values = self.prepare_data()
        
        ax.hist(
            x=values,
            bins=self.bins,
            density=self.density,
            color=self.color
        )
        
        ax.set_xlabel(self.xlabel if self.xlabel else '')
        ax.set_ylabel('Density' if self.density else 'Frequency')

        ax.grid(True)

        if self.title:
            ax.set_title(self.title)




"""
GroupedDataFrameMultiHistogramPlot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GroupedDataFrameMultiHistogramPlot(AxesComponent):

    def __init__(
            self, 
            df: pd.DataFrame,
            value_column: str,
            group_column: str,
            bins: int = 20,
            density: bool = True,
            alpha: float = 0.6,
            xlabel: str = None,
            ylabel: str = 'Frequency',
            title: str = None
        ):
        
        self.df = df
        self.value_column = value_column
        self.group_column = group_column
        self.bins = bins
        self.density = density
        self.alpha = alpha
        self.xlabel = xlabel or value_column
        self.ylabel = ylabel
        self.title = title


    def draw(self, ax: Axes, **kwargs):

        for group_name, group_df in self.df.groupby(self.group_column):
            ax.hist(
                x = group_df[self.value_column],
                bins = self.bins,
                density = self.density,
                label = group_name,
                alpha = self.alpha
            )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.grid(True)
        ax.legend()

        if self.title:
            ax.set_title(self.title)



"""
Multi Source Histograms
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TensorMultiHistogramPlot(AxesComponent):

    def __init__(
            self, 
            data_dict: dict[str, Union[torch.Tensor, np.ndarray]],
            bins: int = 20,
            range: tuple[float, float] = None,
            density: bool = True,
            alpha: float = 0.6,
            xlabel: str = 'Value',
            ylabel: str = 'Frequency',
            title: str = None
        ):
        
        self.data_dict = {
            name: (data.detach().cpu().numpy() if torch.is_tensor(data) else data)
            for name, data in data_dict.items()
        }

        self.bins = bins
        self.range = range
        self.density = density
        self.alpha = alpha
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title


    def draw(self, ax: Axes, **kwargs):

        for name, data in self.data_dict.items():

            ax.hist(
                x = data.flatten(),
                bins = self.bins,
                range = self.range,
                density = self.density,
                label = name,
                alpha = self.alpha,
                histtype='stepfilled',
            )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.grid(True)
        ax.legend()
        
        if self.title:
            ax.set_title(self.title)



class MultiDataFrameHistogramPlot(AxesComponent):

    def __init__(
            self, 
            df_dict: dict[str, pd.DataFrame],
            value_column: str,
            bins: int = 200,
            density: bool = True,
            alpha: float = 0.6,
            xlabel: str = None,
            ylabel: str = 'Frequency',
            title: str = None
        ):
        
        self.df_dict = df_dict
        self.value_column = value_column
        self.bins = bins
        self.density = density
        self.alpha = alpha
        self.xlabel = xlabel or value_column
        self.ylabel = ylabel
        self.title = title


    def draw(self, ax: Axes, **kwargs):

        for name, df in self.df_dict.items():
            ax.hist(
                x = df[self.value_column],
                bins = self.bins,
                density = self.density,
                label = name,
                alpha = self.alpha
            )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.grid(True)
        ax.legend()

        if self.title:
            ax.set_title(self.title)

'''

class DataFrameHistogramPlot(AxesComponent):
    def __init__(self, 
                 df: pd.DataFrame, 
                 value_column: str,
                 group_column: str = None,
                 bins: int = 20,
                 normalize: bool = False,
                 color: str = 'b',
                 title: str = None,
                 xlabel: str = None):
        """
        DataFrame-based histogram plot.
        
        Args:
            df: Input DataFrame
            value_column: Column containing values to plot
            group_column: Optional column to group by before selecting values
            bins: Number of histogram bins
            normalize: Whether to normalize the histogram
            color: Bar color
            title: Plot title
            xlabel: X-axis label (defaults to value_column name)
        """
        self.df = df
        self.value_column = value_column
        self.group_column = group_column
        self.bins = bins
        self.normalize = normalize
        self.color = color
        self.title = title
        self.xlabel = xlabel or value_column

    def prepare_data(self):
        if self.group_column:
            return [group[self.value_column].iloc[-1] 
                   for _, group in self.df.groupby(self.group_column)]
        return self.df[self.value_column].values

    def draw(self, ax: Axes, **kwargs):
        values = self.prepare_data()
        
        ax.hist(x=values,
                bins=self.bins,
                density=self.normalize,
                color=self.color)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Density' if self.normalize else 'Frequency')
        ax.grid(True)
        if self.title:
            ax.set_title(self.title)

            

class DataFrameStackedHistogramPlot(AxesComponent):
    def __init__(self, 
                 df: pd.DataFrame,
                 value_column: str,
                 stack_column: str,
                 group_column: str = None,
                 bins: int = 20,
                 normalize: bool = False,
                 title: str = None,
                 xlabel: str = None):
        """
        DataFrame-based stacked histogram plot.
        
        Args:
            df: Input DataFrame
            value_column: Column containing values to plot
            stack_column: Column used to create separate histograms
            group_column: Optional column to group by before selecting values
            bins: Number of histogram bins
            normalize: Whether to normalize the histogram
            title: Plot title
            xlabel: X-axis label (defaults to value_column name)
        """
        self.df = df
        self.value_column = value_column
        self.stack_column = stack_column
        self.group_column = group_column
        self.bins = bins
        self.normalize = normalize
        self.title = title
        self.xlabel = xlabel or value_column

    def prepare_data(self):
        if self.group_column:
            return {
                name: [group[self.value_column].iloc[-1] 
                      for _, group in subdf.groupby(self.group_column)]
                for name, subdf in self.df.groupby(self.stack_column)
            }
        return {
            name: subdf[self.value_column].values
            for name, subdf in self.df.groupby(self.stack_column)
        }

    def draw(self, ax: Axes, **kwargs):
        data_dict = self.prepare_data()
        
        for name, values in data_dict.items():
            ax.hist(x=values,
                   bins=self.bins,
                   density=self.normalize,
                   label=name,
                   alpha=0.6)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Density' if self.normalize else 'Frequency')
        ax.grid(True)
        ax.legend()
        if self.title:
            ax.set_title(self.title)


class TensorStackedHistogramPlot(AxesComponent):
    def __init__(self, 
                 data_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                 bins: int = 20,
                 normalize: bool = False,
                 title: str = None,
                 xlabel: str = None):
        """
        Tensor/Array-based stacked histogram plot.
        
        Args:
            data_dict: Dictionary mapping names to tensors/arrays
            bins: Number of histogram bins
            normalize: Whether to normalize the histogram
            title: Plot title
            xlabel: X-axis label
        """
        self.data_dict = data_dict
        self.bins = bins
        self.normalize = normalize
        self.title = title
        self.xlabel = xlabel

    def prepare_data(self):
        return {
            name: data.detach().cpu().numpy() 
            if isinstance(data, torch.Tensor) else data
            for name, data in self.data_dict.items()
        }

    def draw(self, ax: Axes, **kwargs):
        data_dict = self.prepare_data()
        
        for name, values in data_dict.items():
            ax.hist(x=values,
                   bins=self.bins,
                   density=self.normalize,
                   label=name,
                   alpha=0.6)
        
        ax.set_xlabel(self.xlabel if self.xlabel else '')
        ax.set_ylabel('Density' if self.normalize else 'Frequency')
        ax.grid(True)
        ax.legend()
        if self.title:
            ax.set_title(self.title)
'''