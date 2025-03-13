
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
from matplotlib.gridspec import GridSpec

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from .helper_tools import remove_duplicate_plot_descriptors

from .axes_components import AxesComponent


"""
PlotMatrix 
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class PlotMatrix:

    def __init__(
            self, 
            title: str = None, 
            sharex: str|bool = False, 
            sharey: str|bool = False,
            remove_dupl_descriptors: bool = True,
            save_path: Path = None,
        ):
        plt.style.use('bmh')
        self.title = title
        self.sharex = sharex
        self.sharey = sharey
        self.remove_dupl_descriptors = remove_dupl_descriptors
        self.save_path = save_path

        self.plots: dict[tuple[int], AxesComponent] = {}

    
    def add_plot_dict(self, plot_dict: dict[tuple, AxesComponent]):
        self.plots.update(plot_dict)


    def setup_fig(self, figsize: tuple[int]):

        key_tuples = self.plots.keys()
        rows = max([key[0] for key in key_tuples]) + 1
        cols = max([key[1] for key in key_tuples]) + 1
        #figsize=(15,0.4*number_diagrams), sharex=True
        self.fig, self.axes = plt.subplots(
            rows, 
            cols, 
            sharex = self.sharex,
            sharey = self.sharey, 
            squeeze = False,
            figsize=figsize,
            constrained_layout=True,
        )


    def adjust_titles_and_labels(self, fontsize):
        
        # Define functions to extract titles, xlabels, and ylabels from axes objects
        get_title = np.vectorize(lambda ax: ax.get_title())
        get_xlabel = np.vectorize(lambda ax: ax.get_xlabel())
        get_ylabel = np.vectorize(lambda ax: ax.get_ylabel())

        # Extract titles, xlabels, and ylabels using vectorized operations
        titles = get_title(self.axes)
        xlabels = get_xlabel(self.axes)
        ylabels = get_ylabel(self.axes)

        if self.remove_dupl_descriptors:
            titles = remove_duplicate_plot_descriptors(titles, axis = 1, inverse = False)
            xlabels = remove_duplicate_plot_descriptors(xlabels, axis = 1, inverse = True)
            ylabels = remove_duplicate_plot_descriptors(ylabels, axis = 0, inverse = False)

        for (i,j), ax in np.ndenumerate(self.axes):
            ax.set_title(titles[i,j], fontdict={'fontsize': fontsize})
            ax.set_xlabel(xlabels[i,j], fontsize = fontsize)
            ax.set_ylabel(ylabels[i,j], fontsize = fontsize)

        #self.fig.tight_layout()
        #plt.subplots_adjust(left=0.1, right=1.2, top=1.2, bottom=0.2, wspace=0.2, hspace=0.3)


    def draw(self, **kwargs):

        fontsize = kwargs.get('fontsize', 10)
        figsize = kwargs.get('figsize', (10, 6))

        self.setup_fig(figsize = figsize)

        for (row, col), plot in self.plots.items():
            #print(row, col, plot)
            ax = self.axes[row, col]  
            plot.draw(ax, fontsize = fontsize)
        
        #.get_current_fig_manager().set_window_title()
        #.subplots_adjust(wspace=0.4)
        if self.title:
            self.fig.suptitle(self.title, fontsize=fontsize)

        #self.fig.align_xlabels(self.axes)
        #self.fig.align_ylabels(self.axes)

        self.adjust_titles_and_labels(fontsize)

        if self.save_path:
            #filename = self.title.lower().replace(" ", "_") if self.title else 'plot'
            self.fig.savefig(
                self.save_path,
                dpi=300,
                bbox_inches='tight',
                #pad_inches=0.5,
            )

        plt.show()
        #self.fig.show()



"""
PlotMosaic
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class PlotMosaic:
    
    def __init__(self, title: str = None, layout="constrained", save_path: Path = None):
        
        self.title = title
        self.layout = layout

        self.plots: dict[str, AxesComponent] = {}
        self.mosaic_layout = []

        self.save_path = save_path


    def add_plot(self, label: str, component: AxesComponent):
        self.plots[label] = component


    def define_mosaic(self, mosaic_list_of_lists: list[list[str]]):
        """
        Example:
        mosaic_list_of_lists = [
            ["top", "top"],
            ["left_bottom", "right_bottom"]
        ]
        This means 'top' spans 2 columns in the first row.
        """
        self.mosaic_layout = mosaic_list_of_lists


    def draw(self, **kwargs):
        
        fontsize = kwargs.get('fontsize', 9)
        figsize = kwargs.get('figsize', (10, 6))
      
        # Create figure and axes
        fig = plt.figure(figsize = figsize, layout = self.layout)  
        axs = fig.subplot_mosaic(self.mosaic_layout)

        # Draw plots (associate axes with labels)
        for label, component in self.plots.items():
            ax = axs[label] 
            component.draw(ax, fontsize=fontsize)

        if self.title:
            fig.suptitle(self.title)

        if self.save:
            fname = self.title.lower().replace(" ", "_") if self.title else 'plot'
            fig.savefig(f'Figures/{fname}', bbox_inches='tight')

        fig.show() 

    









