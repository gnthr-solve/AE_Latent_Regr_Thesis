
import numpy as np
import pandas as pd

from itertools import product
from functools import wraps

from abc import ABC, abstractmethod
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator



"""
Plotting Helper Classes
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class StyleAdministrator:

    def scatter_style(self, style_kwargs: dict):

        scatter_style_dict = {
            'alpha': style_kwargs.get('alpha', 0.7),
            's': style_kwargs.get('marker_size', 15),
            'marker': style_kwargs.get('marker', '.'),
        }

        return scatter_style_dict

    
    def histogram_style(self, style_kwargs: dict):

        hist_style_dict = {
            'bins': style_kwargs.get('bins', 20),
            'alpha': style_kwargs.get('alpha', 0.7),
        }

        return hist_style_dict



style_admin = StyleAdministrator()


"""
Plotting Helper Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def remove_duplicate_plot_descriptors(array: np.ndarray, axis: int, inverse: bool):

    shape = array.shape
    axis_iter = [
        array[k,:] if axis == 0 else array[:, k]
        for k in range(shape[axis])
    ]

    for k, slice in enumerate(axis_iter):

        if inverse:
            mask_slice = slice[::-1]
        else:
            mask_slice = slice
        
        _, unique_slice_inds = np.unique(mask_slice, return_index = True)
        
        mask = np.zeros_like(slice, dtype=bool)
        mask[unique_slice_inds] = True

        if inverse:
            slice = np.where(mask[::-1], slice, '')
        else:
            slice = np.where(mask, slice, '')

        if axis == 0:
            array[k, :] = slice
        else:
            array[:, k] = slice
        
    return array
        



royal_blue = [0, 20/256, 82/256]
def annotate(ax, x, y, text, code):
    # Circle marker
    c = Circle((x, y), radius=0.15, clip_on=False, zorder=10, linewidth=2.5,
               edgecolor=royal_blue + [0.6], facecolor='none',
               path_effects=[withStroke(linewidth=7, foreground='white')])
    ax.add_artist(c)

    # use path_effects as a background for the texts
    # draw the path_effects and the colored text separately so that the
    # path_effects cannot clip other texts
    for path_effects in [[withStroke(linewidth=7, foreground='white')], []]:
        color = 'white' if path_effects else royal_blue
        ax.text(x, y-0.2, text, zorder=100,
                ha='center', va='top', weight='bold', color=color,
                style='italic', fontfamily='monospace',
                path_effects=path_effects)

        color = 'white' if path_effects else 'black'
        ax.text(x, y-0.33, code, zorder=100,
                ha='center', va='top', weight='normal', color=color,
                fontfamily='monospace', fontsize='medium',
                path_effects=path_effects)



