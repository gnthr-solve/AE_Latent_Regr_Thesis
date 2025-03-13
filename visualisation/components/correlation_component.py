
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from matplotlib.ticker import AutoLocator, MaxNLocator, MultipleLocator, AutoMinorLocator

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from ..axes_components import AxesComponent


class CorrelationPlot(AxesComponent):
    def __init__(
            self, data: pd.DataFrame, 
            method: str = 'pearson',
            cmap: str = 'coolwarm',
            **kwargs
        ):
        super().__init__(**kwargs)
        self.data = data
        self.method = method
        self.cmap = cmap

    def draw(self, ax: Axes, **kwargs):
        fontsize = kwargs.get('fontsize', 10)
        
        corr = self.data.corr(method=self.method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, cmap=self.cmap,
                   annot=True, fmt='.2f', ax=ax,
                   cbar_kws={"shrink": .5})
        
        
