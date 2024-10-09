
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from helper_tools import plot_training_losses, plot_param_norms

"""
Autoencoder Parameter Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossObserver:

    def __init__(self, *names):
        
        self.losses = {name: [] for name in names}
    

    def __call__(self, name: str, loss: Tensor):

        if torch.isnan(loss).any():
            print(f"{name} contains NaN values")
            raise StopIteration
        
        if torch.isinf(loss).any():
            print(f"{name} contains Inf values")
            raise StopIteration
        
        self.losses[name].append(loss.item())

        
    def plot_results(self):

        title: str = "Loss Development",

        mosaic_layout = [
            [f'loss_{name}', f'loss_{name}']
            for name in self.losses.keys()
        ]

        fig = plt.figure(figsize=(14, 7), layout = 'constrained')  
        axs = fig.subplot_mosaic(mosaic_layout)

        for name, losses in self.losses.items():

            axs[f'loss_{name}'] = plot_training_losses(
                losses = losses, 
                axes = axs[f'loss_{name}'],
                title = f'Losses {name}',
            )

        fig.suptitle(title)

        plt.show()

