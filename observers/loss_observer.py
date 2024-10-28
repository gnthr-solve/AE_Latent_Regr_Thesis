
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .training_observer import IterObserver

from helper_tools import plot_training_losses, plot_param_norms

"""
Loss Observer First Trial
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossObserverAlpha:

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



"""
Loss Observer Prime
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossObserver(IterObserver):

    def __init__(self, n_epochs, n_iterations):
        
        self.losses = torch.zeros(size = (n_epochs, n_iterations))
    

    def __call__(self, epoch: int, iter_idx: int, batch_loss: Tensor, **kwargs):

        if torch.isnan(batch_loss).any():
            print(f"Loss at epoch = {epoch}, iteration = {iter_idx} contains NaN values")
            raise StopIteration
        
        if torch.isinf(batch_loss).any():
            print(f"Loss at epoch = {epoch}, iteration = {iter_idx} contains Inf values")
            raise StopIteration
        
        self.losses[epoch, iter_idx] = batch_loss.detach()

        



# """
# Detailed Loss Observer
# -------------------------------------------------------------------------------------------------------------------------------------------
# """
# class DetailedLossObserver:

#     def __init__(self, batch_size: int, dataset_size: int, n_epochs: int):

#         self.batch_size = batch_size
#         self.sample_losses = torch.zeros(size = (n_epochs, dataset_size))

    
#     def __call__(self, epoch: int, batch_idx: int, sample_batch_losses: Tensor):

#         start_idx = self.batch_size * batch_idx
#         end_idx = start_idx + self.batch_size

#         self.sample_losses[epoch, start_idx:end_idx] = sample_batch_losses.detach()