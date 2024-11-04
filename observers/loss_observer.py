
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
Loss Observer Prime
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TrainingLossObserver(IterObserver):

    def __init__(self, n_epochs: int, n_iterations: int):
        
        self.losses = torch.zeros(size = (n_epochs, n_iterations))
    

    def __call__(self, epoch: int, iter_idx: int, batch_loss: Tensor, **kwargs):

        if torch.isnan(batch_loss).any():
            print(f"Loss at epoch = {epoch}, iteration = {iter_idx} contains NaN values")
            raise StopIteration
        
        if torch.isinf(batch_loss).any():
            print(f"Loss at epoch = {epoch}, iteration = {iter_idx} contains Inf values")
            raise StopIteration
        
        self.losses[epoch, iter_idx] = batch_loss.detach()



"""
DictLoss Observer Prime
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class LossTermObserver(IterObserver):

    def __init__(self, n_epochs: int, n_iterations: int):
        
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations

        self.losses = torch.zeros(size = (n_epochs, n_iterations))

        self.epoch = 0
        self.iter_idx = 0
        

    def __call__(self, batch_loss: Tensor, **kwargs):

        if torch.isnan(batch_loss).any():
            print(f"Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains NaN values")
            raise StopIteration
        
        if torch.isinf(batch_loss).any():
            print(f"Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains Inf values")
            raise StopIteration
            
        self.losses[self.epoch, self.iter_idx] = batch_loss.detach()

        #update indices
        if self.iter_idx + 1 == self.n_iterations:

            self.epoch += 1
            self.iter_idx = 0

        else:
            self.iter_idx += 1





class CompositeLossTermObserver(IterObserver):

    def __init__(self, n_epochs: int, n_iterations: int, loss_names: list[str]):
        
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations

        self.losses = {name: torch.zeros(size = (n_epochs, n_iterations)) for name in loss_names}

        self.epoch = 0
        self.iter_idx = 0
        

    def __call__(self, batch_losses: Tensor, **kwargs):

        for name, batch_loss in batch_losses.items():

            if torch.isnan(batch_loss).any():
                print(f"Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains NaN values")
                raise StopIteration
            
            if torch.isinf(batch_loss).any():
                print(f"Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains Inf values")
                raise StopIteration
            
            self.losses[name][self.epoch, self.iter_idx] = batch_loss.detach()

        #update indices
        if self.iter_idx + 1 == self.n_iterations:

            self.epoch += 1
            self.iter_idx = 0

        else:
            self.iter_idx += 1


    def plot_results(self):

        title: str = "Loss Development",

        mosaic_layout = [
            [f'loss_{name}', f'loss_{name}']
            for name in self.losses.keys()
        ]

        fig = plt.figure(figsize=(14, 7), layout = 'constrained')  
        axs = fig.subplot_mosaic(mosaic_layout)

        for name, loss_tensor in self.losses.items():
            
            ax: Axes = axs[f'loss_{name}']
            loss_values = loss_tensor.flatten(start_dim = 0, end_dim = 1)
            
            iterations = len(loss_values)

            ax.plot(range(iterations), loss_values)

            for epoch in range(1, self.n_epochs):
                ax.axvline(x = epoch * self.n_iterations, color = 'r', linestyle = '--')

            ax.set_title(name)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss Value')

        fig.suptitle(title)

        plt.show()




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