
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from helper_tools import plot_training_losses, plot_param_norms


"""
LossComponentObserver ABC
-------------------------------------------------------------------------------------------------------------------------------------------
Abstract Base Class for Observers for either a Loss or a LossTerm.
These Observers are incorporated and called in the loss components themselves, 
while IterObservers are incorporated and called in a TrainingProcedure class.

This distinction is necessary to track the contributions of individual loss components in a composite loss function.
"""
class LossComponentObserver(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass




"""
LossObserver
-------------------------------------------------------------------------------------------------------------------------------------------
Tracks scalar losses, i.e. batch-aggregated single loss values, produced by Loss instances.
"""
class LossObserver(LossComponentObserver):

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



"""
LossTermObserver
-------------------------------------------------------------------------------------------------------------------------------------------
For tracking individual loss terms that can occur by themselves or in a composite loss term.

Suppose the same LossTerm is used both in isolation to create a Loss instance, for example for an autoencoder,
and at the same time in a CompositeLossTerm, when also used in an End-to-End fashion in a composed model.
"""
class LossTermObserver(LossComponentObserver):

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




"""
CompositeLossTermObserver
-------------------------------------------------------------------------------------------------------------------------------------------
This Observer is designed to track all the LossTerms in a CompositeLossTerm individually.

For example in a VAE loss, we have both a Log-Likelihood and a KL-Divergence term, 
and understanding the training process requires understanding the contributions of both.
"""
class CompositeLossTermObserver(LossComponentObserver):

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