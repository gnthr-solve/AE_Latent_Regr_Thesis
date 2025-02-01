
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

from helper_tools import AbortTrainingError


"""
LossComponentObserver ABC
-------------------------------------------------------------------------------------------------------------------------------------------
Abstract Base Class for Observers for either a Loss or a LossTerm.
These Observers are incorporated and called in the loss components themselves, 
while IterObservers are incorporated and called in a TrainingProcedure class.

This distinction is necessary to track the contributions of individual loss components in a composite loss function.
"""
class LossComponentObserver(ABC):

    def __init__(self, n_epochs: int, dataset_size: int, batch_size: int, name: str, aggregated: bool = False):
        
        self.name = name
        self.aggregated = aggregated

        self.n_epochs = n_epochs
        self.n_iterations = dataset_size // batch_size + (dataset_size % batch_size > 0)
        self.batch_size = batch_size

        if self.aggregated:
            self.losses = torch.zeros(size = (n_epochs, self.n_iterations))
        else:
            self.losses = torch.zeros(size = (n_epochs, dataset_size))

        self.epoch = 0
        self.iter_idx = 0

    
    def inscribe(self, loss: Tensor, **kwargs):

        if self.aggregated:
            self.losses[self.epoch, self.iter_idx] += loss

        else:
            start_idx = self.batch_size * self.iter_idx
            end_idx = self.batch_size * (self.iter_idx + 1)

            self.losses[self.epoch, start_idx:end_idx] += loss


    def update_indices(self):

        if self.iter_idx + 1 == self.n_iterations:

            self.epoch += 1
            self.iter_idx = 0

        else:
            self.iter_idx += 1


    
    def verify_integrity(self):
        pass

    
    def truncate_observations(self):
        if self.aggregated:
            #print(f'{self.name} up to epoch={self.epoch}, iter_idx ={self.iter_idx}, loss_shape = {self.losses.shape}, losses: \n{self.losses}\n')
            self.losses = self.losses[:self.epoch, :self.iter_idx].flatten()

        else:
            self.losses = self.losses[:self.epoch, :self.iter_idx * self.batch_size].flatten()


    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass





"""
LossTermObserver
-------------------------------------------------------------------------------------------------------------------------------------------
For tracking individual loss terms that can occur by themselves or in a composite loss term.

Suppose the same LossTerm is used both in isolation to create a Loss instance, for example for an autoencoder,
and at the same time in a CompositeLossTerm, when also used in an End-to-End fashion in a composed model.
"""
class LossTermObserver(LossComponentObserver):

    def __init__(
            self, 
            n_epochs: int, 
            dataset_size: int, 
            batch_size: int, 
            name: str = None, 
            aggregated: bool = False,
        ):
        super().__init__(n_epochs, dataset_size, batch_size, name, aggregated)

        
    def __call__(self, loss_batch: Tensor, **kwargs):

        if not self.verify_integrity(loss_batch):
            self.truncate_observations()
            raise AbortTrainingError

        if self.aggregated:
            self.inscribe(loss_batch.mean())

        else:
            self.inscribe(loss_batch)

        self.update_indices()


    def verify_integrity(self, loss_batch: Tensor) -> bool:
        
        loss_batch_valid = True
        
        if torch.isnan(loss_batch).any():
            print(f"{self.name}-Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains NaN values")
            loss_batch_valid = False
            
        if torch.isinf(loss_batch).any():
            print(f"{self.name}-Loss at epoch = {self.epoch}, iteration = {self.iter_idx} contains Inf values")
            loss_batch_valid = False

        return loss_batch_valid


"""
CompositeLossTermObserver
-------------------------------------------------------------------------------------------------------------------------------------------
This Observer is designed to track all the LossTerms in a CompositeLossTerm individually.

For example in a VAE loss, we have both a Log-Likelihood and a KL-Divergence term, 
and understanding the training process requires understanding the contributions of both.
"""
class CompositeLossTermObserver(LossComponentObserver):

    def __init__(
            self, 
            n_epochs: int, 
            dataset_size: int, 
            batch_size: int, 
            members: list[str] | dict[str, LossComponentObserver],
            name: str = None, 
            aggregated: bool = False
        ):
        
        super().__init__(n_epochs, dataset_size, batch_size, name, aggregated)

        if isinstance(members, list):
            self.loss_obs = {
                member_name: LossTermObserver(n_epochs, dataset_size, batch_size, member_name, aggregated)
                for member_name in members
            }

        elif isinstance(members, dict):
            self.loss_obs = members
        

    def __call__(self, loss_batches: dict[str, Tensor], **kwargs):

        for name, loss_batch in loss_batches.items():

            try:
                self.loss_obs[name](loss_batch)

            except AbortTrainingError:
                self.truncate_observations(name)
                raise AbortTrainingError


            if self.aggregated:
                self.inscribe(loss_batch.mean())

            else:
                self.inscribe(loss_batch)
        
        self.update_indices()


    def truncate_observations(self, name_raiser: str):
        super().truncate_observations()

        for name, loss_observer in self.loss_obs.items():

            #raisers observations are truncated already
            if not name == name_raiser:

                loss_observer.truncate_observations()
    

    def plot_agg_results(self):

        title: str = "Loss Development",

        mosaic_layout_children = [
            [f'loss_{name}', f'loss_{name}']
            for name in self.loss_obs.keys()
        ]

        mosaic_layout = [[f'loss_{self.name}', f'loss_{self.name}'], *mosaic_layout_children]

        fig = plt.figure(figsize=(14, 7), layout = 'constrained')  
        axs = fig.subplot_mosaic(mosaic_layout)

        losses = {self.name: self.losses, **{name: loss_ob.losses for name, loss_ob in self.loss_obs.items()}}
        for name, loss_tensor in losses.items():
            
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




"""
ComposedLossTermObserver
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ComposedLossTermObserver(LossComponentObserver):

    def __init__(
            self, 
            n_epochs: int, 
            dataset_size: int, 
            batch_size: int, 
            loss_obs: dict[str, LossComponentObserver],
            name: str = None, 
            aggregated: bool = False
        ):
        
        super().__init__(n_epochs, dataset_size, batch_size, name, aggregated)
        self.loss_obs = loss_obs


    @property
    def losses(self):

        obs_losses = tuple(obs.losses for obs in self.loss_obs.values())
        stacked_obs_losses = torch.stack(obs_losses)

        return torch.sum(stacked_obs_losses, dim=0)


    def plot_agg_results(self):

        title: str = "Loss Development",

        mosaic_layout_children = [
            [f'loss_{name}', f'loss_{name}']
            for name in self.loss_obs.keys()
        ]

        mosaic_layout = [[f'loss_{self.name}', f'loss_{self.name}'], *mosaic_layout_children]

        fig = plt.figure(figsize=(14, 7), layout = 'constrained')  
        axs = fig.subplot_mosaic(mosaic_layout)

        losses = {self.name: self.losses, **{name: loss_ob.losses for name, loss_ob in self.loss_obs.items()}}
        for name, loss_tensor in losses.items():
            
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

    
#     def __call__(self, epoch: int, batch_idx: int, sample_loss_batches: Tensor):

#         start_idx = self.batch_size * batch_idx
#         end_idx = start_idx + self.batch_size

#         self.sample_losses[epoch, start_idx:end_idx] = sample_loss_batches.detach()