
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .iter_observer import IterObserver

"""
VAELatentObserver
-------------------------------------------------------------------------------------------------------------------------------------------
Designed to track the distribution parameters that are output by the inference model of a VAE.
"""
class VAELatentObserver(IterObserver):

    def __init__(self, n_epochs: int, dataset_size: int, batch_size: int, latent_dim: int, n_dist_params: int):

        self.dist_params = torch.zeros(size = (n_epochs, dataset_size, latent_dim, n_dist_params)) 
        self.batch_size = batch_size


    def __call__(self, epoch: int, iter_idx: int, infrm_dist_params: Tensor, **kwargs):

        batch_dist_params = infrm_dist_params.detach()

        start_idx = self.batch_size * iter_idx
        end_idx = self.batch_size * (iter_idx + 1)

        self.dist_params[epoch, start_idx:end_idx] = batch_dist_params
        
    
    def plot_dist_params(self, functional = torch.max):

        dist_params = self.dist_params.unbind(dim = -1)

        n_params = len(dist_params)
        n_rows = int(n_params**0.5)  # Calculate the number of rows for the plot matrix
        n_cols = n_params // n_rows + (n_params % n_rows > 0)  # Calculate the number of columns

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle(f"Distribution Parameter Development", fontsize=16)

        for idx, param_tensor in enumerate(dist_params):

            ax = axes.flatten()[idx] if n_params > 1 else axes  # Handle single parameter case

            n_iter_total = param_tensor.shape[0] * param_tensor.shape[1]
            flattened_param_tensor = param_tensor.flatten(start_dim = 0, end_dim = 1)

            param_values = torch.tensor([
                functional(param).item() 
                for param in flattened_param_tensor
            ])


            
            iterations = len(param_values)
            ax.plot(range(iterations), param_values)

            # Add vertical lines for each epoch
            epochs = param_tensor.shape[0]
            iterations_per_epoch = param_tensor.shape[1]
            for epoch in range(1, epochs):
                ax.axvline(x = epoch * iterations_per_epoch, color = 'r', linestyle = '--')

            ax.set_title(f'Params {idx}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Functional Value')

        # Hide any unused subplots
        if n_params > 1:
            for idx in range(n_params, n_rows * n_cols):
                axes.flatten()[idx].axis('off')

        plt.tight_layout()
        plt.show()


    def plot_dist_params_batch(self, functional: Callable[[Tensor], Tensor] = torch.max):

        dist_params = self.dist_params.unbind(dim = -1)

        n_params = len(dist_params)
        n_rows = int(n_params**0.5)  # Calculate the number of rows for the plot matrix
        n_cols = n_params // n_rows + (n_params % n_rows > 0)  # Calculate the number of columns

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle(f"Distribution Parameter Development", fontsize=16)

        for idx, param_tensor in enumerate(dist_params):

            ax: Axes = axes.flatten()[idx] if n_params > 1 else axes  # Handle single parameter case

            n_epochs = param_tensor.shape[0]
            size_dataset = param_tensor.shape[1]
            max_batch_idx = n_epochs * size_dataset // self.batch_size

            flattened_param_tensor = param_tensor.flatten(start_dim = 0, end_dim = 1)
            
            sample_param_values = torch.tensor([
                functional(param).item() 
                for param in flattened_param_tensor
            ])

            batched_values = [
                sample_param_values[i*self.batch_size : (i+1)*self.batch_size] 
                for i in range(max_batch_idx)
            ]
            batched_values.append(sample_param_values[max_batch_idx * self.batch_size: -1])

            batch_values_mean = torch.tensor([batch_values.mean() for batch_values in batched_values])
            batch_values_std = torch.tensor([batch_values.std() for batch_values in batched_values])

            total_iterations = len(batched_values)
            ax.plot(range(total_iterations), batch_values_mean)

            # Add vertical lines for each epoch
            epochs = n_epochs
            iterations_per_epoch = len(batched_values) / epochs
            for epoch in range(1, epochs):
                ax.axvline(x = epoch * iterations_per_epoch, color = 'r', linestyle = '--')

            ax.fill_between(
                range(total_iterations), 
                batch_values_mean - batch_values_std, 
                batch_values_mean + batch_values_std, 
                alpha=0.2, 
                color='gray',
            )

            ax.set_title(f'Params {idx}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Functional Value')

        # Hide any unused subplots
        if n_params > 1:
            for idx in range(n_params, n_rows * n_cols):
                axes.flatten()[idx].axis('off')

        plt.tight_layout()
        plt.show()




"""
LatentVariableObserver
-------------------------------------------------------------------------------------------------------------------------------------------
Designed to track the latent variables that are put out by an encoder model.
The motivation is to be able to investigate the properties of the latent space.
"""
class LatentVariableObserver(IterObserver):

    def __init__(self, n_epochs: int, dataset_size: int, batch_size: int, latent_dim: int):

        self.dist_params = torch.zeros(size = (n_epochs, dataset_size, latent_dim)) 
        self.batch_size = batch_size


    def __call__(self, epoch: int, iter_idx: int, Z_batch: Tensor, **kwargs):

        start_idx = self.batch_size * iter_idx
        end_idx = self.batch_size * (iter_idx + 1)

        self.dist_params[epoch, start_idx:end_idx] = Z_batch.detach()