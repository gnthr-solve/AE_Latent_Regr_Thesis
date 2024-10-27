
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure


"""
VAE Latent Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VAELatentObserver:

    def __init__(self, n_epochs: int, dataset_size: int, batch_size: int, latent_dim: int, n_dist_params: int):

        self.dist_params = torch.zeros(size = (n_epochs, dataset_size, latent_dim, n_dist_params)) 
        self.batch_size = batch_size


    def __call__(self, epoch: int, iter_idx: int, infrm_dist_params: Tensor):

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

            param_values = torch.tensor([
                functional(param).item() 
                for param in param_tensor.flatten(start_dim = 0, end_dim = 1)
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