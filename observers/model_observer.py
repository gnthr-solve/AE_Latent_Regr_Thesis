
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
from helper_tools import plot_training_losses, plot_param_norms, AbortTrainingError


"""
ModelObserver
-------------------------------------------------------------------------------------------------------------------------------------
Observer that monitors both the values and gradients of the parameters of a Module and its Submodules.
This can allow identifying trends and potential issues of exploding or vanishing gradients and values.

NOTE:
This Observer easily produces memory overflow, as, especially for large input sizes, the parameter values and gradients
of the NN layer weights can be too large to store for every iteration. 
Usage failed for the 'max' dataset where the input dimension was ~1300, but worked for 'key' dataset with input dimension ~180.
"""
class ModelObserver(IterObserver):

    def __init__(self, n_epochs: int, n_iterations: int, model: Module):

        self.submodel_param_values = {
            child_name: {
                name: torch.zeros(size = (n_epochs, n_iterations, *param.shape)) 
                for name, param in child.named_parameters()}
            for child_name, child in model.named_children()
        }

        self.submodel_param_grads = {
            child_name: {
                name: torch.zeros(size = (n_epochs, n_iterations, *param.shape)) 
                for name, param in child.named_parameters()}
            for child_name, child in model.named_children()
        }
    

    def __call__(self, epoch: int, iter_idx: int, model: Module, **kwargs):

        for child_name, child in model.named_children():
            
            #print(child_name)
            child_param_values = self.submodel_param_values[child_name]
            child_param_grads = self.submodel_param_grads[child_name]

            for name, param in child.named_parameters():
            
                #print(name)
                child_param_values[name][epoch, iter_idx] = param.data.clone()
                child_param_grads[name][epoch, iter_idx] = param.grad.clone()
                
                if torch.isnan(param.data).any():
                    print(f"{name} contains NaN values")
                    raise AbortTrainingError
                
                if torch.isinf(param.data).any():
                    print(f"{name} contains Inf values")
                    raise AbortTrainingError
        
    
    def plot_child_param_development(self, child_name: str, functional = torch.max):

        child_param_values = self.submodel_param_values[child_name]

        n_params = len(child_param_values)
        n_rows = int(n_params**0.5)  # Calculate the number of rows for the plot matrix
        n_cols = n_params // n_rows + (n_params % n_rows > 0)  # Calculate the number of columns

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle(f"Parameter Development for {child_name}", fontsize=16)

        for idx, (name, param_tensor) in enumerate(child_param_values.items()):

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

            ax.set_title(name)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Functional Value')

        # Hide any unused subplots
        if n_params > 1:
            for idx in range(n_params, n_rows * n_cols):
                axes.flatten()[idx].axis('off')

        plt.tight_layout()
        plt.show()


    # def apply_functional(self, param_tensor: Tensor, functional = torch.max):

    #     pass
