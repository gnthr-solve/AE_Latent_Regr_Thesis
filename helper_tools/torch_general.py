
"""
Torch General Helper Tools (GHTs) - DataFrameNamedTupleDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch.nn import Module
from torch import Tensor

from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

"""
Torch GHTs - Freeze & Unfreeze Parameters
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def freeze_params(model: Module):

    for param in model.parameters():
    
        param.requires_grad = False



def unfreeze_params(model: Module):

    for param in model.parameters():
    
        param.requires_grad = True



"""
Torch GHTs - Retrieve non-NaN Batch Size
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def get_valid_batch_size(tensor):
    """
    Calculates the effective Tensor size by ignoring NaN entries along the last dimension.
    """

    # Check for NaNs along the last dimension
    mask = torch.isnan(tensor).all(dim = -1)

    # Invert the mask to get valid entries, and count them
    valid_batch_size = (~mask).sum().item()

    return valid_batch_size



"""
Torch GHTs - Model Summary Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def get_model_summary(model: Module):

    
    for name, param in model.named_parameters():
        print(f'{name} value:\n {param.data}')
        print(f'{name} grad:\n {param.grad}')


def get_parameter_summary(model: Module):

    for name, param in model.named_parameters():

        print(f'{name} max value:\n {param.data}')
        print(f'{name} min value:\n {param.data}')
        print(f'{name} max grad:\n {param.grad}')
        print(f'{name} min grad:\n {param.grad}')


"""
Torch GHTs - Tensor list to Numpy array
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def tensor_list_to_numpy_array(tensor_list: list[torch.Tensor]) -> np.ndarray:
    
    np_array = torch.stack(tensor_list).numpy()

    return np_array


"""
Torch GHTs - Plotting Functions - AEParameterObserver
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_training_losses(
    losses: list,
    axes: Axes = None,
    title: str = "Training Losses",
    xlabel: str = "Iterations",
    ylabel: str = "Loss",
    legend: str = "Loss",
    color: str = "blue",
    linestyle: str = "-",
    marker: str = "o",
    ):

    #losses = tensor_list_to_numpy_array(losses)

    axes.plot(losses, color = color, linestyle = linestyle, marker = marker, label = legend)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.legend()

    return axes


def plot_param_norms(
    norms: dict[str, list],
    axes: Axes = None,
    kind: str = "value",
    linestyle: str = "-",
    marker: str = "o",
    ):

    for param_name, norms in norms.items():

        label = f"{param_name}"
        
        #norms = tensor_list_to_numpy_array(norms)
        #print(f"Norms: {norms}")
        axes.plot(norms, linestyle = linestyle, marker = marker, label = label)

    title = f"Parameter {kind} Norms"
    xlabel = "Iterations"
    ylabel = "Norm value"
    
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.legend()

    return axes


def plot_training_characteristics(
    losses: list,
    value_norms: dict[str, list],
    grad_norms: dict[str, list],
    title: str = "Training Characteristics",
    ):

    fig, axes = plt.subplots(3, 1)

    axes[0] = plot_training_losses(losses, axes[0])
    axes[1] = plot_param_norms(value_norms, axes[1], kind = "value")
    axes[2] = plot_param_norms(grad_norms, axes[2], kind = "grad")

    fig.suptitle(title)

    plt.show()




"""
Torch GHTs - plot_loss_tensor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_loss_tensor(observed_losses: Tensor):

    epochs, iterations_per_epoch = observed_losses.shape
    iterations = epochs * iterations_per_epoch

    losses = observed_losses.flatten()
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), losses)

    # Add vertical lines for each epoch
    for epoch in range(1, epochs):
        plt.axvline(x = epoch * iterations_per_epoch, color = 'b', linestyle = '--')

    # Set the plot labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
