
"""
Torch General Helper Tools (GHTs)
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np
import re

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
Torch GHTs - Initialise weights
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def initialize_weights(model: Module):

    for m in model.modules():
        
        if isinstance(m, nn.Linear):

            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)




