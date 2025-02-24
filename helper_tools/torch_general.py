
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
Torch General - Freeze & Unfreeze Parameters
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def freeze_params(model: Module):
    """
    Modifies a Module in place to disable param gradient tracking.
    """
    for param in model.parameters():
    
        param.requires_grad = False



def unfreeze_params(model: Module):
    """
    Modifies a Module in place to enable param gradient tracking.
    """
    for param in model.parameters():
    
        param.requires_grad = True



"""
Torch General - Retrieve non-NaN Batch Size
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def get_valid_batch_size(tensor: Tensor):
    """
    Calculates the effective Tensor size by ignoring NaN entries along the last dimension.
    """

    # Check for NaNs along the last dimension
    mask = torch.isnan(tensor).all(dim = -1)

    # Invert the mask to get valid entries, and count them
    valid_batch_size = (~mask).sum().item()

    return valid_batch_size




"""
Torch General - Constant Mask
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def constant_mask(tensor: Tensor, axis: int):

    min_vals = tensor.min(dim=axis, keepdim=True).values
    max_vals = tensor.max(dim=axis, keepdim=True).values

    constant_mask = (min_vals == max_vals).squeeze(axis)
    
    return constant_mask