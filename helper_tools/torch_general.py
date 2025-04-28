
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
    """
    Creates a mask for a tensor along a specified dimension, 
    that is True where the tensor is constant along that dimension and False otherwise.
    """
    min_vals = tensor.min(dim=axis, keepdim=True).values
    max_vals = tensor.max(dim=axis, keepdim=True).values

    constant_mask = (min_vals == max_vals).squeeze(axis)
    
    return constant_mask



"""
Torch General - Padding (& Causal) Mask
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def create_padding_mask(lengths: Tensor, is_causal: bool = False) -> Tensor:
        """
        Assume lengths is batch of lengths
        """
        batch_size = lengths.size(0)
        max_len = lengths.max().item()

        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)

        # Create validity mask [batch, max_len] 
        # True where position < length, False otherwise
        valid_positions = positions < lengths.unsqueeze(1)
        
        # Which token can pose a query - padding cannot
        # First expansion: [batch, max_len, 1]
        # Second expansion: [batch, max_len, max_len]
        query_mask = valid_positions.unsqueeze(2).expand(-1, -1, max_len)
        # Which token can provide a key - padding cannot
        key_mask = valid_positions.unsqueeze(1).expand(-1, max_len, -1)

        attention_mask = query_mask & key_mask

        if is_causal:
            # Later tokens cannot influence earlier ones
            # Creates lower triangular matrix (assumes $QK^T$)
            causal_mask = torch.tril(torch.ones(max_len, max_len)).bool()
            return attention_mask & ~causal_mask
        
        else:
             return attention_mask