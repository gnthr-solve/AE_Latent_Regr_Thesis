
"""
Torch General Helper Tools (GHTs) - DataFrameNamedTupleDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch

from torch.nn import Module

from itertools import product
from functools import wraps


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