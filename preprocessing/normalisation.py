
import torch
import pandas as pd
import numpy as np
import yaml
import json

from torch import Tensor
from itertools import product
from pathlib import Path


"""
Normalisation - Min-Max
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def min_max_normalization(tensor: Tensor) -> Tensor:
    """
    Performs Min-Max normalization on a tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (m, n).

    Returns:
        torch.Tensor: The normalized tensor.
    """

    min_val = tensor.min(dim = 0, keepdim = True)[0]
    max_val = tensor.max(dim = 0, keepdim = True)[0]
    
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor



"""
Normalisation - Z-Score
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def z_score_normalization(tensor: Tensor) -> Tensor:
    """
    Performs Z-Score normalization (Standardization) on a tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (m, n).

    Returns:
        torch.Tensor: The normalized tensor.
    """
    
    mean = tensor.mean(dim = 0, keepdim = True)
    std = tensor.std(dim = 0, keepdim = True)
    
    normalized_tensor = (tensor - mean) / std
    
    return normalized_tensor



"""
Normalisation - Robust Scaling
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def robust_scaling(tensor: Tensor) -> Tensor:
    """
    Performs Robust Scaling on a tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (m, n).

    Returns:
        torch.Tensor: The normalized tensor.
    """
    
    median = tensor.median(dim = 0, keepdim = True)[0]
    iqr = tensor.quantile(0.75, dim = 0, keepdim = True) - tensor.quantile(0.25, dim=0, keepdim=True)
    
    normalized_tensor = (tensor - median) / iqr
    
    return normalized_tensor




"""
Normalisation - To apply to stored tensors
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def normalise_stored_tensors():

    tensor_one_mapping = True
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    X_data_tensor_name = 'X_data_tensor.pt'
    y_data_tensor_name = 'y_data_tensor.pt'

    X_data: Tensor = torch.load(tensor_dir / X_data_tensor_name)

    X_values = X_data[:, 1:] if tensor_one_mapping else X_data

    X_values_normalised = min_max_normalization(X_values)

    if tensor_one_mapping:
        X_data[:, 1:] = X_values_normalised

    else:
        X_data = X_values_normalised

    print(X_data[:10])

    name = X_data_tensor_name[:-3] + '_normalised.pt'
    torch.save(obj = X_data, f = tensor_dir / name)