
import torch
import pandas as pd
import numpy as np
import json

from torch import Tensor
from itertools import product
from pathlib import Path


"""
Normalisers - Min-Max
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MinMaxNormaliser:

    def __init__(self):

        self.min_val = None
        self.max_val = None


    def normalise(self, tensor: Tensor) -> Tensor:
        """
        Performs Min-Max normalisation on a tensor.

        Args:
            tensor (torch.Tensor): The input tensor of shape (m, n).

        Returns:
            torch.Tensor: The normalised tensor.
        """

        self.min_val = tensor.min(dim = 0, keepdim = True)[0]
        self.max_val = tensor.max(dim = 0, keepdim = True)[0]
        
        normalised_tensor = (tensor - self.min_val) / (self.max_val - self.min_val)
        
        return normalised_tensor


    def invert(self, normalised_tensor: Tensor) -> Tensor:

        if self.min_val is not None:

            tensor = normalised_tensor * (self.max_val - self.min_val) + self.min_val

            return tensor

        else:
            print('Cannot invert without previous normalise call.')



"""
Normalisers - Z-Score
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ZScoreNormaliser:

    def __init__(self):
        self.mean = None
        self.std = None


    def normalise(self, tensor: Tensor) -> Tensor:
        """
        Performs Z-Score normalisation (Standardization) on a tensor.

        Args:
            tensor (torch.Tensor): The input tensor of shape (m, n).

        Returns:
            torch.Tensor: The normalised tensor.
        """
        
        self.mean = tensor.mean(dim = 0, keepdim = True)
        self.std = tensor.std(dim = 0, keepdim = True)
        
        normalised_tensor = (tensor - self.mean) / self.std
        
        return normalised_tensor


    def invert(self, normalised_tensor: Tensor) -> Tensor:

        if self.mean is not None:

            tensor = normalised_tensor * self.std + self.mean

            return tensor
        
        else: 
            
            print('Cannot invert without previous normalise call.')




"""
Normalisation - Robust Scaling
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RobustScalingNormaliser:

    def __init__(self):
        self.median = None
        self.iqr = None


    def normalise(self, tensor: Tensor) -> Tensor:
        """
        Performs Robust Scaling on a tensor.

        Args:
            tensor (torch.Tensor): The input tensor of shape (m, n).

        Returns:
            torch.Tensor: The normalised tensor.
        """
        
        self.median = tensor.median(dim = 0, keepdim = True)[0]
        self.iqr = tensor.quantile(0.75, dim = 0, keepdim = True) - tensor.quantile(0.25, dim=0, keepdim=True)
        
        normalised_tensor = (tensor - self.median) / self.iqr
        
        return normalised_tensor
    

    def invert(self, normalised_tensor: Tensor) -> Tensor:

        if self.median is not None:

            tensor = normalised_tensor * self.iqr + self.median

            return tensor
        
        else: 
            
            print('Cannot invert without previous normalise call.')




