
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps
from abc import ABC, abstractmethod

"""
Autoencoder Parameter Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class IterationObserver(ABC):

    def __init__(self, n_epochs, n_iterations):
        
        self.observed_metrics = torch.zeros(size = (n_epochs, n_iterations))
        

    @abstractmethod
    def __call__(self, epoch, iter_idx, model, loss):
        pass
        
    