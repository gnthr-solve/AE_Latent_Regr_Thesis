

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable
from abc import ABC, abstractmethod

from .loss_classes import LossTerm


"""
Decorators - Weighted Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class WeightedLossTerm(LossTerm):

    def __init__(self, loss_term: LossTerm, weight: float):

        self.loss_term = loss_term
        self.weight = weight

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        batch_loss = self.loss_term(**tensors)

        return self.weight * batch_loss
    


"""
Adapters - AE and Regression Adapters
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class AEAdapter(LossTerm):

    def __init__(self, loss_term: LossTerm):

        self.loss_term = loss_term

    
    def __call__(self, X_batch: Tensor, X_hat_batch: Tensor, **tensors: Tensor) -> Tensor:
        
        return self.loss_term(t_batch = X_batch, t_hat_batch = X_hat_batch, **tensors)
    



class RegrAdapter(LossTerm):

    def __init__(self, loss_term: LossTerm):

        self.loss_term = loss_term

    
    def __call__(self, y_batch: Tensor, y_hat_batch: Tensor, **tensors: Tensor) -> Tensor:
        
        return self.loss_term(t_batch = y_batch, t_hat_batch = y_hat_batch, **tensors)