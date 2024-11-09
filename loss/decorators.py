

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
class Weigh(LossTerm):

    def __init__(self, loss_term: LossTerm, weight: float):

        self.loss_term = loss_term
        self.weight = weight

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        loss_batch = self.loss_term(**tensors)

        return self.weight * loss_batch
    


class Observe(LossTerm):

    def __init__(self, observer, loss_term: LossTerm):

        self.loss_term = loss_term
        self.observer = observer

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        loss_batch = self.loss_term(**tensors)

        self.notify_observer(loss_batch = loss_batch)
        
        return loss_batch
    

    def notify_observer(self, loss_batch: Tensor):

        batch_loss = loss_batch.detach().mean()

        self.observer(batch_loss)



