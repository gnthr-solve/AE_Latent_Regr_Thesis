

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable
from abc import ABC, abstractmethod

from .loss_classes import LossTerm


"""
Loss
-------------------------------------------------------------------------------------------------------------------------------------------
Loss converts a single or composite LossTerm into an actual loss function, 
by aggregating the loss term values of a batch to a single scalar value.
"""
class Loss(LossTerm):

    def __init__(self, loss_term: LossTerm):

        self.loss_term = loss_term

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        return self.loss_term(**tensors).mean()



"""
Decorators - Weighted LossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Weigh(LossTerm):

    def __init__(self, loss_term: LossTerm, weight: float):

        self.loss_term = loss_term
        self.weight = weight

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        loss_batch = self.loss_term(**tensors)

        return self.weight * loss_batch
    


"""
Decorators - Observed LossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
CompositeLossTerm allows to register an Observer directly, however for a single LossTerm this is not a good approach.
Single LossTerms can instead be wrapped by the Observe decorator, allowing to track the loss values in a similar fashion.
"""
class Observe(LossTerm):

    def __init__(self, observer, loss_term: LossTerm):

        self.loss_term = loss_term
        self.observer = observer

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        loss_batch = self.loss_term(**tensors)

        self.notify_observer(loss_batch = loss_batch)
        
        return loss_batch
    

    def notify_observer(self, loss_batch: Tensor):

        batch_loss = loss_batch.detach()

        self.observer(batch_loss)



