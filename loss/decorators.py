

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable, Any
from abc import ABC, abstractmethod

from observers.loss_observer_fixed import LossTermObserver, ComposedLossTermObserver

from .loss_term_classes import LossTerm, CompositeLossTerm


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
    



class WeightedCompositeLoss(CompositeLossTerm):

    def __init__(self, composite_lt: CompositeLossTerm, weights: dict[str, float]):

        self.composite_lt = composite_lt
        self.loss_terms = composite_lt.loss_terms
        self.callbacks = composite_lt.callbacks
        
        self.weights = weights


    def calc_component(self, name: str, **tensors: Tensor) -> Tensor:

        loss_batch = self.composite_lt.calc_component(name, **tensors)

        return self.weights.get(name, 1.0) * loss_batch
    



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




class ObserveComponent(CompositeLossTerm):
    """
    Allows attaching an observer to a CompositeLossTerm from the outside,
    hence avoiding wrapping the LossTerm at setup.
    """
    def __init__(self, composite: CompositeLossTerm, observer, target_names: list[str]):
        
        self.target_names = target_names

        self.composite = composite
        self.loss_terms = composite.loss_terms

        self.observer = observer
        
    
    def calc_component(self, name: str, **tensors: Tensor) -> Tensor:

        result = super().calc_component(name, **tensors)

        if name in self.target_names:
            self.observer(result)

        return result




"""
Decorators - Normalised Loss Term
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class NormalisedLossTerm(LossTerm):

    def __init__(self, loss_term: LossTerm):

        self.loss_term = loss_term

        self.running_mean = 0
        self.running_var = 1
        # Smoothing factor
        self.alpha = 0.99  


    def __call__(self, **tensors: Tensor) -> Tensor:

        loss_batch = self.loss_term(**tensors)

        batch_mean = loss_batch.mean().item()
        batch_var = loss_batch.var().item()

        # Update running statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * batch_var

        # Normalise loss
        normalised_loss = (loss_batch - self.running_mean) / (self.running_var ** 0.5 + 1e-8)

        return normalised_loss
