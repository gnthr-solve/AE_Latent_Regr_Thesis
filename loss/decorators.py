

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable, Any
from abc import ABC, abstractmethod

from observers.loss_observer import LossTermObserver, ComposedLossTermObserver

from .loss_classes import LossTerm, CompositeLossTerm


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





"""
Decorators - Observed LossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
CompositeLossTerm allows to register an Observer directly, however for a single LossTerm this is not a good approach.
Single LossTerms can instead be wrapped by the Observe decorator, allowing to track the loss values in a similar fashion.
"""
class ObserveComposite(LossTerm):

    def __init__(self, name: str, loss_term: LossTerm | CompositeLossTerm, observer_kwargs: dict[str, Any]):

        self.loss_term = loss_term

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        loss_batch = self.loss_term(**tensors)

        self.notify_observer(loss_batch = loss_batch)
        
        return loss_batch
    

    def construct_observer(self, name: str, loss_term: LossTerm | CompositeLossTerm, observer_kwargs: dict[str, Any]):

        self.observers = {}

        # if isinstance(loss_term, CompositeLossTerm):
        #     members = list(loss_term.loss_terms.keys())
        #     self.observers = {name: CompositeLossTermObserver(name = name, members=members, **observer_kwargs)}
        
        # else:
        #     self.observers = {name: LossTermObserver(name = name, **observer_kwargs)}


    def construct_composite_observer(self, name: str, loss_term: CompositeLossTerm, observer_kwargs: dict[str, Any]):

        clt_observers = {}

        for name, lt in loss_term.loss_terms.items():

            if isinstance(loss_term, CompositeLossTerm):
                observers = self.construct_composite_observer(name = name, loss_term=lt, observer_kwargs=observer_kwargs)
                clt_observers[name] = ComposedLossTermObserver(name = name, loss_obs = observers, **observer_kwargs)
            
            else:
                observer = LossTermObserver(name = name, **observer_kwargs)
                clt_observers[name] = observer
                lt = Observe(observer = observer, loss_term = lt) 

        return clt_observers



