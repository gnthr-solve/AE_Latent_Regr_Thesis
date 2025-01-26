
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable, Optional
from abc import ABC, abstractmethod


"""
Loss Term ABC
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossTerm(ABC):
    """
    LossTerm abstract base class (leaf) for Composite pattern.
    """
    @abstractmethod
    def __call__(self, **tensors: Tensor) -> Tensor:
        """
        Abstract method.
        Concrete LossTerm subclasses map batch tensors to loss batch, i.e.
            tensor (b, *dims) -> loss batch (b,)

        Parameters
        ----------
            **tensors: Tensor
                Tensors passed as keyword arguments.

        Returns:
        ----------
            loss_batch: Tensor
                Tensor of calculated sample-wise losses 
        """
        pass




"""
CompositeLossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class CompositeLossTerm(LossTerm):
    """
    CompositeLossTerm representing the composite in the LossTerm Composite pattern.
    """
    def __init__(self, loss_terms: dict[str, LossTerm] = {}):

        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:
        """
        Relays the call to registered loss terms and returns their sample-wise sum.

        Parameters
        ----------
            **tensors: Tensor
                Tensors passed as keyword arguments.

        Returns:
        ----------
            loss_batch: Tensor
                Sum of all calculated sample-wise losses produced by registered LossTerms
        """
        loss_batches = {
            name: loss_term(**tensors)
            for name, loss_term in self.loss_terms.items()
        }
        self.current_losses = loss_batches

        stacked_losses = torch.stack(tuple(loss_batches.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses


    def get_current_losses(self) -> dict[str, Tensor]:
        """
        Retrieves the individual loss components from the last forward pass.
        """
        return self.current_losses
    

    def add_term(self, name: str, loss_term: LossTerm):

        self.loss_terms[name] = loss_term



"""
Decorateable CompositeLossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class DecCompositeLossTerm(LossTerm):
    """
    CompositeLossTerm representing the composite in the LossTerm Composite pattern.
    Version conducts the result calculation of individual loss terms in the calc_component method,
    instead of directly in __call__, allowing decorators to intercept and modify behaviour.
    """
    def __init__(self, loss_terms: dict[str, LossTerm] = {}):

        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:
        """
        Relays the call to registered loss terms and returns their sample-wise sum.

        Parameters
        ----------
            **tensors: Tensor
                Tensors passed as keyword arguments.

        Returns:
        ----------
            loss_batch: Tensor
                Sum of all calculated sample-wise losses produced by registered LossTerms
        """
        loss_batches = {
            name: self.calc_component(name, **tensors)
            for name in self.loss_terms
        }

        stacked_losses = torch.stack(tuple(loss_batches.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses


    def calc_component(self, name: str, **tensors: Tensor) -> Tensor:
        """
        Calculate individual component loss. 
        Separate calculation method allows decorator integration 
        """
        return self.loss_terms[name](**tensors)




"""
Callback-CompositeLossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class CBCompositeLossTerm(LossTerm):
    """
    CompositeLossTerm representing the composite in the LossTerm Composite pattern.
    Version integrates a callback mechanism after the calculation of each loss term. 
    Callbacks could handle secondary tasks, like reporting to Ray Tune.
    """
    def __init__(
        self, 
        loss_terms: dict[str, LossTerm],
        callbacks: Optional[dict[str, list[Callable]]] = None,
        ):
        
        self.loss_terms = loss_terms
        self.callbacks = callbacks or {}


    def __call__(self, **tensors: Tensor) -> Tensor:

        loss_batches = {}

        for name, term in self.loss_terms.items():
            
            result = term(**tensors)
            loss_batches[name] = result
            
            # Apply callbacks registered for LossTerm name
            for callback in self.callbacks.get(name, []):
                callback(name, result)

        stacked_losses = torch.stack(tuple(loss_batches.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses


    def add_callback(self, name: str, callback: Callable):

        if name not in self.callbacks:
            self.callbacks[name] = []

        self.callbacks[name].append(callback)


