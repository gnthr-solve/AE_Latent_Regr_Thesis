
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
            tensors of shape (b, *dims) -> loss batch of shape (b,)

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
        Shares LossTerm signature, i.e.
            tensors of shape (b, *dims) -> loss batch (b,)

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
CompositeLossTermPrime
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class CompositeLossTermPrime(LossTerm):
    """
    CompositeLossTerm representing the composite in the LossTerm Composite pattern.
    Adapted to integrate callbacks and allows decorating the calculation of individual terms 
    via application to the composite from the outside.
    """
    def __init__(
        self, 
        loss_terms: dict[str, LossTerm],
        callbacks: Optional[dict[str, list[Callable]]] = None,
        ):
        
        self.loss_terms = loss_terms
        self.callbacks = callbacks or {}
        

    def __call__(self, **tensors: Tensor) -> Tensor:
        """
        Relays the call to registered loss terms and returns their sample-wise sum.
        Shares LossTerm signature, i.e.
            tensors of shape (b, *dims) -> loss batch (b,)
            
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
        Calculate individual component loss and apply callbacks. 
        Separate calculation method allows Composite decorator integration.
        """
        loss_batch = self.loss_terms[name](**tensors)

        for callback in self.callbacks.get(name, []):
            callback(name, loss_batch)

        return loss_batch


    def add_callback(self, name: str, callback: Callable):

        if name not in self.callbacks:
            self.callbacks[name] = []

        self.callbacks[name].append(callback)