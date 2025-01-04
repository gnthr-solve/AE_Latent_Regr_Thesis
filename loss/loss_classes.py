
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable
from abc import ABC, abstractmethod


"""
Loss Term ABC
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossTerm(ABC):

    @abstractmethod
    def __call__(self, **tensors: Tensor) -> Tensor:
        pass




"""
CompositeLossTerm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class CompositeLossTerm(LossTerm):

    def __init__(self, loss_terms: dict[str, LossTerm] = {}):

        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:

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
CompositeLossTermObs
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class CompositeLossTermObs(LossTerm):

    def __init__(self, observer = None, **loss_terms: LossTerm):

        self.observer = observer
        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:

        loss_batches = {
            name: loss_term(**tensors)
            for name, loss_term in self.loss_terms.items()
        }

        if self.observer is not None:
            self.notify_observer(loss_batches = loss_batches)

        #print(tuple(loss_batches.values()))
        stacked_losses = torch.stack(tuple(loss_batches.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses


    def notify_observer(self, loss_batches: dict[str, Tensor]):

        losses = {name: loss_batch.detach() for name, loss_batch in loss_batches.items()}

        self.observer(losses)

        # for loss_term in self.loss_terms.values():

        #     if isinstance(loss_term, CompositeLossTermObs):
        #         loss_term.notify_observer(loss_batches)


"""
CompositeLossTerm - Alternative Implementations
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class CompositeLossTermZeros(LossTerm):

    def __init__(self, **loss_terms: LossTerm):

        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:

        batch_losses = None

        for name, loss_term in self.loss_terms.items():

            loss_term_batch = loss_term(**tensors)

            # print(
            #     f'{name}:\n'
            #     f'-----------------------------------\n'
            #     f'shape: \n{loss_term_batch.shape}\n'
            #     f'values[:5]: \n{loss_term_batch[:5]}\n'
            #     f'-----------------------------------\n\n'
            # )

            if batch_losses is None:
                batch_losses = torch.zeros_like(loss_term_batch)

            batch_losses = batch_losses + loss_term_batch

        return batch_losses




class CompositeLossTermPrint(LossTerm):

    def __init__(self, print_losses: bool = False, **loss_terms: LossTerm):

        self.print_losses = print_losses
        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:

        loss_batches = {
            name: loss_term(**tensors)
            for name, loss_term in self.loss_terms.items()
        }

        if self.print_losses:
            self.print_loss_terms(loss_batches = loss_batches)

        stacked_losses = torch.stack(tuple(loss_batches.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses


    def print_loss_terms(self, loss_batches: dict[str, Tensor]):

        losses = {name: loss_batch.detach().mean() for name, loss_batch in loss_batches.items()}

        loss_strings = [f'{name}:\n{loss}\n' for name, loss in losses.items()]
        loss_str = f'--------------------------\n'.join(loss_strings)

        print(
            f'Total Loss: {sum(losses.values())}\n'
            f'--------------------------\n'
            f'{loss_str}\n'
        )




class CompositeLossTermMemory(LossTerm):

    def __init__(self, loss_terms: dict[str, LossTerm] = {}):
        self.loss_terms = loss_terms
        self.loss_values = {}

    def __call__(self, **tensors: Tensor) -> Tensor:
        self.loss_values = {
            name: loss_term(**tensors)
            for name, loss_term in self.loss_terms.items()
        }
        stacked_losses = torch.stack(tuple(self.loss_values.values()))
        batch_losses = torch.sum(stacked_losses, dim=0)
        return batch_losses

    def add_term(self, name: str, loss_term: LossTerm):
        self.loss_terms[name] = loss_term

    def get_individual_losses(self) -> dict[str, Tensor]:
        return self.loss_values