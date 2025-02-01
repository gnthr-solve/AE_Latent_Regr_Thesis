

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable
from abc import ABC, abstractmethod

from .loss_term_classes import LossTerm



"""
Adapters - AE and Regression Adapters
-------------------------------------------------------------------------------------------------------------------------------------------
Some LossTerms like Lp losses can be used in either context and hence have signature t_batch, t_hat_batch.
To allow automatic assignment of arguments these adapters change the signature.
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
    


"""
Adapters - Auto Signature Adapter Idea
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AutoAdapter(LossTerm):

    def __init__(self, term: LossTerm, signature: dict[str, str]):

        self.term = term
        self.signature = signature
        

    def __call__(self, **tensors: Tensor):

        mapped = {k: tensors[v] for k,v in self.signature.items()}

        return self.term(**mapped)
