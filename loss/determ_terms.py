
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from .loss_classes import LossTerm

"""
Loss Functions - MeanLpLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LpNorm(LossTerm):

    def __init__(self, p: int):
        self.p = p


    def __call__(self, X_batch: Tensor, X_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diff = X_batch - X_hat_batch

        diff_norms: Tensor = tla.norm(diff, ord = self.p, dim = 1)

        return diff_norms
    



class RelativeLpNorm(LossTerm):

    def __init__(self, p: int):
        self.p = p


    def __call__(self, X_batch: Tensor, X_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diff = X_batch - X_hat_batch

        X_batch_norm_mean = self.norm(X_batch).mean()
        
        diff_norm = self.norm(diff) 

        return diff_norm / X_batch_norm_mean
    

    def norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = self.p, dim = 1)
        
        return batch_norms
    


"""
Loss Functions - HuberLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Huber(LossTerm):

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction = None, delta = delta)


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor, **tensors: Tensor) -> Tensor:

        loss_batch = self.loss_fn(t_in_batch, t_out_batch)

        return loss_batch
    


class RelativeHuber(LossTerm):

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction = None, delta = delta)


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor, **tensors: Tensor) -> Tensor:

        t_in_batch_norms = self.norm(t_batch = t_in_batch)

        loss_batch = self.loss_fn(t_in_batch, t_out_batch) / t_in_batch_norms.mean()

        return loss_batch
    

    def norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = 2, dim = 1)
        
        return batch_norms
    
