
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn



"""
Loss Functions - MeanLpLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MeanLpLoss:

    def __init__(self, p: int):
        self.p = p


    def __call__(self, x_batch: Tensor, x_hat_batch: Tensor) -> Tensor:

        diff = x_batch - x_hat_batch

        diff_norms: Tensor = tla.norm(diff, ord = self.p, dim = 1)

        mean_norm = diff_norms.mean()

        return mean_norm
    



class RelativeMeanLpLoss:

    def __init__(self, p: int):
        self.p = p


    def __call__(self, x_batch: Tensor, x_hat_batch: Tensor):

        diff = x_batch - x_hat_batch

        mean_x_batch_norm = self.mean_norm(x_batch)
        mean_diff_norm = self.mean_norm(diff)

        return mean_diff_norm / mean_x_batch_norm
    

    def mean_norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = self.p, dim = 1)
        
        return batch_norms.mean()
    


"""
Loss Functions - HuberLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class HuberLoss:

    reduction = 'mean'

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction =  self.reduction, delta = delta)


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor) -> Tensor:

        loss = self.loss_fn(t_in_batch, t_out_batch)

        return loss
    


class RelativeHuberLoss:

    reduction = 'mean'

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction =  self.reduction, delta = delta)


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor) -> Tensor:

        loss = self.loss_fn(t_in_batch, t_out_batch) / self.mean_norm(t_batch = t_in_batch)

        return loss
    

    def mean_norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = 2, dim = 1)
        
        return batch_norms.mean()
    
