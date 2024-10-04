
"""
Loss Functions - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn




"""
Loss Functions - Simple Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss()


    def forward(self, x: Tensor, x_hat: Tensor):

        loss = self.loss_fn(x, x_hat)

        return loss
    



"""
Loss Functions - Composite Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class WeightedCompositeLoss:

    def __init__(self, loss_regr, loss_reconstr, w_regr, w_reconstr):
        
        self.loss_regr = loss_regr
        self.loss_reconstr = loss_reconstr

        self.w_regr = w_regr
        self.w_reconstr = w_reconstr


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor):

        regr_component = self.w_regr * self.loss_regr(t_in_batch, t_out_batch)
        reconstr_component = self.w_reconstr * self.loss_reconstr(t_in_batch, t_out_batch)

        return regr_component + reconstr_component
    



"""
Loss Functions - MeanLpLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MeanLpLoss:

    def __init__(self, p: int):
        self.p = p


    def __call__(self, x_batch: Tensor, x_hat_batch: Tensor):

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
    

    def mean_norm(self, t_batch: Tensor):

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


    def __call__(self, t_in_batch: Tensor, t_out_batch: Tensor):

        loss = self.loss_fn(t_in_batch, t_out_batch)

        return loss