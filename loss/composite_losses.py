
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from typing import Callable


"""
Loss Functions - VAE-Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VAECompositeLoss:

    def __init__(self, reconstr_loss: Callable[[Tensor, Tensor], Tensor], kl_div_loss: Callable[[Tensor], Tensor]):

        self.reconstr_loss = reconstr_loss
        self.kl_div_loss = kl_div_loss
        

    def __call__(self, X_batch: Tensor, gen_model_params: Tensor, inference_model_params: Tensor) -> Tensor:
        
        reconstr_loss = self.reconstr_loss(X_batch, *gen_model_params)
        kl_div_loss = self.kl_div_loss(*inference_model_params)

        return - reconstr_loss + kl_div_loss 




"""
Loss Functions - Weighted Composite Loss
-------------------------------------------------------------------------------------------------------------------------------------------
Composite of regression and (deterministic) reconstruction loss
"""
class WeightedCompositeLoss:

    def __init__(self, loss_regr, loss_reconstr, w_regr, w_reconstr):
        
        self.loss_regr = loss_regr
        self.loss_reconstr = loss_reconstr

        self.w_regr = w_regr
        self.w_reconstr = w_reconstr


    def __call__(self, X_batch: Tensor, X_hat_batch: Tensor, y_batch: Tensor, y_hat_batch: Tensor) -> Tensor:

        reconstr_component = self.w_reconstr * self.loss_reconstr(X_batch, X_hat_batch)
        regr_component = self.w_regr * self.loss_regr(y_batch, y_hat_batch)
        
        return regr_component + reconstr_component
    
