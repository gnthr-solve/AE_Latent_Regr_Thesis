
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

    def __init__(self, ll_loss: Callable[[Tensor, Tensor], Tensor], kl_div_loss: Callable[[Tensor], Tensor]):

        self.ll_loss = ll_loss
        self.kl_div_loss = kl_div_loss
        

    def __call__(self, X_batch: Tensor, gen_model_params: Tensor, inference_model_params: Tensor) -> Tensor:
        
        ll_loss = self.ll_loss(X_batch, *gen_model_params)
        kl_div_loss = self.kl_div_loss(*inference_model_params)

        # print(
        #     f'Loss: {- ll_loss + kl_div_loss}\n'
        #     f'----------------------------------------\n'
        #     f'Reconstruction Term:\n{-ll_loss}\n'
        #     f'----------------------------------------\n'
        #     f'KL-Divergence Term:\n{kl_div_loss}\n'
        #     f'----------------------------------------\n\n'
        # )
        return - ll_loss + kl_div_loss 




"""
Loss Functions - ELBOLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
from .vae_kld import AnalyticalKLDiv, MonteCarloKLDiv
from .vae_ll import LogLikelihood


class NegativeELBOLoss:

    def __init__(self, ll_term: LogLikelihood, kl_div_term: AnalyticalKLDiv | MonteCarloKLDiv):

        self.ll_term = ll_term
        self.kl_div_term = kl_div_term
        

    def __call__(self, X_batch: Tensor, Z_batch: Tensor, genm_dist_params: Tensor, infrm_dist_params: Tensor) -> Tensor:
        
        ll_term = self.ll_term(X_batch = X_batch, genm_dist_params = genm_dist_params)

        if isinstance(self.kl_div_term, AnalyticalKLDiv):
            kl_div_term = self.kl_div_term(infrm_dist_params = infrm_dist_params)
        
        else: 
            kl_div_term = self.kl_div_term(Z_batch = Z_batch, infrm_dist_params = infrm_dist_params)

        neg_elbo_loss = kl_div_term - ll_term

        print(
            f'Loss: {neg_elbo_loss}\n'
            f'----------------------------------------\n'
            f'Reconstruction Term:\n{-ll_term}\n'
            f'----------------------------------------\n'
            f'KL-Divergence Term:\n{kl_div_term}\n'
            f'----------------------------------------\n\n'
        )

        return neg_elbo_loss 




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
    
