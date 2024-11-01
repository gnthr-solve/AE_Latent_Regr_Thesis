
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
Single-Term Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Loss:

    def __init__(self, loss_term: LossTerm):

        self.loss_term = loss_term

    
    def __call__(self, **tensors: Tensor) -> Tensor:
        
        return self.loss_term(**tensors).mean()



"""
CompositeLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class CompositeLossTerm(LossTerm):

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




class CompositeLossTermAlt(LossTerm):

    def __init__(self, **loss_terms: LossTerm):

        self.loss_terms = loss_terms
        

    def __call__(self, **tensors: Tensor) -> Tensor:

        loss_batches = []

        for name, loss_term in self.loss_terms.items():

            loss_term_batch = loss_term(**tensors)

            loss_batches.append(loss_term_batch)

        stacked_losses = torch.stack(loss_batches)
        batch_losses = torch.sum(stacked_losses, dim=0)

        return batch_losses




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


