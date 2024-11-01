
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from abc import ABC, abstractmethod
from .loss_classes import LossTerm

"""
Reconstruction Loss - ABC
-------------------------------------------------------------------------------------------------------------------------------------------
The reconstruction loss term must minimise the negative log-likelihood of the input data, 
given the distribution parameters of the generative model, produced by the decoder.
It is a single-sample Monte Carlo estimate.
"""
class LogLikelihood(LossTerm):

    @abstractmethod
    def __call__(self, X_batch: Tensor, genm_dist_params: Tensor, **tensors: Tensor) -> Tensor:

        pass




"""
Reconstruction Term for Gaussians - GaussianUnitVarRLT
-------------------------------------------------------------------------------------------------------------------------------------------
Loss term for a generative model with a Gaussian distribution, 
where both mu and a diagonal covariance are learnable by the decoder.
"""
class GaussianUnitVarLL(LogLikelihood):

    def __call__(self, X_batch: Tensor, genm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        
        mu = genm_dist_params
    
        diff_mean = (X_batch - mu).pow(2)

        adj_mean_sums = diff_mean.sum(dim = 1)

        ll_batch = -0.5 * (adj_mean_sums)

        return ll_batch
    



"""
Reconstruction Term for Gaussians - GaussianDiagRLT
-------------------------------------------------------------------------------------------------------------------------------------------
Loss term for a generative model with a Gaussian distribution, 
where both mu and a diagonal covariance are learnable by the decoder.
"""
class GaussianDiagLL(LogLikelihood):

    def __call__(self, X_batch: Tensor, genm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        
        mu, logvar = genm_dist_params.unbind(dim = -1)
    
        diff_mean = (X_batch - mu).pow(2)
        var = torch.exp(logvar)

        adj_mean_sums = (diff_mean / var + logvar).sum(dim = 1)
        #adj_mean_sums = (diff_mean / var + var).sum(dim = 1)

        ll_batch = -0.5 * (adj_mean_sums)

        return ll_batch
    
