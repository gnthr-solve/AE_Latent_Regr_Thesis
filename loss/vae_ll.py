
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from abc import ABC, abstractmethod
from .loss_term_classes import LossTerm

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
        """
        Log-Likelihood calculation in the case where the variance of a Gaussian generative model is unit matrix I.
        """
        mu = genm_dist_params
    
        diff_mean = (X_batch - mu).pow(2)

        adj_mean_sums = diff_mean.sum(dim = -1)

        ll_batch = -0.5 * (adj_mean_sums)

        return ll_batch
    



"""
Reconstruction Term for Gaussians - GaussianDiagRLT
-------------------------------------------------------------------------------------------------------------------------------------------
Loss term for a generative model with a Gaussian distribution, 
where both mu and a diagonal covariance are learnable by the decoder.
"""
class GaussianDiagLL(LogLikelihood):

    def __init__(self, receives_logvar: bool = True):
        self.receives_logvar = receives_logvar


    def __call__(self, X_batch: Tensor, genm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        """
        Log-Likelihood calculation for decoder generative model parameters consist of
        mean and diagonal variance matrix, where the variance component is hence a vector.
            genm_dist_params[..., 0] <--> mean 
            genm_dist_params[..., 1] <--> logvar | sigma    (log of variance or std)

        Args:
            X_batch: Tensor
                Input data, shape (b, d)
            genm_dist_params: Tensor
                Generative model dist. params, shape (b, d, 2)
        """
        mu, var, logvar = self._get_params(genm_dist_params = genm_dist_params)

        diff_mean = (X_batch - mu).pow(2)
        
        adj_mean_sums = (diff_mean / var + logvar).sum(dim = -1)
        #adj_mean_sums = (diff_mean / var + var).sum(dim = -1)

        ll_batch = -0.5 * (adj_mean_sums)
        
        return ll_batch.squeeze()
    

    def _get_params(self, genm_dist_params: Tensor):
        """
        Unpacks and returns mean, variance and log of variance based on two cases:
            1. Decoders genm_dist_params[..., 1] = logvar (i.e. log(sigma^2))
            2. Decoders genm_dist_params[..., 1] = sigma 
        
        Args:
            genm_dist_params: Tensor
                Generative model dist. params, shape (b, d, 2)
        """
        if self.receives_logvar:
            mu, logvar = genm_dist_params.unbind(dim = -1)
            var = torch.exp(logvar)
            
        else:
            mu, sigma = genm_dist_params.unbind(dim = -1)
            var = sigma**2
            logvar = 2 * torch.log(sigma)
        
        return mu, var, logvar




"""
Reconstruction Term for Independent Beta
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class IndBetaLL(LogLikelihood):

    def __call__(self, X_batch: Tensor, genm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        
        logalpha, logbeta = genm_dist_params.unbind(dim = -1)
        alpha, beta = torch.exp(logalpha), torch.exp(logbeta)
        
        const_comp = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        
        ll_batch_summands = const_comp + (alpha - 1) * torch.log(X_batch) + (beta - 1) * torch.log(1 - X_batch)
        
        ll_batch = ll_batch_summands.sum(dim = -1)

        return ll_batch.squeeze()
    
