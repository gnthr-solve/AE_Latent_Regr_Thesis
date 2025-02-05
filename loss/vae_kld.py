
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from abc import ABC, abstractmethod
from .loss_term_classes import LossTerm

"""
Analytical KL-Divergence - ABC
-------------------------------------------------------------------------------------------------------------------------------------------
The KL-divergence loss term calculates the divergence from the inference model
"""
class AnalyticalKLDiv(LossTerm):

    @abstractmethod
    def __call__(self, infrm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        pass
    



"""
Monte Carlo KL Divergence - MCGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Monte Carlo estimate for KL-divergence loss term for Gaussian inference model and standard Gaussian prior,
following the same principle as for the reconstruction
"""
class MonteCarloKLDiv(LossTerm):

    @abstractmethod
    def __call__(self, Z_batch: Tensor, infrm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        pass




"""
KL Divergence - GaussianAnaKLDiv
-------------------------------------------------------------------------------------------------------------------------------------------
Analytical KL-divergence loss term for Gaussian inference model and standard Gaussian prior
"""
class GaussianAnaKLDiv(AnalyticalKLDiv):

    def __init__(self, receives_logvar: bool = True):
        self.receives_logvar = receives_logvar


    def __call__(self, infrm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        """
        Analytical KL-Divergence calculation for encoder inference model parameters consisting of
        mean and diagonal variance matrix, where the variance component is hence taken as a vector.
            infrm_dist_params[..., 0] <--> mean 
            infrm_dist_params[..., 1] <--> logvar | sigma    (log of variance or std)

        Args:
            infrm_dist_params: Tensor
                Inference model dist. params, shape (b, d, 2)
        """
        mu, var, logvar = self._get_params(infrm_dist_params = infrm_dist_params)

        kld_batch = 0.5 * (-1 - logvar + mu.pow(2) + var).sum(dim = -1)
        
        return kld_batch
    

    def _get_params(self, infrm_dist_params: Tensor):
        """
        Unpacks and returns mean, variance and log of variance based on two cases:
            1. Encoders infrm_dist_params[..., 1] = logvar (i.e. log(sigma^2))
            2. Encoders infrm_dist_params[..., 1] = sigma 
        
        Args:
            infrm_dist_params: Tensor
                Inference model dist. params, shape (b, d, 2)
        """
        if self.receives_logvar:
            mu, logvar = infrm_dist_params.unbind(dim = -1)
            var = torch.exp(logvar)
            
        else:
            mu, sigma = infrm_dist_params.unbind(dim = -1)
            var = sigma**2
            logvar = 2 * torch.log(sigma)
        
        return mu, var, logvar


"""
KL Divergence - MCGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Monte Carlo estimate for KL-divergence loss term for Gaussian inference model and standard Gaussian prior,
following the same principle as for the reconstruction
"""
class GaussianMCKLDiv(MonteCarloKLDiv):

    def __init__(self, receives_logvar: bool = True):
        self.receives_logvar = receives_logvar


    def __call__(self, Z_batch: Tensor, infrm_dist_params: Tensor, **tensors: Tensor) -> Tensor:
        """
        Monte-Carlo KL-Divergence calculation for encoder inference model parameters consisting of
        mean and diagonal variance matrix, where the variance component is hence taken as a vector.
            infrm_dist_params[..., 0] <--> mean 
            infrm_dist_params[..., 1] <--> logvar | sigma    (log of variance or std)

        Args:
            Z_batch: Tensor
                Tensor of latent representations obtained via the reparameterisation trick.
            infrm_dist_params: Tensor
                Inference model dist. params, shape (b, d, 2)
        """
        mu, var, logvar = self._get_params(infrm_dist_params = infrm_dist_params)

        sq_diff_means = (Z_batch - mu).pow(2)
        sq_Z_batch = Z_batch.pow(2)

        sq_mean_deviations = (sq_diff_means / var)

        kld_summands = sq_mean_deviations + logvar - sq_Z_batch
        #kld_summands = sq_mean_deviations + var - sq_Z_batch
        
        kld_batch = -0.5 * kld_summands.sum(dim = -1)

        # if kld_batch.ndim == 0:
        #     kld_batch = kld_batch.unsqueeze(0)

        return kld_batch


    def _get_params(self, infrm_dist_params: Tensor):
        """
        Unpacks and returns mean, variance and log of variance based on two cases:
            1. Encoders infrm_dist_params[..., 1] = logvar (i.e. log(sigma^2))
            2. Encoders infrm_dist_params[..., 1] = sigma 
        
        Args:
            infrm_dist_params: Tensor
                Inference model dist. params, shape (b, d, 2)
        """
        if self.receives_logvar:
            mu, logvar = infrm_dist_params.unbind(dim = -1)
            var = torch.exp(logvar)
            
        else:
            mu, sigma = infrm_dist_params.unbind(dim = -1)
            var = sigma**2
            logvar = 2 * torch.log(sigma)
        
        return mu, var, logvar
    



"""
Alternative Monte Carlo KL Divergence - GaussianMCKLDiv
-------------------------------------------------------------------------------------------------------------------------------------------
Analogous, but intended to be more numerically stable
"""
MAX_LOGVAR = 10
class GaussianMCKLDivClamp(MonteCarloKLDiv):

    def __call__(self, Z_batch: Tensor, infrm_dist_params: Tensor, **tensors: Tensor) -> Tensor:

        mu, logvar = infrm_dist_params.unbind(dim=-1)
        var = torch.exp(logvar.clamp(max=MAX_LOGVAR))
        
        # Use log-sum-exp trick for numerical stability
        log_ratio = -0.5 * (
            (Z_batch - mu).pow(2) / var + logvar - Z_batch.pow(2)
        )

        return torch.logsumexp(log_ratio, dim=-1)
