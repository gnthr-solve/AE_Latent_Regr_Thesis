
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from abc import ABC, abstractmethod


"""
Analytical KL-Divergence - ABC
-------------------------------------------------------------------------------------------------------------------------------------------
The KL-divergence loss term calculates the divergence from the inference model
"""
class AnalyticalKLDiv(ABC):

    @abstractmethod
    def __call__(self, infrm_dist_params: Tensor) -> Tensor:
        pass
    



"""
Monte Carlo KL Divergence - MCGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Monte Carlo estimate for KL-divergence loss term for Gaussian inference model and standard Gaussian prior,
following the same principle as for the reconstruction
"""
class MonteCarloKLDiv(ABC):

    @abstractmethod
    def __call__(self, Z_batch: Tensor, infrm_dist_params: Tensor) -> Tensor:
        pass




"""
KL Divergence - GaussianAnaKLDiv
-------------------------------------------------------------------------------------------------------------------------------------------
Analytical KL-divergence loss term for Gaussian inference model and standard Gaussian prior
"""
class GaussianAnaKLDiv(AnalyticalKLDiv):

    def __call__(self, infrm_dist_params: Tensor) -> Tensor:

        mu , logvar = infrm_dist_params.unbind(dim = -1)

        kld_batch = 0.5 * (-1 - logvar + mu.pow(2) + torch.exp(logvar)).sum(dim = 1)

        total_kld = kld_batch.mean()

        return total_kld
    



"""
KL Divergence - MCGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Monte Carlo estimate for KL-divergence loss term for Gaussian inference model and standard Gaussian prior,
following the same principle as for the reconstruction
"""
class GaussianMCKLDiv(MonteCarloKLDiv):

    def __call__(self, Z_batch: Tensor, infrm_dist_params: Tensor) -> Tensor:

        mu, logvar = infrm_dist_params.unbind(dim = -1)

        sq_diff_means = (Z_batch - mu).pow(2)
        sq_Z_batch = Z_batch.pow(2)
        var = torch.exp(logvar)

        sq_mean_deviations = (sq_diff_means / var)

        kld_summands = sq_mean_deviations + logvar - sq_Z_batch
        #kld_summands = sq_mean_deviations + var - sq_Z_batch

        kld_sum = kld_summands.sum(dim = 1)

        return -0.5 * kld_sum.mean()

