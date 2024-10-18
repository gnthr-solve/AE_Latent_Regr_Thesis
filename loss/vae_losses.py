

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn




"""
Reconstruction - GaussianReconstrLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GaussianReconstrLoss:

    def __call__(self, X_batch: Tensor, gen_model_means: Tensor, gen_model_logvar: Tensor) -> Tensor:

        dim = X_batch.shape[-1]
        const = dim * torch.log(Tensor([2 * torch.pi]))
    
        diff_mean = (X_batch - gen_model_means).pow(2)
        var = torch.exp(gen_model_logvar)

        adj_mean_sums = (diff_mean / var + var).sum(dim = 1)

        loss = -0.5 * (const + (adj_mean_sums.mean()))

        return loss


"""
KL Divergence - AnaGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Analytical KL-divergence loss term for Gaussian inference model and standard Gaussian prior
"""
class AnaGaussianKLDLoss:

    def __call__(self, mu: Tensor, logvar: Tensor) -> Tensor:

        kld_batch = 0.5 * (-1 - logvar + mu.pow(2) + torch.exp(logvar)).sum(dim = 1)

        total_kld = kld_batch.mean()

        return total_kld
    



"""
KL Divergence - MCGaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
Monte Carlo estimate for KL-divergence loss term for Gaussian inference model and standard Gaussian prior,
following the same principle as for the reconstruction
"""
class MCGaussianKLDLoss:

    def __call__(self, Z_batch: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:

        sq_diff_means = (Z_batch - mu).pow(2)
        sq_Z_batch = Z_batch.pow(2)
        var = torch.exp(logvar)

        sq_mean_deviations = (sq_diff_means / var)

        kld_summands = sq_mean_deviations + var - sq_Z_batch
        kld_sum = kld_summands.sum(dim = 1)

        return -0.5 * kld_sum.mean()



        
        