

import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn




"""
Reconstruction - GaussianReconstrLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GaussianReconstrLoss:

    reduction = 'mean'

    def __init__(self):
        
        self.reconstr_loss = nn.MSELoss(reduction = self.reduction)


    def __call__(self, X_batch: Tensor, gen_model_means: Tensor) -> Tensor:

        return self.reconstr_loss(X_batch, gen_model_means)
    


"""
KL Divergence - GaussianKLDLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GaussianKLDLoss:

    def __call__(self, mu: Tensor, log_sigma: Tensor) -> Tensor:

        kld_batch = 0.5 * (-1 - log_sigma + mu.pow(2) + log_sigma.exp()).mean(dim = 0)

        total_kld = kld_batch.sum()

        return total_kld