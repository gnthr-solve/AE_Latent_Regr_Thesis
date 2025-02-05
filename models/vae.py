
import torch
import numpy as np

from torch import Tensor
from torch import nn

from abc import ABC, abstractmethod

"""
VAE - ABC
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VAE(nn.Module, ABC):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        z = self.reparameterise(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        return z, infrm_dist_params, genm_dist_params

    
    @abstractmethod
    def reparameterise(self, dist_params: Tensor) -> Tensor:
        pass




"""
VAE - GaussVAE
-------------------------------------------------------------------------------------------------------------------------------------------
Inference Model is assumed Gaussian with diagonal covariance matrix.
Expects encoder to return mean and log(variance) as parameters.
"""
class GaussVAE(VAE):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
    

    def reparameterise(self, dist_params: Tensor):

        mu, logvar = dist_params.unbind(dim = -1)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z




"""
VAE - GaussVAESigma
-------------------------------------------------------------------------------------------------------------------------------------------
Inference Model is assumed Gaussian with diagonal covariance matrix.
Expects encoder to return mean and standard deviation as parameters.
"""
class GaussVAESigma(VAE):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
    

    def reparameterise(self, dist_params: Tensor):

        mu, sigma = dist_params.unbind(dim = -1)

        eps = torch.randn_like(sigma)

        z = mu + sigma * eps

        return z



