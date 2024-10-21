
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
VAE - ABC
-------------------------------------------------------------------------------------------------------------------------------------------
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




