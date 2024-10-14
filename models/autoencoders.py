
"""
Autoencoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Autoencoder Classes - Simple Autoencoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class SimpleAutoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat
    



"""
Autoencoder Classes - GaussVAE
-------------------------------------------------------------------------------------------------------------------------------------------
Assumption that both the inference model, parametrised by the encoder, and the generative model, parametrised by the decoder,
are Gaussian and both mean and variance/std are learnable.
"""
class GaussVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        mu_l, log_sigma_l = self.encoder(x)
        
        z = self.sample_gaussian(mu = mu_l, log_sigma = log_sigma_l)

        mu_r, log_sigma_r = self.decoder(z)

        return mu_l, log_sigma_l, mu_r, log_sigma_r
    

    def sample_gaussian(self, mu, log_sigma):

        std = torch.exp(log_sigma)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z




"""
Autoencoder Classes - NaiveVAE
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class NaiveVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        mu_l, log_sigma_l = self.encoder(x)
        
        z = self.sample_gaussian(mu = mu_l, log_sigma = log_sigma_l)

        mu_r, log_sigma_r = self.decoder(z)

        x_hat = self.sample_gaussian(mu = mu_r, log_sigma = log_sigma_r)

        return x_hat
    

    def sample_gaussian(self, mu, log_sigma):

        std = torch.exp(log_sigma)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z
