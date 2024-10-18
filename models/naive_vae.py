
import torch
import numpy as np

from torch import Tensor
from torch import nn


"""
NaiveVAE
-------------------------------------------------------------------------------------------------------------------------------------------
In the concrete case of Gaussian prior, inference model and generative model,
one can use the reparameterisation trick to sample the reconstruction as well.
In larger dimensional data this might be compute-cost prohibitive, but here it works well.
"""
class NaiveVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        z = self.sample_gaussian(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.sample_gaussian(genm_dist_params)

        return x_hat
    

    def sample_gaussian(self, dist_params: Tensor) -> Tensor:

        mu, logvar = dist_params.unbind(dim = -1)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z




"""
NaiveVAESigma - predicts the log of the std instead of the log of the variance
-------------------------------------------------------------------------------------------------------------------------------------------
Strangely, predicting the log of the std = sigma works better than predicting the log of the variance ~2-3%.
Perhaps because squaring small values makes them smaller, and the logarithm translates this to larger negative values, 
that might be more difficult to handle/learn for the NNs.
"""
class NaiveVAESigma(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        z = self.sample_gaussian(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.sample_gaussian(genm_dist_params)

        return x_hat
    

    def sample_gaussian(self, dist_params: Tensor) -> Tensor:

        mu, log_sigma = dist_params.unbind(dim = -1)

        std = torch.exp(log_sigma)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z
