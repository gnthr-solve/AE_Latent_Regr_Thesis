
import torch
import numpy as np

from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod

from .vae import VAE
from .autoencoders import AE

"""
NaiveVAE
-------------------------------------------------------------------------------------------------------------------------------------------
In the concrete case of Gaussian prior, inference model and generative model,
one can use the reparameterisation trick to sample the reconstruction as well.
In larger dimensional data this might be compute-cost prohibitive, but here it works well.
"""
class NaiveVAE(AE, ABC):

    def __init__(self, encoder, decoder):
        super().__init__(encoder = encoder, decoder = decoder)


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        z = self.reparameterise(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.reparameterise(genm_dist_params)

        return z, x_hat
    

    @abstractmethod
    def reparameterise(self, dist_params: Tensor) -> Tensor:

        pass
    

"""
NaiveVAE
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class NaiveVAE_LogVar(NaiveVAE):

    def __init__(self, encoder, decoder, input_dim: int):
        super().__init__(encoder = encoder, decoder = decoder)

        self.batch_norm = nn.BatchNorm1d(num_features = input_dim)


    def forward(self, x: Tensor) -> Tensor:
        
        x = self.batch_norm(x)

        return super().forward(x)


    def reparameterise(self, dist_params: Tensor) -> Tensor:

        mu, logvar = dist_params.unbind(dim = -1)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z




"""
NaiveVAE_LogSigma - predicts the log of the std instead of the log of the variance
-------------------------------------------------------------------------------------------------------------------------------------------
Strangely, predicting the log of the std = sigma works better than predicting the log of the variance ~2-3%.
Perhaps because squaring small values makes them smaller, and the logarithm translates this to larger negative values, 
that might be more difficult to handle/learn for the NNs.
"""
class NaiveVAE_LogSigma(NaiveVAE):

    def reparameterise(self, dist_params: Tensor) -> Tensor:

        mu, log_sigma = dist_params.unbind(dim = -1)

        #log_sigma = torch.clamp(log_sigma, max = 10)

        std = torch.exp(log_sigma)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z




"""
NaiveVAESigma - predicts the std directly
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class NaiveVAE_Sigma(NaiveVAE):

    def reparameterise(self, dist_params: Tensor) -> Tensor:

        mu, std = dist_params.unbind(dim = -1)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z
