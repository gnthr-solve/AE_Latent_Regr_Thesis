
import torch
import numpy as np

from torch import Tensor
from torch import nn

from .vae import VAE
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
        
        z = self.reparameterise(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.reparameterise(genm_dist_params)

        return x_hat
    

    def reparameterise(self, dist_params: Tensor) -> Tensor:

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
class NaiveVAELogSigma(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        # with torch.no_grad():
        #     mu = infrm_dist_params[:, :, 0].detach()
        #     log_sigma = infrm_dist_params[:, :, 1].detach()
        #     print(
        #         f'Inference Model Parameters:\n'
        #         f'-----------------------------\n'
        #         f'Shape: \n {infrm_dist_params.shape}\n'
        #         f'-----------------------------\n'
        #         f'mu Max:\n {mu.max()}\n'
        #         f'mu Min:\n {mu.min()}\n'
        #         #f'mu Norm:\n {torch.norm(mu, dim = -1)}\n'
        #         f'mu[:3]:\n {mu[:3]}\n'
        #         f'-----------------------------\n'
        #         f'log_sigma Max:\n {log_sigma.max()}\n'
        #         f'log_sigma  Min:\n {log_sigma.min()}\n'
        #         #f'log_sigma  Norm:\n {torch.norm(log_sigma, dim = -1)}\n'
        #         f'log_sigma[:3]:\n {log_sigma[:3]}\n'
        #         f'-----------------------------\n\n'
        #     )

        z = self.reparameterise(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.reparameterise(genm_dist_params)

        return x_hat
    

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
class NaiveVAESigma(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        infrm_dist_params = self.encoder(x)
        
        z = self.reparameterise(infrm_dist_params)

        genm_dist_params = self.decoder(z)

        x_hat = self.reparameterise(genm_dist_params)

        return x_hat
    

    def reparameterise(self, dist_params: Tensor) -> Tensor:

        mu, std = dist_params.unbind(dim = -1)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z
