
import torch
import numpy as np

from torch import Tensor
from torch import nn


"""
Variational Decoder Classes - Gaussian Relu Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GaussianVarDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_layers: int = 4):
        super().__init__()

        self.output_dim = output_dim

        dist_dim = 2 * output_dim

        transition_step = (dist_dim - latent_dim) // n_layers

        n_f = lambda i: latent_dim + transition_step * i
        
        self.layers = nn.ModuleList()

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        self.layers.append(nn.Linear(in_features = n_f(n_layers - 1), out_features = dist_dim))

        self.activation = nn.ReLU()
        #self.activation = nn.PReLU()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        mu_sigma: Tensor = self.layers[-1](x)

        #mu_sigma = mu_sigma.reshape(-1, self.latent_dim, 2).squeeze()
        mu_sigma = mu_sigma.view(-1, self.output_dim, 2).squeeze()

        mu, sigma = mu_sigma.unbind(dim = -1)

        return mu, sigma
    



