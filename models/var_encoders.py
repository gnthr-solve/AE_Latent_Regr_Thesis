
"""
Variational Encoder Classes - TODO
-------------------------------------------------------------------------------------------------------------------------------------------
NOTE:
    
    In principle, assuming we treat the dimensions of the distribution on the latent space independently,
    we need e.g. for a Gaussian a more than two dimensional output.
    
    1. Ansatz:
    Map to a two dim output of shape (l, 2) where latent_dim == l, something like:
    
    [[mu_1, sigma_1],
      ...,
     [mu_l, sigma_l]]

    This could be somewhat challenging to accomplish with the batch-wise processing.
    (b, n) --> (b, l, 2)
    Not so trivial as before to scale down.

    2. Ansatz:
    Alternatively we could map to a flattened version, i.e.
    [mu_1, sigma_1,..., mu_l, sigma_l]
    which would mean mapping 
    (b, n) --> (b, l * 2)

    This would work with a similar scaling approach as the deterministic versions,
    and one could reshape the tensor appropriately before sampling

"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Variational Encoder Classes - Ansatz 2. Flatten - GaussianVarEncoder
-------------------------------------------------------------------------------------------------------------------------------------------
Assumption: Gaussian with diagonal covariance
"""
class GaussianVarEncoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, n_layers: int = 4):
        super().__init__()

        self.latent_dim = latent_dim

        dist_dim = 2 * latent_dim

        transition_step = (input_dim - dist_dim) // n_layers
        remainder = (input_dim - dist_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features = input_dim, out_features = start))

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        self.activation = nn.ReLU()
        #self.activation = nn.PReLU()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        mu_sigma: Tensor = self.layers[-1](x)

        #mu_sigma = mu_sigma.reshape(-1, self.latent_dim, 2).squeeze()
        mu_sigma = mu_sigma.view(-1, self.latent_dim, 2).squeeze()

        mu, sigma = mu_sigma.unbind(dim = -1)

        return mu, sigma
    



"""
Variational Encoder Classes - FlattenVarEncoder Generalisation Ideas
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GenFlattenVarEncoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, n_dist_params: int, n_layers: int = 4):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_dist_params = n_dist_params

        dist_dim = n_dist_params * latent_dim

        transition_step = (input_dim - dist_dim) // n_layers
        remainder = (input_dim - dist_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features = input_dim, out_features = start))

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        self.activation = nn.ReLU()
        #self.activation = nn.PReLU()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        dist_params: Tensor = self.layers[-1](x)

        #dist_params = dist_params.reshape(-1, self.latent_dim, 2).squeeze()
        dist_params = dist_params.view(-1, self.latent_dim, self.n_dist_params).squeeze()

        #if the distribution had 3 params, would give a 3 tuple
        dist_params_tuple = dist_params.unbind(dim = -1)

        return dist_params_tuple