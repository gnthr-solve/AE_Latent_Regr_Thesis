
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
from torch.nn.functional import softplus

from .activations import ACTIVATIONS


"""
Variational Encoder Classes - Vector Parameters Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
This class should work for all inference models where the conditional distributions parameters 
are vectors of the same dimension as the latent space (e.g. mean and diagonal variance in a Gaussian)
"""
class VarEncoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, n_dist_params: int, n_layers: int = 4, activation = 'ReLU'):
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
        
        ###--- Activation ---###
        self.activation = ACTIVATIONS[activation]()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        infrm_dist_params: Tensor = self.layers[-1](x)

        #infrm_dist_params = infrm_dist_params.reshape(-1, self.latent_dim, self.n_dist_params).squeeze()
        #infrm_dist_params = infrm_dist_params.view(-1, self.n_dist_params, self.latent_dim).squeeze()
        infrm_dist_params = infrm_dist_params.view(-1, self.latent_dim, self.n_dist_params).squeeze()

        return infrm_dist_params
    



"""
Variational Encoder Classes - Experiment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SigmaGaussVarEncoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, n_layers: int = 4, activation = 'ReLU'):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_dist_params = 2

        dist_dim = self.n_dist_params * latent_dim

        transition_step = (input_dim - dist_dim) // n_layers
        remainder = (input_dim - dist_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features = input_dim, out_features = start))

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        ###--- Activation ---###
        self.activation = ACTIVATIONS[activation]()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        infrm_dist_params: Tensor = self.layers[-1](x)

        # shape = (b, l, 2)
        infrm_dist_params = infrm_dist_params.view(-1, self.latent_dim, self.n_dist_params).squeeze()

        ###--- transform variance-related parameter for stability, interpret as std sigma. ---###
        mean = infrm_dist_params[..., 0]
        var_param = infrm_dist_params[..., 1]
        
        # softplus guarantees smaller positive values, addition of small stability factor avoids log(0) = -inf 
        sigma = softplus(var_param) + 1e-6

        return torch.stack([mean, sigma], dim=-1).squeeze()
    

