
import torch
import numpy as np

from torch import Tensor
from torch import nn


"""
Variational Decoder Classes - Base class
-------------------------------------------------------------------------------------------------------------------------------------------
This class should work for all generative models where the conditional distributions parameters 
are vectors of the same dimension as the input/output space (e.g. mean and diagonal variance in a Gaussian)
"""
class VarDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_dist_params: int, n_layers: int = 4):
        super().__init__()

        self.output_dim = output_dim
        self.n_dist_params = n_dist_params

        dist_dim = n_dist_params * output_dim

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

        genm_dist_params: Tensor = self.layers[-1](x)

        #genm_dist_params = genm_dist_params.reshape(-1, self.latent_dim, self.n_dist_params).squeeze()
        #genm_dist_params = genm_dist_params.view(-1, self.n_dist_params, self.output_dim).squeeze()
        genm_dist_params = genm_dist_params.view(-1, self.output_dim, self.n_dist_params).squeeze()

        return genm_dist_params
    



"""
Variational Decoder Classes - Experiment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VarDecoderExp(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_dist_params: int, n_layers: int = 4, dtype = torch.float64):
        super().__init__()

        self.output_dim = output_dim
        self.n_dist_params = n_dist_params

        dist_dim = n_dist_params * output_dim

        transition_step = (dist_dim - latent_dim) // n_layers

        n_f = lambda i: latent_dim + transition_step * i
        
        self.layers = nn.ModuleList()

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True, dtype = dtype))
        
        self.layers.append(nn.Linear(in_features = n_f(n_layers - 1), out_features = dist_dim, dtype = dtype))

        self.activation = nn.ReLU()
        #self.activation = nn.PReLU()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        genm_dist_params: Tensor = self.layers[-1](x)

        #genm_dist_params = genm_dist_params.reshape(-1, self.latent_dim, self.n_dist_params).squeeze()
        #genm_dist_params = genm_dist_params.view(-1, self.n_dist_params, self.output_dim).squeeze()
        genm_dist_params = genm_dist_params.view(-1, self.output_dim, self.n_dist_params).squeeze()

        return genm_dist_params
    


#VarDecoder = VarDecoderExp