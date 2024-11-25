
"""
Decoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Decoder Classes - General Linear Relu Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_layers: int = 4, activation = 'ReLU'):
        super().__init__()

        ###--- Layers ---###
        transition_step = (output_dim - latent_dim) // n_layers

        n_f = lambda i: latent_dim + transition_step * i
        
        self.layers = nn.ModuleList()

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        self.layers.append(nn.Linear(in_features = n_f(n_layers - 1), out_features = output_dim))

        ###--- Activation ---###
        if activation == 'ReLU':
            self.activation = nn.ReLU()

        elif activation == 'PReLU':
            self.activation = nn.PReLU()

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        elif activation == 'Softplus':
            self.activation = nn.Softplus()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        x = self.layers[-1](x)

        return x
    



