
"""
Encoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn


"""
Encoder Classes - General Linear Relu Encoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, n_layers: int = 4, activation = 'ReLU'):
        super().__init__()

        ###--- Layers ---###
        transition_step = (input_dim - latent_dim) // n_layers
        remainder = (input_dim - latent_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features = input_dim, out_features = start))

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
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
    

"""
NOTE: PReLU acts in the same way as LeakyReLU, but it learns the slope of the negative part of the function.
"""