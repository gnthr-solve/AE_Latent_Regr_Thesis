
"""
Encoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn

from .activations import ACTIVATIONS


"""
Encoder Classes - General Linear Encoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, n_layers: int = 4, activation = 'ReLU'):
        super().__init__()

        self.activation = ACTIVATIONS[activation]()

        ###--- Layers ---###
        transition_step = (input_dim - latent_dim) // n_layers
        remainder = (input_dim - latent_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        layers = []

        layers.append(nn.Linear(in_features = input_dim, out_features = start))

        for i in range(n_layers - 1):

            layers.extend([
                self.activation,
                nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True),
            ])

        self.encoder_network = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:

        return self.encoder_network(x)
    
    
"""
NOTE: PReLU acts in the same way as LeakyReLU, but it learns the slope of the negative part of the function.
"""