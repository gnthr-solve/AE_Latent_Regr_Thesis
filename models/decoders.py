
"""
Decoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn

from .activations import ACTIVATIONS


"""
Decoder Classes - General Linear Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_layers: int = 4, activation = 'ReLU'):
        super().__init__()

        self.activation = ACTIVATIONS[activation]()

        ###--- Layers ---###
        transition_step = (output_dim - latent_dim) // n_layers

        n_f = lambda i: latent_dim + transition_step * i
        
        layers =[]
        for i in range(n_layers - 1):

            layers.extend([
                nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True),
                self.activation,
                ])

        layers.append(nn.Linear(in_features = n_f(n_layers - 1), out_features = output_dim))

        self.decoder_network = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:

        return self.decoder_network(x)
    



