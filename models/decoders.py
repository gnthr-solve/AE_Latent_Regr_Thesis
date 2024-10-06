
"""
Decoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Decoder Classes - SimpleLinearReluDecoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleLinearReluDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()

        self.lin_in = nn.Linear(in_features = latent_dim, out_features = 40, bias = True)
        self.lin_one = nn.Linear(in_features = 40, out_features = 120, bias = True)
        self.lin_two = nn.Linear(in_features = 120, out_features = 200, bias = True)
        self.lin_out = nn.Linear(in_features = 200, out_features = 297, bias = False)


    def forward(self, x: Tensor) -> Tensor:

        x = torch.relu(self.lin_in(x))
        x = torch.relu(self.lin_one(x))
        x = torch.relu(self.lin_two(x))

        z = self.lin_out(x)

        return z
    



"""
Decoder Classes - Linear Relu Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearReluDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int):
        super().__init__()

        transition_step = (output_dim - latent_dim) // 4

        n_f = lambda i: latent_dim + transition_step * i
        
        self.lin_in = nn.Linear(in_features = latent_dim, out_features = n_f(1), bias = True)
        self.lin_one = nn.Linear(in_features = n_f(1), out_features = n_f(2), bias = True)
        self.lin_two = nn.Linear(in_features = n_f(2), out_features = n_f(3), bias = True)
        self.lin_out = nn.Linear(in_features = n_f(3), out_features = output_dim, bias = False)


    def forward(self, x: Tensor) -> Tensor:

        x = torch.relu(self.lin_in(x))
        x = torch.relu(self.lin_one(x))
        x = torch.relu(self.lin_two(x))

        x = self.lin_out(x)

        return x




"""
Decoder Classes - General Linear Relu Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class GeneralLinearReluDecoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, n_layers: int = 4):
        super().__init__()

        transition_step = (output_dim - latent_dim) // n_layers

        n_f = lambda i: latent_dim + transition_step * i
        
        self.layers = nn.ModuleList()

        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = n_f(i), out_features = n_f(i+1), bias = True))
        
        self.layers.append(nn.Linear(in_features = n_f(n_layers - 1), out_features = output_dim))

        #self.activation = nn.ReLU()
        self.activation = nn.PReLU()


    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers[:-1]:

            x = self.activation(layer(x))

        x = self.layers[-1](x)

        return x
    



