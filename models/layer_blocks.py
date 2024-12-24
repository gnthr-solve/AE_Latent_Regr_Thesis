
import torch
import numpy as np
import math

from torch import Tensor
from torch import nn


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Softplus': nn.Softplus,
}


"""
Layer Blocks - LinearReluBlock
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearReluBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList()

        if input_dim > output_dim:

            dim_step = input_dim - output_dim // n_layers
            
            in_features = lambda i: input_dim - i * dim_step
            out_features = lambda i: input_dim - (i + 1) * dim_step

        else:
            dim_step = output_dim - input_dim // n_layers

            in_features = lambda i: input_dim + i * dim_step
            out_features = lambda i: input_dim + (i + 1) * dim_step
        
        
        for i in range(n_layers - 1):

            self.layers.append(nn.Linear(in_features = in_features(i), out_features = out_features(i), bias = True))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(in_features = out_features(n_layers - 1), out_features = output_dim))
        

    def forward(self, x: Tensor):

        for layer in self.layers:
            x = layer(x)

        return x
    



"""
Layer Blocks - LinearFunnel
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearFunnel(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int,
            n_layers: int = 3,
            dropout_rate: float = 0.05, 
            activation: str = 'ReLU'
        ):
        super().__init__()

        self.activation = ACTIVATIONS[activation]()

        ###--- Layer Nodes ---###
        transition_step = (input_dim - output_dim) // n_layers
        remainder = (input_dim - output_dim) % n_layers

        start = input_dim - (transition_step + remainder)
        n_f = lambda i: start - transition_step * i

        layer_nodes = [input_dim, *(n_f(i) for i in range(n_layers - 1))]

        ###--- Create Layers ---###
        layers = []
        for i in range(len(layer_nodes) - 1):
            print(layer_nodes[i], layer_nodes[i + 1])
            layers.extend([
                nn.Linear(layer_nodes[i], layer_nodes[i + 1]),
                nn.BatchNorm1d(layer_nodes[i + 1]),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
        
        # final output layer wo norm, activation or dropout
        layers.extend([
            nn.Linear(layer_nodes[-1], output_dim)
        ])

        self.funnel_network = nn.Sequential(*layers)


    def forward(self, x):
        return self.funnel_network(x)
    


"""
Layer Blocks - ExponentialFunnel
-------------------------------------------------------------------------------------------------------------------------------------------
Scales down by powers of 2, without an n_layers parameter
"""
class ExponentialFunnel(nn.Module):

    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            dropout_rate: float = 0.05, 
            activation: str = 'ReLU'
        ):
        super().__init__()

        self.activation = ACTIVATIONS[activation]()

        ###--- Layer Nodes ---###
        # Find nearest power of 2 greater than or equal to input_dim
        start_power = math.ceil(math.log2(input_dim))
        end_power = math.ceil(math.log2(output_dim)) + 1
        
        # Generate layer dimensions using powers of 2
        hidden_dims = [2**p for p in range(start_power, end_power - 1, -1)]
        
        # Ensure first layer matches input_dim exactly
        hidden_dims[0] = input_dim


        ###--- Create Layers ---###
        layers = []
        for i in range(len(hidden_dims) - 1):
            
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
        
        # final output layer
        layers.extend([
            nn.Linear(hidden_dims[-1], output_dim)
        ])
        
        self.funnel_network = nn.Sequential(*layers)


    def forward(self, x):
        return self.funnel_network(x)