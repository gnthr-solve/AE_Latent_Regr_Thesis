
import torch
import numpy as np

from torch import Tensor
from torch import nn




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