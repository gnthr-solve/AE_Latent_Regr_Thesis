
"""
Regressor Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np
import math

from torch import Tensor
from torch import nn

from .layer_blocks import LinearFunnel, ExponentialFunnel
    

"""
Regressor Classes - Linear
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LinearRegr(nn.Module):

    def __init__(self, latent_dim: int, y_dim: int = 2):
        super().__init__()
        
        self.regr_map = nn.Linear(in_features = latent_dim, out_features = y_dim, bias = True)
        

    def forward(self, z: Tensor) -> Tensor:

        y_hat = self.regr_map(z)

        return y_hat




"""
Regressor Classes - Deep Neural Network
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class FunnelDNNRegr(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 2, 
            dropout_rate: float = 0.05, 
            activation: str = 'ReLU'
            ):
        super().__init__()
        
        ###--- Layer Nodes ---###
        # Find nearest power of 2 greater than or equal to input_dim
        start_power = math.ceil(math.log2(input_dim))
        #end_power = max(math.ceil(math.log2(output_dim)), start_power - n_layers)
        end_power = math.ceil(math.log2(output_dim)) + 1
        
        # Generate layer dimensions using powers of 2
        hidden_dims = [2**p for p in range(start_power, end_power - 1, -1)]
        
        # Ensure first layer matches input_dim exactly
        hidden_dims[0] = input_dim
        

        ###--- Activation ---###
        if activation == 'ReLU':
            self.activation = nn.ReLU()

        elif activation == 'PReLU':
            self.activation = nn.PReLU()

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        elif activation == 'Softplus':
            self.activation = nn.Softplus()
        

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
        
        self.network = nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> Tensor:

        return self.network(x)




class DNNRegr(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 2, 
            n_fixed_layers: int = 3, 
            fixed_layer_size: int = 300,
            n_funnel_layers: int = 3,
            dropout_rate: float = 0.05, 
            activation: str = 'ReLU'
        ):
        super().__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()

        elif activation == 'PReLU':
            self.activation = nn.PReLU()

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        elif activation == 'Softplus':
            self.activation = nn.Softplus()

        else:
            raise ValueError(f"Unsupported activation: {activation}")
        

        ###--- Fixed Hidden layers ---###
        layers = []
        
        prev_dim = input_dim
        for _ in range(n_fixed_layers):
            layers.extend([
                nn.Linear(prev_dim, fixed_layer_size),
                nn.BatchNorm1d(fixed_layer_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = fixed_layer_size
        
        
        ###--- Set up funnel ---###
        if n_funnel_layers is None:
            # if number of layers is None the funnel is exponential with self determined n_layers
            funnel = ExponentialFunnel(
                input_dim = prev_dim,
                output_dim = output_dim,
                dropout_rate = dropout_rate,
                activation = activation,
            )

        else:
            funnel = LinearFunnel(
                input_dim = prev_dim,
                output_dim = output_dim,
                n_layers = n_funnel_layers,
                dropout_rate = dropout_rate,
                activation = activation,
            )

        self.network = nn.Sequential(*layers, funnel)


    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)




"""
Regressor Classes - Product Regression Model
-------------------------------------------------------------------------------------------------------------------------------------------
Problem:
    If the latent variable is negative and the weight is fractional, the result will be a complex number,
    because we are taking a root of a negative number.
    Either we take the absolute value of the latent variable before taking the power,
    or the latent variable has to be guaranteed to be positive.
"""
class ProductRegr(nn.Module):

    def __init__(self, latent_dim: int, y_dim: int = 2):
        super().__init__()
        
        self.out_weights = nn.Parameter(torch.rand(latent_dim, y_dim))
        self.out_bias = nn.Parameter(torch.rand(y_dim))
        #print(f'out_weights: {self.out_weights}')

    def forward(self, z: torch.Tensor) -> Tensor:

        z = torch.abs(z)
        #powered_z = torch.pow(z.unsqueeze(1), self.out_weights)
        powered_z = torch.pow(z.unsqueeze(-1), self.out_weights)  
        # print(
        #     f'powered_z: \n{powered_z}\n'
        #     f'powered_z shape: {powered_z.shape}'
        # )

        product_result = torch.prod(powered_z, dim=1) 
        #print(f'product_result shape: {product_result.shape}')

        y_hat = self.out_bias * product_result  

        return y_hat