
"""
Regressor Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn

    

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