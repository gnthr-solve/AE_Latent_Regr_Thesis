
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
        

    def forward(self, z: Tensor):

        y_hat = self.regr_map(z)

        return y_hat
    