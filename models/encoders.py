
"""
Encoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn




"""
Encoder Classes - Simple Encoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        self.in_layer = nn.Linear(in_features = input_dim, out_features = latent_dim, bias = True)


    def forward(self, x: Tensor):

        z = self.in_layer(x)

        return z