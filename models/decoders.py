
"""
Decoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn




"""
Decoder Classes - Simple Decoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleDecoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.reconst_layer = nn.Linear(in_features = latent_dim, out_features = input_dim, bias = True)


    def forward(self, z: Tensor):

        x_hat = self.reconst_layer(z)
        
        return x_hat