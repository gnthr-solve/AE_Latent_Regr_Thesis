
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


    def forward(self, x: Tensor):

        x = torch.relu(self.lin_in(x))
        x = torch.relu(self.lin_one(x))
        x = torch.relu(self.lin_two(x))

        z = self.lin_out(x)

        return z