
"""
Autoencoder Classes - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Autoencoder Classes - Simple Autoencoder
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class AE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x: Tensor) -> Tensor:

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
    

