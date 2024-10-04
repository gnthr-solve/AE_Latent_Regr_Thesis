
"""
Composite Models - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn



"""
Composite Models - Encoder, Regressor
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class EnRegrComposite(nn.Module):

    def __init__(self, encoder: nn.Module, regressor: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.regressor = regressor


    def forward(self, x: Tensor):

        z = self.encoder(x)

        y_hat = self.regressor(z)

        return y_hat