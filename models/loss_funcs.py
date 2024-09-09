
"""
Loss Functions - Imports
-------------------------------------------------------------------------------------------------------------------------------------------
"""

import torch
import numpy as np

from torch import Tensor
from torch import nn




"""
Loss Functions - Simple Loss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss()


    def forward(self, x: Tensor, x_hat: Tensor):

        loss = self.loss_fn(x, x_hat)

        return loss