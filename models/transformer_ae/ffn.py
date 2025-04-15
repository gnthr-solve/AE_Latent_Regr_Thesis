
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from ..activations import ACTIVATIONS


class FFN(nn.Module):

    def __init__(self, d_model: int, d_inner: int, activation = 'ReLU'):
        super().__init__()

        self.L_in = nn.Linear(in_features = d_model, out_features = d_inner, bias = True)
        self.L_out = nn.Linear(in_features = d_inner, out_features = d_model, bias = True)
    
        self.activation = ACTIVATIONS[activation]()

    
    def forward(self, x: Tensor):

        return self.L_out(self.activation(self.L_in(x)))