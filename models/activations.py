
from torch import nn


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Softplus': nn.Softplus,
}