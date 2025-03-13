
import torch
import torch.nn as nn
import math

from torch import Tensor

def print_t(tensor: Tensor, name: str = '', n: int = 5):
    print(
        f'Tensor {name} \n'
        f'--------------------------------------\n'
        f'tensor.shape: \n{tensor.shape}\n\n'
        f'--------------------------------------\n'
        f'tensor[:{n}]: \n{tensor[:n]}\n\n'
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, K: float = 10000.0):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print_t(pe, name = 'pe')
        print_t(position, name = 'position')

        # Compute the positional encodings once in logarithmic space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(K) / d_model))
        print_t(div_term, name = 'div_term')

        # for even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # for odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  #unsqueeze for batch dimension
        self.register_buffer('pe', pe)


    def forward(self, x):
        
        x = x + self.pe[:x.size(0)]

        return x



class PositionalEncoding0(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
        # Compute the positional encodings once in logarithmic space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # x shape: (sequence_length, batch_size, d_model)
        x = x + self.pe[:x.size(0)]

        return x
