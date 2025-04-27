
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor

from .ffn import FFN
from .attention_head import SelfAttentionHead, AttentionHead
from .multihead_attention import MultiHeadSelfAttention, MultiHeadAttention



class TransformerLayer(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_inner: int, num_heads: int = 1, attn_bias = True):
        super().__init__()

        if num_heads == 1:
            self.attn = AttentionHead(d_model = d_model, d_k = d_k, d_v = d_model, bias = attn_bias)
        else:
            self.attn = MultiHeadAttention(d_model = d_model, num_heads = num_heads, d_v = d_k, bias = attn_bias)

        self.ffn = FFN(d_model = d_model, d_inner = d_inner)
        self.attn_lnorm = nn.LayerNorm(d_model)
        self.ffn_lnorm = nn.LayerNorm(d_model)


    def forward(self, input: Tensor, lengths: Tensor):
        
        attn_residual = self.attn_lnorm(input + self.attn(input = input, lengths = lengths))
        output = self.ffn_lnorm(attn_residual + self.ffn(attn_residual))

        return output
    


class BasicTransformer(nn.Module):

    def __init__(self, num_layers: int, layer_kwargs: dict):
        super().__init__()

        self.transformer_layers = nn.ModuleList([TransformerLayer(**layer_kwargs) for _ in range(num_layers)])

    
    def forward(self, input: Tensor, lengths: Tensor):

        output = input

        for layer in self.transformer_layers:
            output = layer(input = output, lengths = lengths)

        return output