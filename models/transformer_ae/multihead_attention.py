
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor

from .attention_head import SelfAttentionHeadDV


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module for Transformer AE
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)




class MultiHeadAttentionDV(nn.Module):
    """
    Multi-Head Attention Module for Transformer AE
    """
    def __init__(self, d_model: int, num_heads: int, d_v: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_v
        self.d_vh = d_v * num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_vh)
        self.W_o = nn.Linear(self.d_vh, d_model)
        
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_vh)
        
        return self.W_o(output)



class MultiHeadAttentionFromSingleHeads(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHeadDV(d_model, self.d_k, self.d_k)
            for _ in range(num_heads)
        ])
        self.W_o = nn.Linear(num_heads * self.d_k, d_model)

    def forward(self, query, key, value, mask=None):
        # Each head processes the same inputs
        head_outputs = [head(query, key, value, mask) for head in self.heads]
        
        # Concatenate along feature dimension
        combined = torch.cat(head_outputs, dim=-1)  # [batch, seq, num_heads*d_k]
        
        # Project back to d_model
        return self.W_o(combined)  # [batch, seq, d_model]
