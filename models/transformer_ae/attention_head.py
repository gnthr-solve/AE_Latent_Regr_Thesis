
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor


"""
Single Attention Head - SelfAttentionHead
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SelfAttentionHead(nn.Module):

    def __init__(self, d_model: int, d_k: int, bias: bool = True):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k, bias = bias)
        self.W_k = nn.Linear(d_model, d_k, bias = bias)
        self.W_v = nn.Linear(d_model, d_model, bias = bias)
        self.d_k = d_k


    def forward(self, input: Tensor, lengths: Tensor = None) -> Tensor:

        key = self.W_q(input)  # [batch, seq_len, d_k]
        query = self.W_k(input)    # [batch, seq_len, d_k]
        value = self.W_v(input)  # [batch, seq_len, d_model]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        print(
            f'Lengths shape: {lengths.shape}\n'
            f'query shape: {query.shape}\n'
            f'key shape: {key.shape}\n'
            f'value shape: {value.shape}\n'
            f'Scores shape: {scores.shape}\n'
        )
        # Masking (optional)
        if lengths is not None:
            mask = self.create_scores_mask(lengths = lengths)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
        
        # Aggregate values
        output = torch.matmul(attention, value)  # [batch, seq_len, d_model]
        
        return output


    def create_scores_mask(self, lengths: Tensor) -> Tensor:
        batch_size = lengths.size(0)
        seq_len = lengths.max().item()

        positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)

        # Create validity mask [batch, seq_len] 
        # True where position < length, False otherwise
        valid_positions = positions < lengths.unsqueeze(1)
        
        # Which token can pose a query - padding cannot
        # First expansion: [batch, seq_len, 1]
        # Second expansion: [batch, seq_len, seq_len]
        query_mask = valid_positions.unsqueeze(2).expand(-1, -1, seq_len)
        # Which token can provide a key - padding cannot
        key_mask = valid_positions.unsqueeze(1).expand(-1, seq_len, -1)

        attention_mask = query_mask & key_mask

        # Later tokens cannot influence earlier ones
        # Creates lower triangular matrix
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        print(
            f'causal_mask: \n{causal_mask[:10, :10]}\n'
        )
        return attention_mask & ~causal_mask




"""
Single Attention Head - AttentionHead for composed MultiHeadAttention - implemented for understanding, less efficient
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AttentionHead(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, bias: bool = True):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k, bias = bias)
        self.W_k = nn.Linear(d_model, d_k, bias = bias)
        self.W_v = nn.Linear(d_model, d_v, bias = bias)
        self.d_k = d_k


    def forward(self, query: Tensor, key: Tensor, value: Tensor, lengths: Tensor = None) -> Tensor:
        
        Q = self.W_q(query)  # [batch, seq_len, d_k]
        K = self.W_k(key)    # [batch, seq_len, d_k]
        V = self.W_v(value)  # [batch, seq_len, d_v]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Masking (optional)
        if lengths is not None:
            mask = self.create_scores_mask(lengths = lengths)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
        
        # Aggregate values
        output = torch.matmul(attention, V)  # [batch, seq_len, d_v]
        
        return output


    def create_scores_mask(self, lengths: Tensor) -> Tensor:
        batch_size = lengths.size(0)
        seq_len = lengths.max().item()

        positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)

        # Create validity mask [batch, seq_len] 
        # True where position < length, False otherwise
        valid_positions = positions < lengths.unsqueeze(1)
        
        # Which token can pose a query - padding cannot
        # First expansion: [batch, seq_len, 1]
        # Second expansion: [batch, seq_len, seq_len]
        query_mask = valid_positions.unsqueeze(2).expand(-1, -1, seq_len)
        # Which token can provide a key - padding cannot
        key_mask = valid_positions.unsqueeze(1).expand(-1, seq_len, -1)

        attention_mask = query_mask & key_mask

        return attention_mask
