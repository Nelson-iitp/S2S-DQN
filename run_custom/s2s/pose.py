

__all__ = [
    'FixedSinusoidalPE', 'TrainableLinearPE',
]


import torch as tt
import torch.nn as nn
from math import log

class FixedSinusoidalPE(nn.Module):

    def __init__(self, input_size: int, block_size: int, dim_constant:float=1e4, dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device) 
        self.input_size, self.block_size = input_size, block_size
        # self.dropout = nn.Dropout(p=dropout)  # we do not need dropout in embedding
        position = tt.arange(block_size).unsqueeze(1)
        div_term = tt.exp(tt.arange(0, input_size, 2) * (-log(dim_constant)/ input_size))
        embedding = tt.zeros((block_size, 1, input_size), dtype=dtype)
        embedding[:, 0, 0::2] = tt.sin(position * div_term)
        embedding[:, 0, 1::2] = tt.cos(position * div_term)
        embedding.swapaxes_(1,0)
        embedding = embedding.to(**self.factory)
        self.register_buffer('embedding', embedding) #<--- optinal

    def forward(self, x): return x + self.embedding
    
class TrainableLinearPE(nn.Module):

    def __init__(self, input_size: int, block_size: int, dtype=None, device=None ):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.input_size, self.block_size = input_size, block_size
        # self.dropout = nn.Dropout(p=dropout)  # we do not need dropout in embedding
        self.embedding = nn.Embedding(block_size, input_size, **self.factory)
        self.position = tt.arange(0, block_size, 1, dtype=tt.long, device=device) #<-- no need to unsqueeze, will broadcast
        #NOTE: call self.embedding(self.position) every time because the embedding weights get trained

    def forward(self, x): return x +  self.embedding(self.position)
