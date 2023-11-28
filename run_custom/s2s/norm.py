

__all__ = ['Norm', 'RMSNorm', 'LayerNorm']
import torch as tt
import torch.nn as nn
from torch.nn import functional as F


class Norm(nn.Module):

    def __init__(self, embed_dim, bias=True, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.embed_dim=embed_dim
        self.eps = eps
        self.weight = nn.Parameter(tt.ones(self.embed_dim, **self.factory))
        self.bias = nn.Parameter(tt.zeros(self.embed_dim, **self.factory)) if bias else None 




class RMSNorm(Norm):

    def __init__(self, embed_dim, bias=True, eps=1e-6, dtype=None, device=None):
        super().__init__(embed_dim, bias, eps, dtype, device)
        if self.bias is None: self.bias = tt.tensor(0, requires_grad=False, **self.factory)
    def _norm(self, x): return x * tt.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight + self.bias

class LayerNorm(Norm):

    def forward(self, x):  return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

