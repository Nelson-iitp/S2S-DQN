

__all__ = ['FFN', 'AttentionBlock']
import math
import torch as tt
import torch.nn as nn
#from torch.nn import functional as F


class FFN(nn.Module):

    def _init_residual_connection_parameters(self, num_layers): 
            tt.nn.init.normal_(self.P.weight, mean=0.0, std=0.02/(2 * num_layers)**0.5)
            tt.nn.init.normal_(self.P.bias, mean=0.0, std=0.02/(2 * num_layers)**0.5)

    def __init__(self, embed_dim, hidden_dim, act2F, dropout=0.0, bias=True, num_layers=None, dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)

        self.embed_dim = embed_dim
        self.dense    = nn.Linear(embed_dim, hidden_dim, bias=bias, **self.factory)
        actF, actA = act2F
        self.activation  = actF(**actA)
        self.P  = nn.Linear(hidden_dim, embed_dim, bias=bias, **self.factory)
        self.dropout = nn.Dropout(dropout)
        if num_layers: self._init_residual_connection_parameters(num_layers)

    def forward(self, x): return self.P( self.dropout( self.activation( self.dense( x ) ) ) )


class AttentionBlock(nn.Module):

    def __init__(self, 
                self_attention_layer, 
                cross_attention_layer, 
                ffn_layer:FFN, 
                normF, norm_first, norm_bias=True, norm_eps=1e-6, 
                cross_pre_norm=False, dropout=0.0,
                dtype=None, device=None):
        super().__init__()
        if self_attention_layer is not None: assert self_attention_layer.embed_dim == ffn_layer.embed_dim
        if cross_attention_layer is not None: assert cross_attention_layer.embed_dim == ffn_layer.embed_dim
        self.factory = dict(dtype=dtype, device=device)
        self.embed_dim=ffn_layer.embed_dim
        self.self_attention = self_attention_layer
        self.cross_attention = cross_attention_layer
        self.ffn = ffn_layer
        self.has_self_attention = self.self_attention is not None
        self.has_cross_attention = self.cross_attention is not None
        self.has_both_attention = self.has_cross_attention and self.has_self_attention
        self.has_no_attention = not (self.has_cross_attention or self.has_self_attention)
        self.norm_first = norm_first

        if self.has_self_attention: self.normSA = normF(self.embed_dim, bias=norm_bias, eps=norm_eps, **self.factory)
        if self.has_cross_attention: self.normCA = normF(self.embed_dim, bias=norm_bias, eps=norm_eps, **self.factory)
        self.normFF = normF(self.embed_dim, bias=norm_bias, eps=norm_eps, **self.factory)

        if cross_pre_norm and norm_first: self.normCPN = normF(self.embed_dim, bias=norm_bias, eps=norm_eps, **self.factory)
        else: self.normCPN = nn.Identity()
        
        if self.has_self_attention: self.dropoutSA = nn.Dropout(dropout)
        if self.has_cross_attention: self.dropoutCA = nn.Dropout(dropout)
        self.dropoutFF = nn.Dropout(dropout)

        assert not self.has_no_attention, f'No attention blocks! Requires at least one.'
        if self.has_both_attention: self.forward = self.forwardSC
        else:
            self.forward = self.forwardC if self.has_cross_attention else self.forwardS

    def do_store_attention(self, store): 
        if self.has_self_attention: self.self_attention.do_store_attention(store)
        if self.has_cross_attention: self.cross_attention.do_store_attention(store)

    def forwardSC(self, X, C, mask=None, cask=None):
        x, c = X, C
        if self.norm_first:
            x = x + self._sa_block(self.normSA(x), mask=mask)
            x = x + self._ca_block(self.normCA(x), self.normCPN(c), mask=cask)
            x = x + self._ff_block(self.normFF(x))
        else:
            x = self.normSA(x + self._sa_block(x, mask=mask))
            x = self.normCA(x + self._ca_block(x, c, mask=cask))
            x = self.normFF(x + self._ff_block(x))
        return x

    def forwardC(self, X, C, mask=None, cask=None):
        x, c = X, C
        if self.norm_first:
            x = x + self._ca_block(self.normCA(x), self.normCPN(c), mask=cask)
            x = x + self._ff_block(self.normFF(x))
        else:
            x = self.normCA(x + self._ca_block(x, c, mask=cask))
            x = self.normFF(x + self._ff_block(x))
        return x
    
    def forwardS(self, X, C, mask=None, cask=None):
        x = X
        if self.norm_first:
            x = x + self._sa_block(self.normSA(x), mask=mask)
            x = x + self._ff_block(self.normFF(x))
        else:
            x = self.normSA(x + self._sa_block(x, mask=mask))
            x = self.normFF(x + self._ff_block(x))
        return x
    
    # self-attention block
    def _sa_block(self, s, mask=None):
        x = self.self_attention.forward(s, mask=mask)
        return self.dropoutSA(x)

    # cross attention block
    def _ca_block(self, s, c, mask=None):
        x = self.cross_attention.forward(s, c, mask=mask) 
        return self.dropoutCA(x)

    # feed forward block
    def _ff_block(self, x): return self.dropoutFF(self.ffn(x))

