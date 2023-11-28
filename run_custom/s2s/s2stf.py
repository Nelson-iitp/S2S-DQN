#import matplotlib.pyplot as plt
import torch as tt
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from known.mod import dense
from .tf import EncodingTransformer, DecodingTransformer

__all__ = ['S2STransformer', 'MS2STransformer' ]


class S2STransformer(nn.Module):

    def view_attention_encoder(self, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        self.encoder.view_attention(batch_index=batch_index, width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    

    def view_attention_decoder(self, cross, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        self.decoder.view_attention(cross=cross, batch_index=batch_index, width=width, 
                values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)   
        
    def do_store_attention(self, do_store, encoder=True, decoder=True):
        if decoder: self.decoder.do_store_attention(do_store)
        if encoder: self.encoder.do_store_attention(do_store)    

    def __init__(self, n_output:int, custom_encoder:EncodingTransformer, custom_decoder:DecodingTransformer,
                 dense_layer_dims:list, dense_actFs:list, dense_bias=True, xavier_init=False,
                 device=None, dtype=None) -> None:
        assert custom_encoder.embed_size==custom_decoder.embed_size, f'Expecting same embedding dimension!'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoder = custom_encoder
        self.decoder = custom_decoder
        self.embed_size = self.decoder.embed_size
        self.n_output = n_output
        self.dense = dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_output,
            actFs=dense_actFs, bias=dense_bias, **factory_kwargs)
        if xavier_init: self._reset_parameters()

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None ) -> Tensor:

        memory = self.encoder.forward(
            x=src,
            mask=src_mask)
        
        output = self.decoder.forward(
            x=tgt, 
            c=memory, 
            multi_memory=False,
            mask=tgt_mask, cask=None)
        return self.dense(output)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class MS2STransformer(nn.Module):

    def view_attention_encoder(self, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        for encoder in self.encoders: encoder.view_attention(batch_index=batch_index, width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    

    def view_attention_decoder(self, cross, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        self.decoder.view_attention(cross=cross, batch_index=batch_index, width=width, 
                values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)   
        
    def do_store_attention(self, do_store, encoder=True, decoder=True):
        if decoder: self.decoder.do_store_attention(do_store)
        if encoder: 
            for encoder in self.encoders: encoder.do_store_attention(do_store)    

    def __init__(self, n_output:int, custom_encoders:list[EncodingTransformer], custom_decoder:DecodingTransformer,
                 dense_layer_dims:list, dense_actFs:list, dense_bias=True, xavier_init=False,
                 device=None, dtype=None) -> None:
        #for custom_encoder in custom_encoders:
            #assert custom_encoder.embed_size==custom_decoder.embed_size, f'Expecting same embedding dimension!'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoders = nn.ModuleList(custom_encoders)
        self.decoder = custom_decoder
        self.embed_size = self.decoder.embed_size
        self.n_output = n_output
        self.dense = dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_output,
            actFs=dense_actFs, bias=dense_bias, **factory_kwargs)
        if xavier_init: self._reset_parameters()

    def forward(self, srcs: Tensor, tgt: Tensor, src_masks = None, tgt_mask  = None ) -> Tensor:

        if src_masks is None:
            memory = tt.cat([encoder.forward(
                        x=src,
                        mask=None ) for encoder,src in zip(self.encoders, srcs)], dim=-1)
        else:
            memory = tt.cat([encoder.forward(
                        x=src,
                        mask=src_mask ) for encoder,src,src_mask in zip(self.encoders, srcs, src_masks)], dim=-1)
        
        output = self.decoder.forward(
            x=tgt, 
            c=memory, 
            multi_memory=False,
            mask=tgt_mask )
        return self.dense(output)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# @staticmethod
# def GPTAdamW(model, device, weight_decay=1e-2, learning_rate=1e-3,  betas=(0.9, 0.999), eps=1e-8):
#     import inspect
#     # start with all of the candidate parameters
#     param_dict = {pn: p for pn, p in model.named_parameters()}
#     # filter out those that do not require grad
#     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#     optim_groups = [
#         {'params': decay_params, 'weight_decay': weight_decay},
#         {'params': nodecay_params, 'weight_decay': 0.0}
#     ]
#     num_decay_params = sum(p.numel() for p in decay_params)
#     num_nodecay_params = sum(p.numel() for p in nodecay_params)
#     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
#     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
#     # Create AdamW optimizer and use the fused version if it is available
#     fused_available = 'fused' in inspect.signature(tt.optim.AdamW).parameters
#     use_fused = fused_available and device == 'cuda'
#     extra_args = dict(fused=True) if use_fused else dict()
#     optimizer = tt.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
#     print(f"using fused AdamW: {use_fused}")

#     return optimizer