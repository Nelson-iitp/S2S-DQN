import matplotlib.pyplot as plt
import torch as tt
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from math import log
from .modular import dense
from .torchtf import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

__all__ = ['FixedSinusoidalPE', 'TrainableLinearPE', 'EncodingTransformer', 'DecodingTransformer', #'EDecodingTransformer',
           'S2STransformer', 'MS2STransformer', 'MXS2STransformer', 
           'configure_optimizers']

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
        self.position = tt.arange(0, block_size, 1, dtype=tt.int32, device=device) #<-- no need to unsqueeze, will broadcast
        #NOTE: call self.embedding(self.position) every time because the embedding weights get trained

    def forward(self, x): return x +  self.embedding(self.position)


class EncodingTransformer(nn.Module):

    def __init__(self, embed_size: int, vocab_size:int,
                 pos_embed:nn.Module, num_heads: int, 
                 hidden_size: int, activation: nn.Module,
                 num_layers: int, dropout: float , layer_norm_eps:float, norm_first:bool, 
                 coder_norm_eps:float,
                 dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.embed_size, self.num_heads, self.hidden_size, self.num_layers  = \
             embed_size, num_heads,      hidden_size,      num_layers
        #self.norm_factor = sqrt(self.input_size)
        self.vocab_size=vocab_size
        self.embeder = nn.Embedding(vocab_size, embed_size, **self.factory)
        self.pos_embed=pos_embed
        self.coder = TransformerEncoder(
            num_layers=num_layers, 
            norm=(nn.LayerNorm(embed_size, eps=coder_norm_eps, **self.factory) if coder_norm_eps else None ) ,
            encoder_layer=TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=num_heads, 
                dim_feedforward=hidden_size, 
                dropout=dropout,
                activation=activation, 
                layer_norm_eps=layer_norm_eps, 
                batch_first=True, 
                norm_first=norm_first, 
                **self.factory))
        self.do_store_attention(False)
    
    def do_store_attention(self, do_store): self.store_attention=do_store
    def forward(self, x, mask=None, key_padding_mask=None): 
        return self.coder.forward(
            src=self.pos_embed(self.embeder(x)), 
            mask=mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=None,
            store_attention=self.store_attention)

    @tt.no_grad()
    def view_attention(self, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
        fig, axs = plt.subplots(self.coder.num_layers, 1, figsize=(width, self.coder.num_layers*width))
        if not title: title=f'{__class__}'
        fig.suptitle(f'{title}')
        single = self.coder.num_layers==1
        for l in range(self.coder.num_layers):
            #print(self.decoder.layers[l].attention_weights.shape)
            if batch_index is None:
                w = self.coder.layers[l].attention_weights.detach().cpu()
            else:
                w = self.coder.layers[l].attention_weights[batch_index].detach().cpu()
            ax = axs if single else axs[l]
            ax.matshow(w.numpy(), **matshow)
            ax.set_xlabel(f'layer: {l+1}')
            if ticks:
                ax.set_xticks(range(w.shape[1]))
                ax.set_yticks(range(w.shape[0]))
            if values:
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            if verbose: print(f'Layer: {l} :: \n{w}')
        if save:
            fig.savefig(save)
            plt.close()
        else:
            plt.show()

class DecodingTransformer(nn.Module):

    def __init__(self, embed_size: int, vocab_size:int,
                 pos_embed:nn.Module, num_heads: int, 
                 hidden_size: int, activation: nn.Module,
                 num_layers: int, dropout: float , layer_norm_eps:float, norm_first:bool, 
                 coder_norm_eps:float,
                 dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.embed_size, self.num_heads, self.hidden_size, self.num_layers  = \
             embed_size, num_heads,      hidden_size,      num_layers
        #self.norm_factor = sqrt(self.input_size)
        self.vocab_size=vocab_size
        self.embeder = nn.Embedding(vocab_size, embed_size, **self.factory)
        self.pos_embed=pos_embed
        self.coder = TransformerDecoder(
            num_layers=num_layers, 
            norm=(nn.LayerNorm(embed_size, eps=coder_norm_eps, **self.factory) if coder_norm_eps else None ) ,
            decoder_layer=TransformerDecoderLayer(
                d_model=embed_size, 
                nhead=num_heads, 
                dim_feedforward=hidden_size, 
                dropout=dropout,
                activation=activation, 
                layer_norm_eps=layer_norm_eps, 
                batch_first=True, 
                norm_first=norm_first, 
                **self.factory))
        self.do_store_attention(False)
    
    def do_store_attention(self, do_store): self.store_attention=do_store
    def forward(self, x, c, multi_memory=False, mask=None, key_padding_mask=None): 
        return self.coder.forward(
            tgt=self.pos_embed(self.embeder(x)), 
            memory=c,
            multi_memory=multi_memory,
            tgt_mask=mask,
            memory_mask=None, 
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=None,
            store_attention=self.store_attention)

    @tt.no_grad()
    def view_attention(self, cross, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
        fig, axs = plt.subplots(self.coder.num_layers, 1, figsize=(width, self.coder.num_layers*width))
        if not title: title=f'{__class__}'
        fig.suptitle(f'{title}')
        single = self.coder.num_layers==1
        for l in range(self.coder.num_layers):
            #print(self.decoder.layers[l].attention_weights.shape)
            if batch_index is None:
                w = self.coder.layers[l].attention_weights_cross.detach().cpu() if cross else self.coder.layers[l].attention_weights.detach().cpu()
            else:
                w = self.coder.layers[l].attention_weights_cross[batch_index].detach().cpu() if cross else self.coder.layers[l].attention_weights[batch_index].detach().cpu()
            ax = axs if single else axs[l]
            ax.matshow(w.numpy(), **matshow)
            ax.set_xlabel(f'layer: {l+1}')
            if ticks:
                ax.set_xticks(range(w.shape[1]))
                ax.set_yticks(range(w.shape[0]))
            if values:
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            if verbose: print(f'Layer: {l} :: \n{w}')
        if save:
            fig.savefig(save)
            plt.close()
        else:
            plt.show()


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

    def __init__(self, n_actions:int, custom_encoder:EncodingTransformer, custom_decoder:DecodingTransformer,
                 dense_layer_dims:list, dense_actFs:list, dense_bias=True, xavier_init=False,
                 device=None, dtype=None) -> None:
        assert custom_encoder.embed_size==custom_decoder.embed_size, f'Expecting same embedding dimension!'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoder = custom_encoder
        self.decoder = custom_decoder
        self.embed_size = self.decoder.embed_size
        self.n_actions = n_actions
        self.dense = dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_actions,
            actFs=dense_actFs, use_bias=dense_bias, use_biasL=dense_bias, **factory_kwargs)
        if xavier_init: self._reset_parameters()

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
            ) -> Tensor:

        memory = self.encoder.forward(
            x=src,
            mask=src_mask, 
            key_padding_mask=src_key_padding_mask)
        
        output = self.decoder.forward(
            x=tgt, 
            c=memory, 
            multi_memory=False,
            mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask)
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

    def __init__(self, n_actions:int, custom_encoders:list[EncodingTransformer], custom_decoder:DecodingTransformer,
                 encoder_decoder_layer_mapping: list, #<---- for each decoder layer, which encoder should it take input from, a list of integers
                 dense_layer_dims:list, dense_actFs:list, dense_bias=True, xavier_init=False,
                 device=None, dtype=None) -> None:
        for custom_encoder in custom_encoders:
            assert custom_encoder.embed_size==custom_decoder.embed_size, f'Expecting same embedding dimension!'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoders = nn.ModuleList(custom_encoders)
        self.decoder = custom_decoder
        self.encoder_decoder_layer_mapping = encoder_decoder_layer_mapping
        self.has_encoder_decoder_layer_mapping = True if self.encoder_decoder_layer_mapping else False
        self.embed_size = self.decoder.embed_size
        self.n_actions = n_actions
        self.dense = dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_actions,
            actFs=dense_actFs, use_bias=dense_bias, use_biasL=dense_bias, **factory_kwargs)
        if xavier_init: self._reset_parameters()
        

    def forward(self, srcs: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
            ) -> Tensor:


        memory_ = [encoder.forward(
                    x=src,
                    mask=src_mask, 
                    key_padding_mask=src_key_padding_mask) for encoder,src in zip(self.encoders, srcs)]
        
        memory = \
            [ memory_[encoder_index] for encoder_index in self.encoder_decoder_layer_mapping ] \
            if self.has_encoder_decoder_layer_mapping else\
            memory_
        # decoder expects one encoded input per layer
        output = self.decoder.forward(
            x=tgt, 
            c=memory, 
            multi_memory=True,
            mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask)
        return self.dense(output)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class MXS2STransformer(nn.Module):

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

    def __init__(self, n_actions:int, custom_encoders:list[EncodingTransformer], custom_decoder:DecodingTransformer,
                 dense_layer_dims:list, dense_actFs:list, dense_bias=True, xavier_init=False,
                 device=None, dtype=None) -> None:
        #for custom_encoder in custom_encoders:
            #assert custom_encoder.embed_size==custom_decoder.embed_size, f'Expecting same embedding dimension!'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoders = nn.ModuleList(custom_encoders)
        self.decoder = custom_decoder
        self.embed_size = self.decoder.embed_size
        self.n_actions = n_actions
        self.dense = dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_actions,
            actFs=dense_actFs, use_bias=dense_bias, use_biasL=dense_bias, **factory_kwargs)
        if xavier_init: self._reset_parameters()

    def forward(self, srcs: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
            ) -> Tensor:

        memory = tt.cat([encoder.forward(
                    x=src,
                    mask=src_mask, 
                    key_padding_mask=src_key_padding_mask) for encoder,src in zip(self.encoders, srcs)], dim=-1)
        
        output = self.decoder.forward(
            x=tgt, 
            c=memory, 
            multi_memory=False,
            mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask)
        return self.dense(output)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



# class EDecodingTransformer(nn.Module):

#     def __init__(self, embed_size: int, vocab_size:int,
#                  pos_embed:nn.Module, num_heads: int, 
#                  hidden_size: int, activation: nn.Module,
#                  num_layers: int, incoder_num_layers:int, dropout: float , layer_norm_eps:float, norm_first:bool, 
#                  coder_norm_eps:float, incoder_norm_eps:float,
#                  dtype=None, device=None):
#         super().__init__()
#         self.factory = dict(dtype=dtype, device=device)
#         self.embed_size, self.num_heads, self.hidden_size, self.num_layers  = \
#              embed_size, num_heads,      hidden_size,      num_layers
#         #self.norm_factor = sqrt(self.input_size)
#         self.vocab_size=vocab_size
#         self.embeder = nn.Embedding(vocab_size, embed_size, **self.factory)
#         self.pos_embed=pos_embed
#         self.incoder = TransformerEncoder(
#             num_layers=incoder_num_layers,
#             norm=(nn.LayerNorm(embed_size, eps=incoder_norm_eps, **self.factory) if incoder_norm_eps else None ) ,
#             encoder_layer=TransformerEncoderLayer(
#                 d_model=embed_size,
#                 nhead=num_heads,
#                 dim_feedforward=hidden_size,
#                 dropout=dropout,
#                 activation=activation,
#                 layer_norm_eps=layer_norm_eps,
#                 batch_first=True,
#                 norm_first=norm_first,
#                 **self.factory
#             ),
#         )
#         self.coder = TransformerDecoder(
#             num_layers=num_layers, 
#             norm=(nn.LayerNorm(embed_size, eps=coder_norm_eps, **self.factory) if coder_norm_eps else None ) ,
#             decoder_layer=TransformerDecoderLayer(
#                 d_model=embed_size, 
#                 nhead=num_heads, 
#                 dim_feedforward=hidden_size, 
#                 dropout=dropout,
#                 activation=activation, 
#                 layer_norm_eps=layer_norm_eps, 
#                 batch_first=True, 
#                 norm_first=norm_first, 
#                 **self.factory))
#         self.do_store_attention(False)
    
#     def do_store_attention(self, do_store): self.store_attention=do_store
#     def forward(self, x, c, multi_memory=False, mask=None, key_padding_mask=None): 
#         tgt = self.incoder.forward(
#             src=self.pos_embed(self.embeder(x)),
#             mask=mask,
#             src_key_padding_mask=key_padding_mask,
#             store_attention=self.store_attention
#         )
#         return self.coder.forward(
#             tgt=tgt, 
#             memory=c,
#             multi_memory=multi_memory,
#             tgt_mask=None, #<--- no masking at upper decoder layers
#             memory_mask=None, 
#             tgt_key_padding_mask=None,
#             memory_key_padding_mask=None,
#             store_attention=self.store_attention)

#     @tt.no_grad()
#     def view_attention(self, cross, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
#         #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
#         fig, axs = plt.subplots(self.coder.num_layers, 1, figsize=(width, self.coder.num_layers*width))
#         if not title: title=f'{__class__}'
#         fig.suptitle(f'{title}')
#         single = self.coder.num_layers==1
#         for l in range(self.coder.num_layers):
#             #print(self.decoder.layers[l].attention_weights.shape)
#             if batch_index is None:
#                 w = self.coder.layers[l].attention_weights_cross.detach().cpu() if cross else self.coder.layers[l].attention_weights.detach().cpu()
#             else:
#                 w = self.coder.layers[l].attention_weights_cross[batch_index].detach().cpu() if cross else self.coder.layers[l].attention_weights[batch_index].detach().cpu()
#             ax = axs if single else axs[l]
#             ax.matshow(w.numpy(), **matshow)
#             ax.set_xlabel(f'layer: {l+1}')
#             if ticks:
#                 ax.set_xticks(range(w.shape[1]))
#                 ax.set_yticks(range(w.shape[0]))
#             if values:
#                 for i in range(w.shape[0]):
#                     for j in range(w.shape[1]):
#                         ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
#                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
#             if verbose: print(f'Layer: {l} :: \n{w}')
#         if save:
#             fig.savefig(save)
#             plt.close()
#         else:
#             plt.show()

#     @tt.no_grad()
#     def view_in_attention(self, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
#         #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
#         fig, axs = plt.subplots(self.incoder.num_layers, 1, figsize=(width, self.incoder.num_layers*width))
#         if not title: title=f'{__class__}'
#         fig.suptitle(f'{title}')
#         single = self.incoder.num_layers==1
#         for l in range(self.incoder.num_layers):
#             #print(self.decoder.layers[l].attention_weights.shape)
#             if batch_index is None:
#                 w = self.incoder.layers[l].attention_weights.detach().cpu()
#             else:
#                 w = self.incoder.layers[l].attention_weights[batch_index].detach().cpu()
#             ax = axs if single else axs[l]
#             ax.matshow(w.numpy(), **matshow)
#             ax.set_xlabel(f'layer: {l+1}')
#             if ticks:
#                 ax.set_xticks(range(w.shape[1]))
#                 ax.set_yticks(range(w.shape[0]))
#             if values:
#                 for i in range(w.shape[0]):
#                     for j in range(w.shape[1]):
#                         ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
#                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
#             if verbose: print(f'Layer: {l} :: \n{w}')
#         if save:
#             fig.savefig(save)
#             plt.close()
#         else:
#             plt.show()
# # contain extra decoder layer that is a self-attention layer with causal masking
# # this is because we want causal masking at only first layer of transformer

def configure_optimizers(model, device_type, weight_decay=1e-2, learning_rate=1e-3,  betas=(0.9, 0.999), eps=1e-8):
    import inspect
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(tt.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = tt.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer