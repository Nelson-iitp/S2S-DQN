import matplotlib.pyplot as plt
import torch as tt
import torch.nn as nn
from math import log
from .attn import SHA, MHA
from .block import AttentionBlock, FFN

__all__ = [
    'EncodingTransformer', 'DecodingTransformer', 
    'Optimizers'
    ]


class EncodingTransformer(nn.Module):
    
    def __init__(self, 
                 embed_size, 
                 block_size,
                 score2F_SA,
                 vocab_size,
                 pos_embed, 
                 num_heads, 
                 hidden_size, 
                 activation2F,
                 num_layers, 
                 dropout , 
                 normF, 
                 norm_first, 
                 cross_pre_norm,
                 final_norm,
                 attention_bias=True,
                 ffn_bias=True,
                 norm_bias=True,
                 dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.eps = 1e-6
        self.embed_size, self.num_heads, self.hidden_size, self.num_layers  = \
             embed_size, num_heads,      hidden_size,      num_layers
        self.is_multi_head = (self.num_heads>1)
        self.block_size=block_size
        self.vocab_size=vocab_size
        self.embeder = nn.Embedding(vocab_size, embed_size, **self.factory)
        self.pos_embed=pos_embed
        self.final_norm = nn.LayerNorm(embed_size, eps=self.eps, **self.factory) if final_norm else nn.Identity() 
        #self.scoreF = scoreF
        #f = MHA if self.is_multi_head else SHA
        self.coder = nn.ModuleList ([AttentionBlock(
            self_attention_layer=\
                MHA.SelfAttention(
                    embed_dim=self.embed_size,
                    block_size=self.block_size,
                    num_heads=self.num_heads,
                    scoreF=score2F_SA[0](**score2F_SA[1]),
                    dropout=dropout,
                    bias=attention_bias,
                    num_layers=num_layers, #<-- only for GPT-based initialization
                    **self.factory),
                cross_attention_layer= None,
                ffn_layer=FFN(
                    embed_dim=self.embed_size,
                    hidden_dim=self.hidden_size,
                    act2F=activation2F,
                    bias=ffn_bias,
                    num_layers=num_layers, #<-- only for GPT-based initialization
                    **self.factory
                ),
                normF=normF,
                norm_first=norm_first,
                norm_bias=norm_bias,
                norm_eps=self.eps,
                cross_pre_norm=cross_pre_norm,
                dropout=dropout,
                **self.factory) for _ in range(self.num_layers)])
    
    def do_store_attention(self, do_store): 
        for c in self.coder: c.do_store_attention(do_store)

    def forward(self, x, mask=None): 
        out = self.pos_embed(self.embeder(x))
        for coder in self.coder:
            out = coder.forward(out, None, mask=mask, cask=None )
        return self.final_norm(out)

    # @tt.no_grad()
    # def view_attention(self, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
    #     #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
    #     fig, axs = plt.subplots(self.coder.num_layers, 1, figsize=(width, self.coder.num_layers*width))
    #     if not title: title=f'{__class__}'
    #     fig.suptitle(f'{title}')
    #     single = self.coder.num_layers==1
    #     for l in range(self.coder.num_layers):
    #         #print(self.decoder.layers[l].attention_weights.shape)
    #         if batch_index is None:
    #             w = self.coder.layers[l].attention_weights.detach().cpu()
    #         else:
    #             w = self.coder.layers[l].attention_weights[batch_index].detach().cpu()
    #         ax = axs if single else axs[l]
    #         ax.matshow(w.numpy(), **matshow)
    #         ax.set_xlabel(f'layer: {l+1}')
    #         if ticks:
    #             ax.set_xticks(range(w.shape[1]))
    #             ax.set_yticks(range(w.shape[0]))
    #         if values:
    #             for i in range(w.shape[0]):
    #                 for j in range(w.shape[1]):
    #                     ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
    #                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #         if verbose: print(f'Layer: {l} :: \n{w}')
    #     if save:
    #         fig.savefig(save)
    #         plt.close()
    #     else:
    #         plt.show()

class DecodingTransformer(nn.Module):

    def __init__(self, 
                 embed_size, 
                 block_size,
                 score2F_SA,
                 score2F_CA,
                 vocab_size,
                 pos_embed, 
                 num_heads, 
                 hidden_size, 
                 activation2F,
                 num_layers, 
                 dropout , 
                 normF, 
                 norm_first, 
                 cross_pre_norm,
                 final_norm,
                 attention_bias=True,
                 ffn_bias=True,
                 norm_bias=True,
                 dtype=None, device=None):
        super().__init__()
        self.factory = dict(dtype=dtype, device=device)
        self.eps = 1e-6
        self.embed_size, self.num_heads, self.hidden_size, self.num_layers  = \
             embed_size, num_heads,      hidden_size,      num_layers
        self.is_multi_head = (self.num_heads>1)
        self.vocab_size=vocab_size
        self.block_size=block_size
        self.embeder = nn.Embedding(vocab_size, embed_size, **self.factory)
        self.pos_embed=pos_embed
        self.final_norm = nn.LayerNorm(embed_size, eps=self.eps, **self.factory) if final_norm else nn.Identity() 
        self.coder = nn.ModuleList ([AttentionBlock(
            self_attention_layer=\
                MHA.SelfAttention(
                    embed_dim=self.embed_size,
                    block_size=self.block_size,
                    num_heads=self.num_heads,
                    scoreF=score2F_SA[0](**score2F_SA[1]),
                    dropout=dropout,
                    bias=attention_bias,
                    num_layers=num_layers, #<-- only for GPT-based initialization
                    **self.factory),
                cross_attention_layer=\
                    MHA.CrossAttention(
                    embed_dim=self.embed_size,
                    block_size=self.block_size,
                    num_heads=self.num_heads,
                    scoreF=score2F_CA[0](**score2F_CA[1]),
                    dropout=dropout,
                    bias=attention_bias,
                    num_layers=num_layers, #<-- only for GPT-based initialization
                    **self.factory),
                ffn_layer=FFN(
                    embed_dim=self.embed_size,
                    hidden_dim=self.hidden_size,
                    act2F=activation2F,
                    bias=ffn_bias,
                    num_layers=num_layers, #<-- only for GPT-based initialization
                    **self.factory
                ),
                normF=normF,
                norm_first=norm_first,
                norm_bias=norm_bias,
                norm_eps=self.eps,
                cross_pre_norm=cross_pre_norm,
                dropout=dropout,
                **self.factory) for _ in range(self.num_layers)])
        self.do_store_attention(False)
    
    def do_store_attention(self, do_store): 
        for c in self.coder: c.do_store_attention(do_store)
       
    def forward(self, x, c, mask=None, cask=None, multi_memory=False): 
        out = self.pos_embed(self.embeder(x))
        if multi_memory:
            for coder,ctx in zip(self.coder,c): out = coder.forward(out, ctx, mask=mask, cask=cask)
        else:
            for coder in self.coder:out = coder.forward(out, c, mask=mask, cask=cask)
        return self.final_norm(out)

    # @tt.no_grad()
    # def view_attention(self, cross, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
    #     #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
    #     fig, axs = plt.subplots(self.coder.num_layers, 1, figsize=(width, self.coder.num_layers*width))
    #     if not title: title=f'{__class__}'
    #     fig.suptitle(f'{title}')
    #     single = self.coder.num_layers==1
    #     for l in range(self.coder.num_layers):
    #         #print(self.decoder.layers[l].attention_weights.shape)
    #         if batch_index is None:
    #             w = self.coder.layers[l].attention_weights_cross.detach().cpu() if cross else self.coder.layers[l].attention_weights.detach().cpu()
    #         else:
    #             w = self.coder.layers[l].attention_weights_cross[batch_index].detach().cpu() if cross else self.coder.layers[l].attention_weights[batch_index].detach().cpu()
    #         ax = axs if single else axs[l]
    #         ax.matshow(w.numpy(), **matshow)
    #         ax.set_xlabel(f'layer: {l+1}')
    #         if ticks:
    #             ax.set_xticks(range(w.shape[1]))
    #             ax.set_yticks(range(w.shape[0]))
    #         if values:
    #             for i in range(w.shape[0]):
    #                 for j in range(w.shape[1]):
    #                     ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
    #                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #         if verbose: print(f'Layer: {l} :: \n{w}')
    #     if save:
    #         fig.savefig(save)
    #         plt.close()
    #     else:
    #         plt.show()


class Optimizers:

    @staticmethod
    def GPTAdamW(model, device, weight_decay=1e-2, learning_rate=1e-3,  betas=(0.9, 0.999), eps=1e-8):
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
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = tt.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer