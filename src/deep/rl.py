



import os
import numpy as np
import matplotlib.pyplot as plt
from known.basic import Remap
from tqdm import tqdm
import torch as tt
import torch.nn as nn

from math import inf, nan
from .modular import clone, requires_grad_
from .tf import EncodingTransformer, DecodingTransformer, TrainableLinearPE, configure_optimizers
from .tf import  S2STransformer, MS2STransformer, MXS2STransformer
# -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 
# Required Function (Internal)
# -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 
class RIE:
    def __init__(self, Alow, Ahigh, seed=None) -> None: 
        self.Alow, self.Ahigh, self.rng = Alow, Ahigh, np.random.default_rng(seed)

    def predict(self, *args): return self.rng.integers(self.Alow, self.Ahigh)

class VIE:
    def __init__(self, Aseq) -> None: 
        self.Aseq = Aseq
    def predict(self, s, t): return self.Aseq[t]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] Base Value Netowrk Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class QPIE: 
    """ base class for value estimators 
    
        NOTE: for Q-Values, underlying parameters value_theta should accept state-action pair as 2 sepreate inputs  
        NOTE: all Value functions (V or Q) are called in batch mode only """

    def _build_target(self): return (requires_grad_(clone(self.theta), False) if self.has_target else self.theta )
    def __init__(self, value_theta,  has_target, dtype, device):
        self.dtype, self.device = dtype, device
        self.has_target = has_target
        self.theta = value_theta.to(dtype=dtype, device=device) 
        self.theta_ =  self._build_target()
        self.eval()
        self.set_running_params()

    def train(self, mode=True):
        self.theta.train(mode)
        if self.has_target:self.theta_.train(mode)

    def eval(self): self.train(False)

    def forward(self, state, target=False): #<-- called in batch mode
        return (self.theta_ ( state ) if target else self.theta ( state ))

    def predict(self, state): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1).item() #.cpu().numpy()[ts]

    def predict_(self, state): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX.item()
    
    def optimizer(self, optA): return tt.optim.Adam(self.theta.parameters(), **optA)

    def save(self, path): tt.save(self.theta.state_dict(), path)

    def load(self, path):
        self.theta.load_state_dict((tt.load(path)))
        self.theta_ =  self._build_target()
        self.eval()
    
    @staticmethod
    def get_running_params(model): return [ param for name, param in model.state_dict().items() if "running_" in name ]
        
    def set_running_params(self): 
        self.running_params_base = __class__.get_running_params(self.theta)
        self.running_params_target = __class__.get_running_params(self.theta_) if self.has_target else None
    
    def update_running_params(self): 
        for base_param, target_param in zip(self.running_params_base, self.running_params_target): target_param.copy_(base_param)

    @tt.no_grad()
    def update_target_polyak(self, tau):
        r""" performs polyak update """
        for base_param, target_param in zip(self.theta.parameters(), self.theta_.parameters()):
            target_param.data.mul_(1 - tau)
            tt.add(target_param.data, base_param.data, alpha=tau, out=target_param.data)
        self.update_running_params()

    @tt.no_grad()
    def update_target_parameters(self):
        r""" copies parameters """
        for base_param, target_param in zip(self.theta.parameters(), self.theta_.parameters()): target_param.copy_(base_param)
        self.update_running_params()
        
    @tt.no_grad()
    def update_target_state_dict(self):
        r""" copies state dict """
        self.theta_.load_state_dict(self.theta.state_dict())
        self.update_running_params()

    def update_target(self, polyak=0.0):
        if not self.has_target: return False

        if polyak>0: self.update_target_polyak(polyak)
        else:        self.update_target_state_dict()
        
        self.theta_.eval()
        return True

#-----------------------------------------------------------------------------------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] Sequence-to-Sequence DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class S2SDQN(QPIE):

    def __init__(self,     
            n_actions,
            embed_dim,

            encoder_block_size,
            encoder_vocab_count,
            encoder_hidden_dim,
            encoder_activation,
            encoder_num_heads,
            encoder_num_layers,
            encoder_norm_eps, # final norm # keep zero to not use norm
            encoder_dropout,
            encoder_norm_first,

            decoder_block_size,
            decoder_vocab_count,
            decoder_hidden_dim,
            decoder_activation,
            decoder_num_heads,
            decoder_num_layers,
            decoder_norm_eps, # final norm # keep zero to not use norm
            decoder_dropout,
            decoder_norm_first,

            dense_layer_dims,
            dense_actFs,
            dense_bias,

            xavier_init,

            has_target = False, 
            dtype = None, 
            device = None,
            ):

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            value_theta = S2STransformer( 
                n_actions=n_actions,
                # ===========================================================
                custom_encoder= EncodingTransformer(
                        embed_size=embed_dim,
                        vocab_size=encoder_vocab_count,
                        pos_embed=TrainableLinearPE(
                            input_size=embed_dim,
                            block_size=encoder_block_size,
                            dtype=dtype, device=device,
                        ),
                        num_heads=encoder_num_heads,
                        hidden_size=encoder_hidden_dim,
                        activation=encoder_activation,
                        num_layers=encoder_num_layers,
                        dropout=encoder_dropout,
                        layer_norm_eps=1e-8,
                        norm_first=encoder_norm_first,
                        coder_norm_eps=encoder_norm_eps,
                        dtype=dtype, device=device,),
                # ===========================================================

                # ===========================================================
                custom_decoder=DecodingTransformer(
                    embed_size=embed_dim,
                    vocab_size=decoder_vocab_count,
                    pos_embed=TrainableLinearPE(
                        input_size=embed_dim,
                        block_size=decoder_block_size,
                        dtype=dtype, device=device,
                    ),
                    num_heads=decoder_num_heads,
                    hidden_size=decoder_hidden_dim,
                    activation=decoder_activation,
                    num_layers=decoder_num_layers,
                    dropout=decoder_dropout,
                    layer_norm_eps=1e-8,
                    norm_first=decoder_norm_first,
                    coder_norm_eps=decoder_norm_eps,
                    dtype=dtype, device=device,),
                # ===========================================================
                dense_layer_dims=dense_layer_dims,
                dense_actFs=dense_actFs,
                dense_bias=dense_bias,
                xavier_init=xavier_init,
                dtype=dtype, device=device,) 
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            super().__init__(value_theta, has_target, dtype, device)
            self.encoder_block_size = encoder_block_size
            self.decoder_block_size = decoder_block_size
            self.causal_mask = tt.triu(tt.full((self.decoder_block_size, self.decoder_block_size), True, device=self.device), diagonal=1)
            # ===========================================================
            self.encoder_start_index = 0
            self.encoder_end_index = self.encoder_start_index+self.encoder_block_size
            self.decoder_start_index = self.encoder_end_index
            self.decoder_end_index = self.decoder_start_index+self.decoder_block_size

    def optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.999)):
        return configure_optimizers(
            model=self.theta,
            device_type=self.device,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas)
    

    """NOTE:
            
        state contains - encoder input sequence and decoder output sequence
        state is a torch tensor of dtype = tt.long

    """
    def forward(self, state, target=False): #<-- called in batch mode
        encoder_in = state[..., self.encoder_start_index:self.encoder_end_index] 
        decoder_in = state[..., self.decoder_start_index:self.decoder_end_index] 
        label_out = state[..., self.decoder_start_index+1:self.decoder_end_index+1]  # shifted right
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        if target: out = self.theta_(encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        else:      out = self.theta (encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        return out

    def predict_batch(self, state, ts): #<-- called in batch mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[:, ts].cpu().numpy()
    
    def predict(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[ts].item() #.cpu().numpy()[ts]

    def predict_(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX[ts].item()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [M] MSequence-to-Sequence DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MS2SDQN(QPIE):

    def __init__(self,     
            n_actions,
            embed_dim,

            encoder_block_size,
            encoder_vocab_counts,
            encoder_hidden_dim,
            encoder_activation,
            encoder_num_heads,
            encoder_num_layers,
            encoder_norm_eps, # final norm # keep zero to not use norm
            encoder_dropout,
            encoder_norm_first,

            encoder_decoder_layer_mapping,

            decoder_block_size,
            decoder_vocab_count,
            decoder_hidden_dim,
            decoder_activation,
            decoder_num_heads,
            decoder_num_layers,
            decoder_norm_eps, # final norm # keep zero to not use norm
            decoder_dropout,
            decoder_norm_first,

            dense_layer_dims,
            dense_actFs,
            dense_bias,

            xavier_init,

            has_target = False, 
            dtype = None, 
            device = None,
            ):

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            value_theta = MS2STransformer( 
                n_actions=n_actions,
                # ===========================================================
                custom_encoders= [EncodingTransformer(
                        embed_size=embed_dim,
                        vocab_size=encoder_vocab_count,
                        pos_embed=TrainableLinearPE(
                            input_size=embed_dim,
                            block_size=encoder_block_size,
                            dtype=dtype, device=device,
                        ),
                        num_heads=encoder_num_heads,
                        hidden_size=encoder_hidden_dim,
                        activation=encoder_activation,
                        num_layers=encoder_num_layers,
                        dropout=encoder_dropout,
                        layer_norm_eps=1e-8,
                        norm_first=encoder_norm_first,
                        coder_norm_eps=encoder_norm_eps,
                        dtype=dtype, device=device,) for encoder_vocab_count in encoder_vocab_counts],
                # ===========================================================
                encoder_decoder_layer_mapping=encoder_decoder_layer_mapping,
                # ===========================================================
                custom_decoder=DecodingTransformer(
                    embed_size=embed_dim,
                    vocab_size=decoder_vocab_count,
                    pos_embed=TrainableLinearPE(
                        input_size=embed_dim,
                        block_size=decoder_block_size,
                        dtype=dtype, device=device,
                    ),
                    num_heads=decoder_num_heads,
                    hidden_size=decoder_hidden_dim,
                    activation=decoder_activation,
                    num_layers=decoder_num_layers,
                    dropout=decoder_dropout,
                    layer_norm_eps=1e-8,
                    norm_first=decoder_norm_first,
                    coder_norm_eps=decoder_norm_eps,
                    dtype=dtype, device=device,),
                # ===========================================================
                dense_layer_dims=dense_layer_dims,
                dense_actFs=dense_actFs,
                dense_bias=dense_bias,
                xavier_init=xavier_init,
                dtype=dtype, device=device,) 
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            super().__init__(value_theta, has_target, dtype, device)
            self.encoder_block_size = encoder_block_size
            self.decoder_block_size = decoder_block_size
            self.causal_mask = tt.triu(tt.full((self.decoder_block_size, self.decoder_block_size), True, device=self.device), diagonal=1)
            self.n_encoders = len(encoder_vocab_counts)
            # ===========================================================
            self.encoder_start_index = list(range(0, self.encoder_block_size*self.n_encoders, self.encoder_block_size))
            self.encoder_end_index = [ esi+self.encoder_block_size for esi in self.encoder_start_index ]

            self.decoder_start_index = self.encoder_end_index[-1]
            self.decoder_end_index = self.decoder_start_index+self.decoder_block_size

    def optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.999)):
        return configure_optimizers(
            model=self.theta,
            device_type=self.device,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas)
    

    """NOTE:
            
        state contains - encoder input sequence and decoder output sequence
        state is a torch tensor of dtype = tt.long

    """
    def forward(self, state, target=False): #<-- called in batch mode
        encoder_in = [state[..., encoder_start_index:encoder_end_index] for (encoder_start_index, encoder_end_index) in zip(self.encoder_start_index,self.encoder_end_index)]
        decoder_in = state[..., self.decoder_start_index:self.decoder_end_index] 
        label_out = state[..., self.decoder_start_index+1:self.decoder_end_index+1]  # shifted right
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        if target: out = self.theta_(encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        else:      out = self.theta (encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        return out

    def predict_batch(self, state, ts): #<-- called in batch mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[:, ts].cpu().numpy()
    
    def predict(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[ts].item() #.cpu().numpy()[ts]

    def predict_(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX[ts].item()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [M] MSequence-to-Sequence DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MXS2SDQN(QPIE):

    def __init__(self,     
            n_actions,
            embed_dim,

            encoder_block_size,
            encoder_vocab_counts,
            encoder_hidden_dim,
            encoder_activation,
            encoder_num_heads,
            encoder_num_layers,
            encoder_norm_eps, # final norm # keep zero to not use norm
            encoder_dropout,
            encoder_norm_first,

            decoder_block_size,
            decoder_vocab_count,
            decoder_hidden_dim,
            decoder_activation,
            decoder_num_heads,
            decoder_num_layers,
            decoder_norm_eps, # final norm # keep zero to not use norm
            decoder_dropout,
            decoder_norm_first,

            dense_layer_dims,
            dense_actFs,
            dense_bias,

            xavier_init,

            has_target = False, 
            dtype = None, 
            device = None,
            ):

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            n_encoders = len(encoder_vocab_counts)
            value_theta = MXS2STransformer( 
                n_actions=n_actions,
                # ===========================================================
                custom_encoders= [EncodingTransformer(
                        embed_size=embed_dim,
                        vocab_size=encoder_vocab_count,
                        pos_embed=TrainableLinearPE(
                            input_size=embed_dim,
                            block_size=encoder_block_size,
                            dtype=dtype, device=device,
                        ),
                        num_heads=encoder_num_heads,
                        hidden_size=encoder_hidden_dim,
                        activation=encoder_activation,
                        num_layers=encoder_num_layers,
                        dropout=encoder_dropout,
                        layer_norm_eps=1e-8,
                        norm_first=encoder_norm_first,
                        coder_norm_eps=encoder_norm_eps,
                        dtype=dtype, device=device,) for encoder_vocab_count in encoder_vocab_counts],
                # ===========================================================
                # ===========================================================
                custom_decoder=DecodingTransformer(
                    embed_size=embed_dim*n_encoders,
                    vocab_size=decoder_vocab_count,
                    pos_embed=TrainableLinearPE(
                        input_size=embed_dim*n_encoders,
                        block_size=decoder_block_size,
                        dtype=dtype, device=device,
                    ),
                    num_heads=decoder_num_heads,
                    hidden_size=decoder_hidden_dim,
                    activation=decoder_activation,
                    num_layers=decoder_num_layers,
                    dropout=decoder_dropout,
                    layer_norm_eps=1e-8,
                    norm_first=decoder_norm_first,
                    coder_norm_eps=decoder_norm_eps,
                    dtype=dtype, device=device,),
                # ===========================================================
                dense_layer_dims=dense_layer_dims,
                dense_actFs=dense_actFs,
                dense_bias=dense_bias,
                xavier_init=xavier_init,
                dtype=dtype, device=device,) 
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            super().__init__(value_theta, has_target, dtype, device)
            self.encoder_block_size = encoder_block_size
            self.decoder_block_size = decoder_block_size
            self.causal_mask = tt.triu(tt.full((self.decoder_block_size, self.decoder_block_size), True, device=self.device), diagonal=1)
            self.n_encoders = len(encoder_vocab_counts)
            # ===========================================================
            self.encoder_start_index = list(range(0, self.encoder_block_size*self.n_encoders, self.encoder_block_size))
            self.encoder_end_index = [ esi+self.encoder_block_size for esi in self.encoder_start_index ]

            self.decoder_start_index = self.encoder_end_index[-1]
            self.decoder_end_index = self.decoder_start_index+self.decoder_block_size

    def optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.999)):
        return configure_optimizers(
            model=self.theta,
            device_type=self.device,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas)
    

    """NOTE:
            
        state contains - encoder input sequence and decoder output sequence
        state is a torch tensor of dtype = tt.long

    """
    def forward(self, state, target=False): #<-- called in batch mode
        encoder_in = [state[..., encoder_start_index:encoder_end_index] for (encoder_start_index, encoder_end_index) in zip(self.encoder_start_index,self.encoder_end_index)]
        decoder_in = state[..., self.decoder_start_index:self.decoder_end_index] 
        label_out = state[..., self.decoder_start_index+1:self.decoder_end_index+1]  # shifted right
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        if target: out = self.theta_(encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        else:      out = self.theta (encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
        return out

    def predict_batch(self, state, ts): #<-- called in batch mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[:, ts].cpu().numpy()
    
    def predict(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        return tt.argmax(out, dim=-1)[ts].item() #.cpu().numpy()[ts]

    def predict_(self, state, ts): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX[ts].item()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# """ [E] E Sequence-to-Sequence DQN Value Network """
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class ES2SDQN(QPIE):

#     def __init__(self,     
#             n_actions,
#             embed_dim,

#             encoder_block_size,
#             encoder_vocab_count,
#             encoder_hidden_dim,
#             encoder_activation,
#             encoder_num_heads,
#             encoder_num_layers,
#             encoder_norm_eps, # final norm # keep zero to not use norm
#             encoder_dropout,
#             encoder_norm_first,

#             decoder_block_size,
#             decoder_vocab_count,
#             decoder_hidden_dim,
#             decoder_activation,
#             decoder_num_heads,
#             decoder_num_layers,
#             decoder_norm_eps, # final norm # keep zero to not use norm
#             decoder_dropout,
#             decoder_norm_first,

#             dense_layer_dims,
#             dense_actFs,
#             dense_bias,

#             xavier_init,

#             has_target = False, 
#             dtype = None, 
#             device = None,
#             ):

#             # ++++++++++++++++++++++++++++++++++++++++++++++++++++
#             value_theta = S2STransformer( 
#                 n_actions=n_actions,
#                 # ===========================================================
#                 custom_encoder= EncodingTransformer(
#                         embed_size=embed_dim,
#                         vocab_size=encoder_vocab_count,
#                         pos_embed=TrainableLinearPE(
#                             input_size=embed_dim,
#                             block_size=encoder_block_size,
#                             dtype=dtype, device=device,
#                         ),
#                         num_heads=encoder_num_heads,
#                         hidden_size=encoder_hidden_dim,
#                         activation=encoder_activation,
#                         num_layers=encoder_num_layers,
#                         dropout=encoder_dropout,
#                         layer_norm_eps=1e-8,
#                         norm_first=encoder_norm_first,
#                         coder_norm_eps=encoder_norm_eps,
#                         dtype=dtype, device=device,),
#                 # ===========================================================

#                 # ===========================================================
#                 custom_decoder=EDecodingTransformer(
#                     embed_size=embed_dim,
#                     vocab_size=decoder_vocab_count,
#                     pos_embed=TrainableLinearPE(
#                         input_size=embed_dim,
#                         block_size=decoder_block_size,
#                         dtype=dtype, device=device,
#                     ),
#                     num_heads=decoder_num_heads,
#                     hidden_size=decoder_hidden_dim,
#                     activation=decoder_activation,
#                     num_layers=decoder_num_layers, incoder_num_layers=1,
#                     dropout=decoder_dropout,
#                     layer_norm_eps=1e-8,
#                     norm_first=decoder_norm_first,
#                     coder_norm_eps=decoder_norm_eps, incoder_norm_eps=decoder_norm_eps,
#                     dtype=dtype, device=device,),
#                 # ===========================================================
#                 dense_layer_dims=dense_layer_dims,
#                 dense_actFs=dense_actFs,
#                 dense_bias=dense_bias,
#                 xavier_init=xavier_init,
#                 dtype=dtype, device=device,) 
#                 # ++++++++++++++++++++++++++++++++++++++++++++++++++++
#             super().__init__(value_theta, has_target, dtype, device)
#             self.encoder_block_size = encoder_block_size
#             self.decoder_block_size = decoder_block_size
#             self.causal_mask = tt.triu(tt.full((self.decoder_block_size, self.decoder_block_size), True, device=self.device), diagonal=1)
#             # ===========================================================
#             self.encoder_start_index = 0
#             self.encoder_end_index = self.encoder_start_index+self.encoder_block_size
#             self.decoder_start_index = self.encoder_end_index
#             self.decoder_end_index = self.decoder_start_index+self.decoder_block_size

#     def optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.999)):
#         return configure_optimizers(
#             model=self.theta,
#             device_type=self.device,
#             weight_decay=weight_decay,
#             learning_rate=learning_rate,
#             betas=betas)
    

#     """NOTE:
            
#         state contains - encoder input sequence and decoder output sequence
#         state is a torch tensor of dtype = tt.long

#     """
#     def forward(self, state, target=False): #<-- called in batch mode
#         encoder_in = state[..., self.encoder_start_index:self.encoder_end_index] 
#         decoder_in = state[..., self.decoder_start_index:self.decoder_end_index] 
#         label_out = state[..., self.decoder_start_index+1:self.decoder_end_index+1]  # shifted right
#         #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
#         if target: out = self.theta_(encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
#         else:      out = self.theta (encoder_in, decoder_in, tgt_mask = self.causal_mask, tgt_key_padding_mask = None)
#         return out

#     def predict_batch(self, state, ts): #<-- called in batch mode
#         state = state.to(device=self.device) #<-- add batch dim
#         out = self.forward(state)
#         return tt.argmax(out, dim=-1)[:, ts].cpu().numpy()
    
#     def predict(self, state, ts): # <---- called in explore mode
#         state = state.to(device=self.device) #<-- add batch dim
#         out = self.forward(state)
#         return tt.argmax(out, dim=-1)[ts].item() #.cpu().numpy()[ts]

#     def predict_(self, state, ts): # <---- called in explore mode
#         state = state.to(device=self.device) #<-- add batch dim
#         out = self.forward(state)
#         outX = tt.argmax(out, dim=-1)
#         return out, outX, outX[ts].item()
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" [E] Policy Evaluation/Testing ~ does not use explorers 
    NOTE: make sure to put policies in .eval() mode before predicting
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Eval:

    @staticmethod
    @tt.no_grad() # r""" 1-env, 1-episode """
    def test_solution(env, acts, max_steps=None, verbose=0, render=0):
        r""" 1-env, 1-episode """
        s, t, d= env.reset()
        cr = 0.0
        if verbose>0: print('[RESET]')
        if render==2: env.render()

        if max_steps is None: max_steps=inf
        while not (d or t>=max_steps):
            a = acts[t]
            s, t, d, r = env.step(a)
            cr+=r
            if verbose>1: 
                print('[STEP]:[{}], Act:[{}], Reward:[{}], Done:[{}], Return:[{}]'.format(t, a, r, d, cr))
            if render==3: env.render()

        if verbose>0: print('[TERM]: Steps:[{}], Return:[{}]'.format(t, cr))
        if (render==1 or render==2): env.render()
        return cr, t, acts

    @staticmethod
    def test_cost(env, acts):
        test_returns, _, _  = __class__.test_solution(env=env, acts=acts)
        return -test_returns

    @staticmethod
    @tt.no_grad() # r""" 1-env, 1-episode """
    def test_policy(env, pie, max_steps=None, verbose=0, render=0):
        r""" 1-env, 1-episode """
        s, t, d= env.reset()
        cr = 0.0
        if verbose>0: print('[RESET]')
        if render==2: env.render()
        acts=[]
        if max_steps is None: max_steps=inf
        while not (d or t>=max_steps):
            a = pie.predict(s, t)
            s, t, d, r = env.step(a)
            acts.append(a)
            cr+=r
            if verbose>1: 
                print('[STEP]:[{}], Act:[{}], Reward:[{}], Done:[{}], Return:[{}]'.format(t, a, r, d, cr))
            if render==3: env.render()

        if verbose>0: print('[TERM]: Steps:[{}], Return:[{}]'.format(t, cr))
        if (render==1 or render==2): env.render()
        return cr, t, acts
    
    @staticmethod # r""" 1-env, N-episode """
    def eval_policy(env, pie, episodes, max_steps=None, verbose=0, render=0, verbose_result=True):
        r""" 1-env, N-episode """
        test_hist = []
        test_actions = []
        for n in range(episodes):
            #print(f'\n-------------------------------------------------\n[Test]: {n}\n')
            result_ret, restult_steps, acts = __class__.test_policy(env, pie, max_steps=max_steps, verbose=verbose, render=render)
            #print(f'steps:[{result[0]}], reward:[{result[1]}]')
            test_hist.append((result_ret, restult_steps))
            test_actions.append(acts)
            
        test_hist=np.array(test_hist)
        test_returns, test_steps = test_hist[:, 0], test_hist[:, 1]
        
        mean_return =  np.mean(test_returns)
        mean_steps =  np.mean(test_steps)
        if verbose_result:
            print(f'[Test Result]:\n\
            \tTotal-Episodes\t[{episodes}]\n\
            \tMean-Reward\t[{mean_return}]\n\
            \tMedian-Reward\t[{np.median(test_returns)}]\n\
            \tMax-Reward\t[{np.max(test_returns)}]\n\
            \tMin-Reward\t[{np.min(test_returns)}]\n\
            ')
        return mean_return, mean_steps, test_returns, test_steps, test_actions
    
    @staticmethod # r""" M-env, N-episode """
    def validate_policy(envs, pie, episodes, max_steps=None, episodic_verbose=0, episodic_render=0, verbose_result=True):
        r""" M-env, N-episode """
        validate_result = []
        validate_acts = []
        for env in envs:
            mean_return, mean_steps, _, _, acts = __class__.eval_policy(
                env=env, pie=pie, episodes=episodes, max_steps=max_steps, 
                verbose=episodic_verbose, render=episodic_render, 
                verbose_result=verbose_result)
            #print(test_results.describe())
            validate_result.append(( mean_return, mean_steps))
            validate_acts.append(acts)
        validate_result = np.array(validate_result)

        mean_return, mean_steps = np.mean(validate_result[:,0]), np.mean(validate_result[:,1])
        sum_return, sum_steps = np.sum(validate_result[:,0]), np.sum(validate_result[:,1])

        return mean_return, mean_steps, sum_return, sum_steps, validate_acts

    @staticmethod # r""" M-env, 1-episode """
    def validate_policy_once(envs, pie, max_steps=None, episodic_verbose=0, episodic_render=0):
        r""" M-env, 1-episode """
        validate_result = []
        validate_acts = []
        for env in envs:
            reward, steps, acts = __class__.test_policy(env=env, pie=pie, 
                    max_steps=max_steps,  verbose= episodic_verbose, render=episodic_render)
            validate_result.append(( reward, steps ))
            validate_acts.append(acts)
        validate_result = np.array(validate_result)


        mean_return, mean_steps = np.mean(validate_result[:,0]), np.mean(validate_result[:,1])
        sum_return, sum_steps = np.sum(validate_result[:,0]), np.sum(validate_result[:,1])

        return mean_return, mean_steps, sum_return, sum_steps, validate_acts

    @staticmethod
    def validation(envs, pie, episodes, max_steps=None, verbose=0):
        return \
            Eval.validate_policy(envs, pie, episodes=episodes, max_steps=max_steps, validate_verbose=verbose) \
            if episodes>1 else \
            Eval.validate_policy_once(envs, pie, max_steps=max_steps, episodic_verbose=verbose)

    @staticmethod
    @tt.no_grad() 
    def explore_policy(n, env, pie, max_steps=None):
        if max_steps is None: max_steps=inf
        buffer = []
        for _ in range(n):
            s, t, d= env.reset()
            R, A = [], []
            while not (d or t>=max_steps):
                a = pie.predict(s, t)
                s, t, d, r = env.step(a)
                A.append(a)
                R.append(r)
            S = tt.clone(s) if tt.is_tensor(s) else tt.tensor(s, dtype=tt.long)
            buffer.append((S, A, R))
        return buffer

    @staticmethod
    @tt.no_grad() 
    def explore_greedy(n, env, pie, rie, epsilon, erng, max_steps=None):
        if max_steps is None: max_steps=inf
        buffer = []
        for _ in range(n):
            s, t, d= env.reset()
            R, A = [], []
            while not (d or t>=max_steps):
                a = (rie.predict(s, t) if erng.random()<epsilon else pie.predict(s, t))
                s, t, d, r = env.step(a)
                A.append(a)
                R.append(r)
            S = tt.clone(s) if tt.is_tensor(s) else tt.tensor(s, dtype=tt.long)
            buffer.append((S, A, R))
        return buffer



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" DQN Training """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DQN:
    # -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 
    @staticmethod
    def train(
        # value params [T.V]
            pie, 
            pie_optA,
            value_lrsF, 
            value_lrsA,
        # env params (training) [E]
            env,
            gamma,
            polyak,
        # learning params [L]
            epochs,
            batch_size,
            learn_times,
        # explore-exploit [X]
            explore_size,
            explore_seed,
            epsilon_range,
            epsilon_seed, 
        # memory params [M]
            memory,
            memory_capacity, 
            memory_seed, 
            min_memory,
        # validation params [V]
            validations_envs, 
            validation_freq, 
            validation_max_steps,
            validation_episodes,
            validation_verbose, 
        # algorithm-specific params [A]
            double,
            tuf,
            gradient_clipping,
        # result params [R]
            plot_results,
            save_at,
            checkpoint_freq,
            ):
    # -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 
        # [*] setup 
        from math import inf, nan
        from deep.rl import Eval, RIE
        
        has_target = (double or tuf>0)
        val_opt = pie.optimizer(**pie_optA)
        val_lrs = value_lrsF(val_opt, **value_lrsA)
        val_loss = nn.MSELoss() #<-- its important that DQN uses MSELoss only (consider using huber loss)

        # checkpointing
        if save_at: os.makedirs(save_at, exist_ok=True)
        do_checkpoint = ((checkpoint_freq>0) and save_at)

        # validation
        do_validate = ((len(validations_envs)>0) and validation_freq and validation_episodes)
        mean_validation_return, mean_validation_steps = nan, nan
        validation_max_steps = inf if validation_max_steps is None else validation_max_steps

        # ready training
        train_hist = []
        validation_hist = []
        count_hist = []
        learn_count, update_count = 0, 0
        if memory is None: memory = []
        mrng = np.random.default_rng(memory_seed)
        mselector = np.arange(memory_capacity)
        indexer = tt.arange(env.T, device=pie.device) # required for double dqn
        erng = np.random.default_rng(epsilon_seed) # epsilon-greedy explorer
        epsilonF = Remap(Input_Range=(0,1), Output_Range=epsilon_range)
        rie = RIE(Alow=0, Ahigh=env.A, seed=explore_seed) # <--- random exploration policy
        pie.eval()

        

        # fill up memory with min_memory episondes
        len_memory = len(memory)
        if len_memory < min_memory:
            min_explore = min_memory-len_memory
            memory.extend(Eval.explore_policy(n=min_explore, env=env, pie=rie))
            print(f'[*] Explored Min-Memory [{min_explore}] Steps, Memory Size is [{len(memory)}]')

        #------------------------------------pre-training results
        if do_checkpoint:
            check_point_as =  os.path.join(save_at, f'pre.pie')  
            pie.save(check_point_as)
            print(f'Checkpoint @ {check_point_as}')

        if do_validate:
            mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps, validate_acts = \
                Eval.validation(validations_envs, pie, validation_episodes, validation_max_steps, validation_verbose)
               
            validation_hist.append((mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps))
            print(f' [Pre-Validation]')
            print(f' => (MEAN) :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
            print(f' => (SUM)  :: Return:{sum_validation_return}, Steps:{sum_validation_steps}')
        #------------------------------------pre-training results
        
        for epoch in tqdm(range(epochs)):
            epoch_ratio = epoch/epochs
        # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

            # eploration phase
            epsilon = epsilonF.forward(epoch_ratio)
            memory.extend(Eval.explore_greedy(n=explore_size, env=env, pie=pie, rie=rie, epsilon=epsilon, erng=erng))
            extra = len(memory) - memory_capacity
            if extra>0: del memory[0:extra]
            
            
            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            # Learning update
            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            pie.train()
            len_memory = len(memory)
            val_losses = []
            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            for _ in range(learn_times):
                
                # [1] pick random batch from memory # items in memory are like :: (S, A, R)
                i_memory = mrng.choice(mselector[0:len_memory], size=batch_size, replace=(len_memory<batch_size))

                for im in i_memory:
                    S,A,R = memory[im]
                    
                    S=S.to(dtype=tt.long, device=pie.device)
                    A=tt.tensor(A, dtype=tt.long, device=pie.device)
                    R=tt.tensor(R, dtype=tt.float32, device=pie.device)
                    #print(f'{S.shape=}\n{S=}')
                    #print(f'{len(A)=}\n{A=}')
                    #print(f'{len(R)=}\n{R=}')
                    # [2] compute target Q-values I = tt.arange(0, pick)
                    
                    baseQ =   pie.forward(S, target=False) # Q-values of state (base network has grad)

                    if pie.has_target:
                        targetQ = pie.forward(S, target=True)  # Q-values of state (target network has no_grad)
                    else:
                        targetQ = baseQ.clone().detach()
                    #print(f'{baseQ.shape=}\n{baseQ=}')
                    #print(f'{targetQ.shape=}\n{targetQ=}')

                    # we need to update Q-values of all states including terminal state
                    # base_S is shape (T, nA), each item is Q-value of one state
                    # Q -> R + gamma * Max(Q(nS))  # if Q is not final state
                    # Q -> R                       # if Q is final state
                    updatedQ = tt.zeros_like(R)
                    if not double:
                        maxQ, _ = tt.max(targetQ, dim=-1)
                    else:
                        baseQ_nograd = baseQ.clone().detach()
                        maxQb, maxQbi = tt.max(baseQ_nograd, dim=-1)
                        maxQ = targetQ[indexer,maxQbi] 

                    updatedQ[:-1] = R[:-1] + maxQ[1:]*gamma 
                    updatedQ[-1] = R[-1]
                    predQ = baseQ[indexer, A]
                    #assert(predQ.shape==updatedQ.shape)
                    
                    # [3] update parameters via gradient descent
                    val_opt.zero_grad()
                    loss =  val_loss(predQ, updatedQ) 
                    val_losses.append(loss.item())
                    loss.backward()
                    # Clip gradient norm
                    if gradient_clipping>0.0: nn.utils.clip_grad_norm_(pie.theta.parameters(), gradient_clipping)
                    val_opt.step()
            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            
            train_hist.append((epsilon, val_lrs.get_last_lr()[-1], np.mean(val_losses)))
            count_hist.append((learn_count, update_count, len_memory))
            val_lrs.step()
            learn_count+=1
            if (has_target):
                if learn_count % tuf == 0:
                    pie.update_target(polyak=polyak)
                    update_count+=1

            pie.eval()
            if do_checkpoint:
                if ((epoch+1)%checkpoint_freq==0):
                    check_point_as =  os.path.join(save_at, f'{epoch+1}.pie')  
                    pie.save(check_point_as)
                    print(f'Checkpoint @ {check_point_as}')

            if do_validate:
                if ((epoch+1)%validation_freq==0):
                    mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps, validate_acts = \
                        Eval.validation(validations_envs, pie, validation_episodes, validation_max_steps, validation_verbose)
                    validation_hist.append((mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps))
                    print(f' [Validation]')
                    print(f' => (MEAN) :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
                    print(f' => (SUM)  :: Return:{sum_validation_return}, Steps:{sum_validation_steps}')
        # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

        pie.eval()
        if do_validate:
            mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps, validate_acts = \
                Eval.validation(validations_envs, pie, validation_episodes, validation_max_steps, validation_verbose)
            validation_hist.append((mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps))
            print(f' [Final-Validation]')
            print(f' => (MEAN) :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
            print(f' => (SUM)  :: Return:{sum_validation_return}, Steps:{sum_validation_steps}')

        if save_at:
            save_as = os.path.join(save_at, f'final.pie')
            pie.save(save_as)
            print(f'Saved @ {save_as}')

        validation_hist, train_hist, count_hist = np.array(validation_hist), np.array(train_hist), np.array(count_hist)
        res = dict( train=train_hist, val=validation_hist, count=count_hist )
        if plot_results: _ = __class__.plot_training_result( validation_hist, train_hist, count_hist )
        if save_at:
            save_as = os.path.join(save_at, f'results.npz')
            np.savez( save_as, **res )

        return res
    # -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 



    @staticmethod
    def plot_training_result(validation_hist, train_hist, count_hist):
            tEpsilon, tLR, tLoss = train_hist[:, 0], train_hist[:, 1], train_hist[:, 2]
            vReturn, vSteps = validation_hist[:, 0], validation_hist[:, 1]
            cLearn, cUpdate, cMemory = count_hist[:, 0], count_hist[:, 1], count_hist[:, 2]

            fig, ax = plt.subplots(2,4, figsize=(16,8))

            
            ax_return, ax_steps =  ax[0, 0], ax[0, 1]
            ax_epsilon, ax_lr =    ax[0, 2], ax[0, 3]
            ax_loss, ax_cmemory =  ax[1, 0], ax[1, 1]
            ax_clearn, ax_cupdate= ax[1, 2], ax[1, 3]

            ax_return.plot(vReturn, color='tab:green', label='Return')
            ax_return.scatter(np.arange(len(vReturn)), vReturn, color='tab:green')
            ax_return.legend()

            ax_steps.plot(vSteps, color='tab:blue', label='Steps')
            ax_steps.scatter(np.arange(len(vSteps)), vSteps, color='tab:blue')
            ax_steps.legend()

            ax_epsilon.plot(tEpsilon, color='tab:purple', label='Epsilon')
            ax_epsilon.legend()

            ax_lr.plot(tLR, color='tab:orange', label='Learn-Rate')
            ax_lr.legend()

            ax_loss.plot(tLoss, color='tab:red', label='Value Loss')
            ax_loss.legend()

            ax_cmemory.plot(cMemory, color='tab:pink', label='Memory')
            ax_cmemory.legend()

            ax_clearn.plot(cLearn, color='tab:brown', label='Learn Count')
            ax_clearn.legend()

            ax_cupdate.plot(cUpdate, color='tab:olive', label='Update Count')
            ax_cupdate.legend()

            plt.show()
            return fig

    @staticmethod
    def load_plot_training_result(path):
        res = np.load(path)
        fig = __class__.plot_training_result(res['val'], res['train'], res['count'])
        res.close()
        return fig
