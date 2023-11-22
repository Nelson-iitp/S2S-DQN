import maco, deep
import torch as tt
import torch.nn as nn



def tenvF(): return maco.WorkersEnv.get(duration=60, n_workers=4, n_states=5, rng_seed=10293874756, max_time_ratio=1.0)
def venvF(): return maco.WorkersEnv.get(duration=60, n_workers=4, n_states=5, rng_seed=10293874756, max_time_ratio=1.0).freeze(0)
def venvsF(): return [ maco.WorkersEnv.get(duration=60, n_workers=4, n_states=50, rng_seed=10293874756, max_time_ratio=1.0).freeze(f) for f in range(5)]

env = tenvF()
def pieF(**factory): return deep.rl.S2SDQN(
    n_actions = env.A,
    embed_dim = 64,

    encoder_block_size = env.T,
    encoder_vocab_count = env.task_vocab.count,
    encoder_hidden_dim = 128,
    encoder_activation = nn.GELU(),
    encoder_num_heads = 4,
    encoder_num_layers = 1,
    encoder_norm_eps = 0.0, # final norm # keep zero to not use norm
    encoder_dropout = 0.0,
    encoder_norm_first = True,

    decoder_block_size = env.T,
    decoder_vocab_count = env.worker_vocab.count,
    decoder_hidden_dim = 128,
    decoder_activation = nn.GELU(),
    decoder_num_heads = 4,
    decoder_num_layers = 6,
    decoder_norm_eps = 0.0, # final norm # keep zero to not use norm
    decoder_dropout = 0.0,
    decoder_norm_first = True,

    dense_layer_dims = [128 ,128, 128],
    dense_actFs = [nn.Tanh(), nn.ReLU() ],
    dense_bias = True,

    xavier_init = False,

    has_target = True, 
    **factory)
