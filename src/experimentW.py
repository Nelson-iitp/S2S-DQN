
import os
import maco

import s2sdqn
import torch as tt
import torch.nn as nn

import numpy as np

auto_device = 'cuda' if tt.cuda.is_available() else 'cpu'

class Exp:
    auto_device=   auto_device
    def __init__(self, alias, device=None) -> None:
        self.alias =         alias
        
        self.factory =       dict(dtype=tt.float32, device = device)
        self.env =           self._Env()
        self.A =             self.env.A
        self.T =             self.env.T
        self.n_tasks =       self.env.n_tasks
        

    def _Env(self, rng_seed=None, env_seed=None): return maco.WorkersEnv.get(
                n_workers=2,
                duration=12,
                n_states=5,
                scale=0.1,
                rng_seed=rng_seed, env_seed=env_seed )
        
    def tEnv(self, rng_seed=None, env_seed=None): return self._Env( rng_seed=rng_seed, env_seed=env_seed )

    def iEnv(self, i, rng_seed=None, env_seed=None): return self._Env( rng_seed=rng_seed, env_seed=env_seed ).freeze(i)

    # # validation envs
    # def vEnvs(self, rng_seeds=None, env_seeds=None):
    #     if rng_seeds is None: rng_seeds=[None for _ in range(self.n_tasks)]
    #     if env_seeds is None: env_seeds=[None for _ in range(self.n_tasks)]
    #     if len(rng_seeds)==1: rng_seeds = [rng_seeds[0] for _ in range(self.n_tasks)]
    #     if len(env_seeds)==1: env_seeds = [env_seeds[0] for _ in range(self.n_tasks)]
    #     return [ self._Env(rng_seed=rng_seeds[i], env_seed=env_seeds[i] ).freeze(i) for i in range(self.n_tasks) ]

    

class ExpA(Exp):

    def __init__(self, device=None) -> None:
        super().__init__(
                alias='A',
                device=device)
        
        self.coderArgs={
            'd_model':          128,
            'nhead':            8,
            'dim_feedforward':  64*4,
            'dropout':          0.0,
            'activation':       "gelu",
            'normF':            s2sdqn.ktf.ex.Norm.layerN,
            'normA':            {'bias': True, 'eps': 1e-06},
            'norm_first':       True,
            }

    def encoder(self):
        return  s2sdqn.ktf.Encoder(
                vocab_size=self.env.encoder_vocab_size,
                pose=s2sdqn.ktf.Pose.TrainableLinear(
                    input_size=self.coderArgs['d_model'],
                    block_size=self.T, 
                    **self.factory),
                coderA=self.coderArgs,
                num_layers=1,
                norm=None,
                **self.factory) 

    def decoder(self):
        return s2sdqn.ktf.Decoder(
            vocab_size=self.env.decoder_vocab_size,
            pose=s2sdqn.ktf.Pose.TrainableLinear(
                    input_size=self.coderArgs['d_model'],
                    block_size=self.T, 
                    **self.factory),
                coderA=self.coderArgs,
                num_layers=3,
                norm=None,
                **self.factory)
    
    def theta(self):
        return s2sdqn.ktf.Seq2SeqTransformer(
            custom_encoder=self.encoder(),
            custom_decoder=self.decoder(),
            n_outputs=self.A,
            dense_layer_dims=[128, 128, 128 ],
            dense_actFs=[ nn.Tanh(), nn.ReLU() ],
            dense_bias=True,
            xavier_init=False,
            **self.factory
        )
    
    def pie(self):
        return s2sdqn.pie.S2SDQN(
            encoder_block_size=self.T,
            decoder_block_size=self.T,
            value_theta=self.theta(),
            has_target=True,
            **self.factory
        )

    def optim(self, pie, learning_rate, weight_decay=0.0):
        return s2sdqn.ktf.Optimizers.GPTAdamW(
            model=pie.theta,
            device=self.factory.get('device', 'cpu'),
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(0.9, 0.999))

    def train(self,
        tenv,
        venvs,
        pie,
        learning_rate, 
        weight_decay=0.0,
        epochs = 10_00,
        batch_size = 32,
        learn_times = 2,
        tuf = 4,
        epsilon_range = (1.0, 0.1),
        explore_pies = None,
        explore_per_pie = True,
        min_memory_pies=  None,
        double = False,
        validation_interval = 0.1,
        checkpoint_interval = 0.1,
        gradient_clipping = 0.5,
        save_at =       '__results__',
    ):
        optim = self.optim(pie, learning_rate=learning_rate, weight_decay=weight_decay)
        lrs = tt.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=epochs)

        return s2sdqn.dqn.Train(
            # value params [T.V]
                pie =                   pie, 
                pie_opt =               optim,
                pie_lrs =               lrs,
            # env params (training) [E]
                env =                   tenv,
                gamma =                 1.0,
                polyak =                0.0,
            # learning params [L]
                epochs =                epochs,
                batch_size =            batch_size,
                learn_times =           learn_times,
            # explore-exploit [X]
                explore_size =          batch_size*learn_times,
                explore_seed =          None,
                epsilon_range =         epsilon_range,
                epsilon_seed =          None, 
                explore_pies=           explore_pies,
                explore_per_pie=        explore_per_pie,        
            # memory params [M]
                memory =                None,
                memory_capacity =       int(1e6), 
                memory_seed =           None, 
                min_memory =            int(1e3),
                min_memory_pies=        min_memory_pies,
            # selector
                selector_seed=          None,
            # validation params [V]
                validations_envs =      venvs, 
                validation_freq =       int(validation_interval*epochs), 
                validation_max_steps =  None,
                validation_episodes =   1,
                validation_verbose =    False, 
            # algorithm-specific params [A]
                double =                double,
                tuf =                   tuf,
                gradient_clipping =     gradient_clipping,
            # result params [R]
                plot_results =          True,
                save_at =               save_at,
                checkpoint_freq =       int(checkpoint_interval*epochs),
        )
        


































