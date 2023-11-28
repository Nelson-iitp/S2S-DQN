
import os
import maco

import s2sdqn
import torch as tt
import torch.nn as nn

import numpy as np

auto_device = 'cuda' if tt.cuda.is_available() else 'cpu'

class Exp:
    auto_device=   auto_device
    def __init__(self, alias, network, ds_name, scale, quanta, n_steps, device=None) -> None:
        self.alias =         alias
        self.network =       network
        self.params =        maco.config.GLOBAL_PARAMS
        self.A =             self.network.mec.A
        self.T              = n_steps

        self.ds_dir =        '__data__'
        self.ds_name =       ds_name
        self.ds_path =       os.path.join(self.ds_dir, f'{self.ds_name}.npy')

        self.scale =         scale # scale the cost/reward
        self.quanta =        quanta # seconds, duration of 1 time slot
        self.factory =      dict(dtype=tt.float32, device = device)
        self.env =          self._Env()
    
    def set_ds(self, ds_name, ds_path):

        self.ds_dir =        ds_path
        if ds_name is not None: self.ds_name =       ds_name
        self.ds_path =       os.path.join(self.ds_dir, f'{self.ds_name}.npy')
        return self

    def list_Huristic(self):
        return [ s for s in dir(self.env) if s.startswith('pie_Huristic_') ]


    def create_ds(self, n_apps, n_paths, app_seed, path_seed):
        # generate dataset of states
        os.makedirs(self.ds_dir, exist_ok=True)
        maco.config.Gene.make_ds(
            path=       self.ds_path,
            network=    self.network,
            n_apps=     n_apps,
            n_paths=    n_paths,
            n_steps=    self.T,
            app_seed=   app_seed,
            path_seed=  path_seed,   )

    
    def load_ds(self, split_ratio=0.8):
        self.ds_states =     np.load(self.ds_path)
        self.n_states, n_input_seq, T = self.ds_states.shape
        assert(n_input_seq==4)
        assert(T==self.T)

        split =        int(split_ratio*self.n_states) # for training testing split
        self.ds_train, self.ds_test =     self.ds_states[:split], self.ds_states[split:]
        return self


    def _Env(self):
        return maco.Environment(
            net=self.network,
            horizon=self.T,
            scale=self.scale,
            quanta=self.quanta, )
        
    def tEnv(self, seed=None, heed=None):
        # training env
        return self._Env().load_states(
            states=self.ds_train,
            frozen=None,
            seed=seed, ).set_huristic_seed(heed)


    # validation envs
    def vEnv(self, heed=None):
        return self._Env().load_states(
            states=self.ds_test,
            frozen=list(range(len(self.ds_test))), #<--- for sequential test
            seed=None, ).set_huristic_seed(heed)

    # validation envs
    def vEnvs(self, heeds=None):
        if heeds is None: heeds=[None for _ in range(len(self.ds_test))]
        return [self._Env().load_states(
            states=self.ds_test,
            frozen=i, #<--- for sequential test
            seed=None, ).set_huristic_seed(heeds[i]) for i in range(len(self.ds_test)) ]

    # single frozen env
    def sEnv(self, i:int, heed=None):
        return self._Env().load_states(
            states=self.ds_states,
            frozen=int(i), #<--- for sequential test
            seed=None,  ).set_huristic_seed(heed)
    

class ExpA(Exp):

    def __init__(self, device=None) -> None:
        super().__init__(
                alias='A',
                network=maco.config.db.X3e1c(),
                ds_name='X3e1c_A',
                scale=0.01, quanta=5.0, n_steps=48, device=device)
        
        self.coderArgsE={
            'd_model':          32,
            'nhead':            4,
            'dim_feedforward':  64,
            'dropout':          0.1,
            'activation':       "gelu",
            'normF':            s2sdqn.ktf.ex.Norm.layerN,
            'normA':            {'bias': True, 'eps': 1e-06},
            'norm_first':       True,
            }
        self.coderArgsD={
            'd_model':          32*4,
            'nhead':            8,
            'dim_feedforward':  64*4,
            'dropout':          0.1,
            'activation':       "gelu",
            'normF':            s2sdqn.ktf.ex.Norm.layerN,
            'normA':            {'bias': True, 'eps': 1e-06},
            'norm_first':       True,
            }
        
        self.thetaArgs=dict(
            catenate=True,
            dense_layer_dims=[64*4, 64*4, 64*4 ],
            dense_actFs=[ nn.Tanh(), nn.ELU() ],
            dense_bias=True,
            #edl_mapping=list(range(len(self.env.encoder_vocab_sizes)))+list(range(len(self.env.encoder_vocab_sizes))),
            xavier_init=False,
        )
        
    def encoders(self):
        return [
            s2sdqn.ktf.Encoder(
                vocab_size=encoder_vocab_size,
                pose=s2sdqn.ktf.Pose.TrainableLinear(
                    input_size=self.coderArgsE['d_model'],
                    block_size=self.T, 
                    **self.factory),
                coderA=self.coderArgsE,
                num_layers=2,
                norm=None,
                **self.factory) \
            for encoder_vocab_size in self.env.encoder_vocab_sizes]

    def decoder(self):
        return s2sdqn.ktf.Decoder(
            vocab_size=self.env.decoder_vocab_size,
            pose=s2sdqn.ktf.Pose.TrainableLinear(
                    input_size=self.coderArgsD['d_model'],
                    block_size=self.T, 
                    **self.factory),
                coderA=self.coderArgsD,
                num_layers=8,
                norm=None,
                **self.factory)
    
    def theta(self):
        return s2sdqn.ktf.MultiSeq2SeqTransformer(
            custom_encoders=self.encoders(),
            custom_decoder=self.decoder(), 
            n_outputs=self.A,
            **self.thetaArgs,
            **self.factory
        )
    
    def pie(self):
        return s2sdqn.pie.MS2SDQN(
            encoder_block_size=[self.T for _ in range(4)],
            decoder_block_size=self.T,
            value_theta=self.theta(),
            has_target=True,
            **self.factory
        )

    def optim(self, pie, learning_rate=1e-4, weight_decay=0.0):
        return s2sdqn.ktf.Optimizers.GPTAdamW(
            model=pie.theta,
            device=self.factory.get('device', 'cpu'),
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(0.99, 0.999))

    def train(self,
        tenv,
        venvs,
        pie,
        learning_rate=1e-4, 
        lref=0.1,
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
        lrs = tt.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=lref, total_iters=epochs)

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
                memory_capacity =       int(1e5), 
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
        


































