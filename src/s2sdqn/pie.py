
# ========================================================
# S2S-DQN : Sequence-to-Sequence Deep Q Network
# ========================================================
# pie.py
# Implements Q-Value based discrete policies
# Implements other type of policies like Random and Fixed
# Implements policy evaluation methods
# ========================================================
# Author: Nelson.S
# ========================================================


import numpy as np
import torch as tt
import matplotlib.pyplot as plt
from math import inf
from known.ktf import Mod

__all__ = [
    'RandomPie',
    'FixedPie',
    'ValuePie',
    'S2SDQN',
    'MS2SDQN',
    'Eval',
]

class RandomPie:
    r""" Implements random policy """
    def __init__(self, Alow, Ahigh, seed=None) -> None: self.Alow, self.Ahigh, self.rng = Alow, Ahigh, np.random.default_rng(seed)
    def __call__(self, *args): return self.rng.integers(self.Alow, self.Ahigh)

class FixedPie:
    r""" Implements constant policy """
    def __init__(self, Aseq) -> None:  self.Aseq = Aseq
    def __call__(self, s, t): return self.Aseq[t]
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [*] Base Value Netowrk Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ValuePie: 
    """ base class for value estimators 
        NOTE: for Q-Values, underlying parameter network ```value_theta``` should accept state as input
    """

    def Count(self, requires_grad=None): return Mod.Count(self.theta, requires_grad=requires_grad)

    def State(self, values=False): return Mod.State(self.theta, values=values)

    def Show(self, values=False): return Mod.Show(self.theta, values=values)

    def _build_target(self): return (Mod.SetGrad(Mod.Clone(self.theta), False) if self.has_target else self.theta )

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

    def __call__(self, state): # <---- called in explore mode
        _, _, action = self.predict(state)
        return action

    def predict(self, state): # <---- called in explore mode
        state = state.to(device=self.device) #<-- add batch dim
        out = self.forward(state)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX.item()
    
    #def optimizer(self, optF, optA): return optF(self.theta.parameters(), **optA)

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [*] Sequence-to-Sequence DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class S2SDQN(ValuePie):

    def __init__(self, 
        encoder_block_size, 
        decoder_block_size,
        value_theta,
        has_target = False, 
        dtype = None, 
        device = None):
        self.factory = {'device': device, 'dtype': dtype}
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++
        super().__init__(value_theta, has_target, dtype, device)
        self.encoder_block_size = encoder_block_size
        self.decoder_block_size = decoder_block_size
        self.causal_mask = tt.triu(tt.full((decoder_block_size, decoder_block_size), -tt.inf, **self.factory), diagonal=1)
        # ==========================================================

    def forward(self, state, target=False): #<-- called in batch mode
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        if target: 
            out = self.theta_(        
                src=state[0], 
                tgt=state[-1], 
                src_mask = None, 
                tgt_mask  = self.causal_mask,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,)
        else:      
            out = self.theta (        
                src=state[0], 
                tgt=state[-1], 
                src_mask = None, 
                tgt_mask  = self.causal_mask,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,)
        return out
    
    def __call__(self, state, ts): # <---- called in explore mode
        _, _, action = self.predict(state, ts)
        return action

    def predict(self, state, ts): # <---- called in explore mode
        if tt.is_tensor(state): state = state.to(device=self.device) #<-- add batch dim
        else:                   state = [s.to(device=self.device) for s in state] #<-- add batch dim
        out = self.forward(state, target=False)
        outX = tt.argmax(out, dim=-1)
        return out, outX, outX[ts].item()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [*] Multi Sequence-to-Sequence DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MS2SDQN(S2SDQN):

    def forward(self, state, target=False): #<-- called in batch mode
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        if target: 
            out = self.theta_(        
                src=state[:-1], 
                tgt=state[-1], 
                src_mask = None, 
                tgt_mask  = self.causal_mask,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,)
        else:      
            out = self.theta (        
                src=state[:-1], 
                tgt=state[-1], 
                src_mask = None, 
                tgt_mask  = self.causal_mask,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,)
        return out
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [*] Multi MLP DQN Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MMLPDQN(S2SDQN):

    def forward(self, state, target=False): #<-- called in batch mode
        #key_padding_mask = state[:, self.sep_mask_L:self.sep_mask_H]
        print(f'{state=}')

        x = tt.cat(state, dim=-1).to(**self.factory) #<--- catentate all input sequences
        print(f'{x.shape=}, {x=}')
        if target: out = self.theta_(x)
        else:      out = self.theta (x)
        return out
    
""" [*] Policy Evaluation/Testing 
    NOTE: make sure to put policies in .eval() mode before predicting
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Eval:

    @staticmethod
    @tt.no_grad() # r""" 1-env, 1-episode """
    def test_solution(env, acts, max_steps=None, verbose=0, render=0):
        r""" 1-env, 1-episode """
        s, t, d, cr = env.reset()
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
        s, t, d, cr = env.reset()
        if verbose>0: print('[RESET]')
        if render==2: env.render()
        acts=[]
        if max_steps is None: max_steps=inf
        if isinstance(pie, str): pie = getattr(env, pie)
        while not (d or t>=max_steps):
            a = pie(s, t)
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
            __class__.validate_policy(envs, pie, episodes=episodes, max_steps=max_steps, verbose_result=verbose) \
            if episodes>1 else \
            __class__.validate_policy_once(envs, pie, max_steps=max_steps, episodic_verbose=verbose)

    @staticmethod
    @tt.no_grad() 
    def explore_policy(n, env, pie, max_steps=None):
        if max_steps is None: max_steps=inf
        if isinstance(pie, str): pie = getattr(env, pie)
        buffer = []
        for _ in range(n):
            s, t, d, _ = env.reset()
            R, A = [], []
            while not (d or t>=max_steps):
                a = pie(s, t)
                s, t, d, r = env.step(a)
                A.append(a)
                R.append(r)
            S = tt.clone(s) if tt.is_tensor(s) else tuple([tt.clone(si) for si in s])
            buffer.append((S, A, R))
        return buffer

    @staticmethod
    @tt.no_grad() 
    def explore_greedy(n, env, pie, rie, epsilon, erng, max_steps=None):
        if max_steps is None: max_steps=inf
        if isinstance(rie, str): rie = getattr(env, rie)
        if isinstance(pie, str): pie = getattr(env, pie)
        buffer = []
        for _ in range(n):
            s, t, d, _ = env.reset()
            R, A = [], []
            while not (d or t>=max_steps):
                a = (rie(s, t) if erng.random()<epsilon else pie(s, t))
                s, t, d, r = env.step(a)
                A.append(a)
                R.append(r)
            S = tt.clone(s) if tt.is_tensor(s) else tuple([tt.clone(si) for si in s])
            buffer.append((S, A, R))
        return buffer

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

