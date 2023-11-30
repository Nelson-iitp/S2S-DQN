

from math import floor
import numpy as np
import torch as tt
from scipy.sparse.csgraph import floyd_warshall
from known.ktf import Vocab

__all__ = [
    'ComputeNetwork',
    'Infra',
    'Env',
    'Simulator',
    'Environment',
    'WorkersEnv',

]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class ComputeNetwork:
    r""" represnts a network of computing units.
    each unit has a processing capacity in Hz or cycles-per-sec.
    units are connected to others with bandwidth in bits-per-sec.
    Its is assumed that time unit is seconds but can be changed."""

    DTYPE = np.float32

    def __init__(self, n_units, default_cc=1.0, default_bw=1.0, default_decimals=None) -> None:
        self.n_units = n_units
        self.default_cc, self.default_bw, self.default_decimals = default_cc, default_bw, default_decimals
        self.cc = np.zeros((n_units,), dtype=__class__.DTYPE) 
        self.bw = np.zeros((n_units,n_units), dtype=__class__.DTYPE) 
        #self.rng = np.random.default_rng(seed)
        #self.dr = np.zeros_like(self.bw)

        self.cc+=default_cc
        self.bw+=default_bw
        np.fill_diagonal(self.bw, 0) # a dense network
        self.update_effective_bandwidth()

    def update_effective_bandwidth(self):
        #<--- effective bandwidth, 
        # inf means zero transmission time (soure == target)
        # zero means inf transmission time (no connection)
        bw = 1/floyd_warshall(csgraph=(1/self.bw), directed=True, return_predecessors=False) 
        self.bw_ = bw if (self.default_decimals is None) else np.round(bw, decimals=self.default_decimals)

    def set_resources(self, cc, bw):
        self.bw[:], self.cc[:]  = bw, cc
        self.update_effective_bandwidth()

    def is_connected(self, n1, n2): return (self.bw[n1, n2]!=0)

    def get_neighbours(self, n): return np.where(self.bw[n]!=0)[0]

    def ngF(self, n): return np.where(self.bw[n, 0:self.mec.E]!=0)[0]

    def render(self):
        print(f'[NET]')
        print(f'Units: {self.n_units}')
        print(f'CC:\n{self.cc}')
        print(f'BW:\n{self.bw}')
        print(f'EBW:\n{self.bw_}')

    def graph(self, show_ebw=False, show_nodes=False, show_edges=False, save_format=None):
        from graphviz import Digraph
        dot = Digraph(name='network')
        # add nodes and edges
        for i in range(self.n_units):
            if show_nodes: dot.node(f'{i}', f'Node-{i}\n{self.cc[i]} cc')
            else:          dot.node(f'{i}', f'Node-{i}')

        bw = self.bw_ if show_ebw else self.bw
        for i in range(self.n_units):
            for j in range(self.n_units):

                if bw[i,j]: 
                    if show_edges: dot.edge(f'{i}', f'{j}', f'{bw[i,j]}')
                    else:          dot.edge(f'{i}', f'{j}')

        if save_format is not None:
            dot.format = save_format
            dot.render(directory=f'{dot.name}')
        return dot
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Infra:
    
    def render(self):
        print ('[INFRA]\nA: {}, E: {}, C: {},\nVR:{}\nDR:\n{}'.format(self.A, self.E, self.C, self.VR, self.DR))

    def __init__(self, E, C, params) -> None:
        self.E, self.C = E, C
        self.A = self.E+self.C
        self.params = params
        #for atr in ('BEC','BEE','BCC', 'BUE', 'CE','CC'): setattr(self, atr, getattr(params, atr))
        self.VR =  np.array(([self.params.CE for _ in range(E) ] + [self.params.CC for _ in range(C)]), dtype='float') 
        self.DR = np.zeros((self.A, self.A), dtype='float')
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    def is_edge(self, n): return n<self.E
    def connect(self, n1, n2, bidir=True):
        if n1<self.E:
            self.DR[n1, n2] = self.params.BEE if n2<self.E else self.params.BEC
        else:
            self.DR[n1, n2] = self.params.BEC if n2<self.E else self.params.BCC
        if bidir: self.DR[n2, n1] = self.DR[n1, n2]
    def connect_mesh(self):
        for i in range(self.A):
            for j in range(i, self.A):
                self.connect(i, j)
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" Environment Class Signature for S2S-Env """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Env:
    def __init__(self, A, T) -> None:
        self.A = A #<-- no of actions
        self.T = T #<-- horizon
        self.t = 0     # current time step
        self.state = None   # obs is may be a tuple - should return tensors

    def reset(self, **kwargs): return self.state,  self.t, self.done, 0.0
    def step(self, action, **kwargs): return self.state, self.t, self.done, 0.0
    def render(self, **kwargs): pass




class Simulator:
    
    def __init__(self, 
                net, 
                horizon,
                scale, # scaling factor roh
                quanta, # seconds per time-step
                ) -> None:
        super().__init__()
        self.net = net
        self.scale = scale

        # derive
        self.mec = self.net.mec
        self.E, self.C, self.A = self.mec.E, self.mec.C, self.mec.A
        self.q = quanta
        self.bw_ue = self.mec.params.BUE #bps
        self.large_computation_limit = self.mec.params.large_computation_limit
        self.large_data_limit = self.mec.params.large_data_limit
        self.steps = horizon
    
    def ready(self, state):
        #assert (state.shape == token.shape)
        # initialize load
        self.edge_cc = np.zeros(self.E, dtype=np.float32)
        self.low_cc = np.zeros_like(self.edge_cc)
        self.high_cc = np.zeros_like(self.edge_cc)+np.inf

        self.state = state
        self.di, self.do, self.cc, self.ll = self.state[0, :], self.state[1, :], self.state[2, :], self.state[3, :]
        self.ts = 0
        self.near_edge = int(self.ll[self.ts])

    def step(self, action):

        self.edge_cc[:] =  np.clip(self.edge_cc - self.q*self.net.cc[0:self.E], a_min=self.low_cc, a_max=self.high_cc)
        #action = policy[self.ts]
        #near_edge = int(self.ll[self.ts])
        if action<self.E: # placed at edge
            self.edge_cc[action]+=self.cc[self.ts]
            ex_cost = self.edge_cc[action]/self.net.cc[action] 
        else:ex_cost = self.cc[self.ts]/self.net.cc[action]
        
        tx_cost = self.di[self.ts]/self.net.bw_[self.near_edge, action]
        next_slot = floor((ex_cost + tx_cost + self.di[self.ts]/self.bw_ue)/self.q)
        #print(f'{self.cc[self.ts]}, {next_slot=}')

        if self.ts+next_slot>=self.steps: next_edge = int(self.ll[-1])
        else: next_edge = int(self.ll[self.ts+next_slot])

        rx_cost = self.do[self.ts]/self.net.bw_[action, next_edge]
        #if rx_cost<=0:print(f'{rx_cost=}')
        reward = -self.scale*float(tx_cost+ex_cost+rx_cost)

        self.ts+=1
        done = self.ts>=self.steps
        self.near_edge = -1 if done else int(self.ll[self.ts])
        #print(f'{self.ts}>={self.steps}, {done=}')
        return reward, done, next_slot

class Environment(Env):

    @property
    def encoder_vocab_sizes(self): return self.di_vocab.size, self.do_vocab.size, self.cc_vocab.size, self.ll_vocab.size

    @property
    def decoder_vocab_size(self): return self.aa_vocab.size

    def __init__(self, 
                net, 
                horizon, 
                scale, # scaling factor roh
                quanta, # seconds per time-step
        ) -> None:
        self.sim = Simulator(net=net, horizon=horizon, scale=scale, quanta=quanta)
        super().__init__(self.sim.A, self.sim.steps)

        # generate vocab
        p = self.sim.net.mec.params
        self.di_vocab = Vocab.FloatVocab(p.di_range)
        self.do_vocab = Vocab.FloatVocab(p.do_range)
        self.cc_vocab = Vocab.FloatVocab(p.cc_range)
        self.ll_vocab = Vocab.FloatVocab(np.arange(self.sim.E))
        self.aa_vocab = Vocab.FloatVocab(np.arange(self.A))

        # build states
        block = tt.zeros((self.T,), dtype=tt.long)
        self.DT, self.OT, self.CT, self.ZT = \
        tt.zeros_like(block),tt.zeros_like(block),tt.zeros_like(block),tt.zeros_like(block)
        self.iPOS = np.zeros((self.T+1,)) + self.aa_vocab.UKN
        self.iPOS[0] = self.aa_vocab.BOS
        self.WT = tt.tensor(self.aa_vocab.forward(self.iPOS), dtype=tt.long)
     
    
    def load_states(self, states, frozen=None, seed=None):
        # states is 3-D array (batch, 4, T)
        self.rng = np.random.default_rng(seed)
        self.states = np.load(states) if isinstance(states, str) else states
        self.n_states = len(self.states)
        
        if frozen is not None: # choosees randomly
            if isinstance(frozen, int): frozen = [frozen,]
            else: frozen = [*frozen]
            self.frozen_index = -1
            self.frozen_nos = len(frozen)

        self.frozen = frozen
        self.I, self.S = -1, None
        return self

    def reset(self):
        if (self.frozen is None):
            self.I = self.rng.integers(0, self.n_states)
        else:
            self.frozen_index+=1
            self.I = self.frozen[self.frozen_index%self.frozen_nos]

        self.S = self.states[self.I]

        self.sim.ready(self.S)
        self.t = 0
        self.last_action = self.sim.near_edge - 1

        self.di_vocab.forward_(self.sim.di, self.DT)
        self.do_vocab.forward_(self.sim.do, self.OT)
        self.cc_vocab.forward_(self.sim.cc, self.CT)
        self.ll_vocab.forward_(self.sim.ll, self.ZT)
        self.aa_vocab.forward_(self.iPOS, self.WT)
        return (self.DT, self.OT, self.CT, self.ZT, self.WT[:-1]), self.t, False, 0.0

    def step(self, action):
        reward, done, slot = self.sim.step(action)
        self.t+=1
        self.last_action = action
        self.aa_vocab.forward1_(action, self.WT, self.t)
        return (self.DT, self.OT, self.CT, self.ZT, self.WT[:-1]), self.t, done, reward


    def set_huristic_seed(self, seed): 
        self.hrng = np.random.default_rng(seed)
        return self

    def pie_Huristic_Random(self, *args):
        return self.hrng.integers(0, self.sim.A)

    def pie_Huristic_Random_Edge(self, *args):
        return self.hrng.integers(0, self.sim.E)
    
    def pie_Huristic_Round_Robin_Edge(self, *args):
        return (self.last_action + 1) % self.sim.E

    def pie_Huristic_Round_Robin_Edge_Cloud(self, *args):
        return (self.last_action + 1) % self.sim.A
    
    def pie_Huristic_Zonal_Edge(self, *args):
        # always place on zonal edge
        return self.sim.near_edge   

    def pie_Huristic_Edge(self, *args):
        # always place on edges only
        if self.sim.edge_cc[self.sim.near_edge] <= 0.0:  # if zonal edge edge is free then post there
            return self.sim.near_edge
        else: # find the best edge i.e one with least occupied resources
            return  np.argmin(self.sim.edge_cc)
    
    def pie_Huristic_Edge_Cloud(self, *args):
        # try to complete using edges only, if all are busy then send to clouds

        if self.sim.edge_cc[self.sim.near_edge] <= 0.0:  # if zonal edge edge is free then post there
            return self.sim.near_edge
        else: # find free edges
            free_edges = np.where(self.sim.edge_cc==0)[0]
            if len(free_edges):
                return self.hrng.choice(free_edges, size=1)[0] # choose a random free edge
            else:
                return self.sim.E # send to cloud

    def pie_Huristic_Edge_Cloud_Limited(self, *args):
        # try to complete using edges only, if all are busy then send to clouds

        if self.sim.edge_cc[self.sim.near_edge] <= 0.0:  # if zonal edge edge is free then post there
            return self.sim.near_edge
        else: # find free edges
            free_edges = np.where(self.sim.edge_cc==0)[0]
            if len(free_edges):
                return self.hrng.choice(free_edges, size=1)[0] # choose a random free edge
            else:
                if self.sim.cc[self.sim.ts] > self.sim.large_computation_limit:
                    return self.sim.E # send to cloud
                else: return  np.argmin(self.sim.edge_cc)
                
class WorkersEnv(Env):
    r"""
    WorkersEnv:

        Given a sequence of tasks ( tokens )
        Place them among workers that have fixed consumption rate per type of task
        Reduce the total finish time
    """
    @property
    def encoder_vocab_size(self): return self.task_vocab.size

    @property
    def decoder_vocab_size(self): return self.worker_vocab.size

    def __init__(self, task_symbols, duration, task_set, workers, scale, seed=None ) -> None:
        r"""
        # workers is a list of processing time per unit of each type of n_tasks -> list of dict
        # tasks is a list of list of task for each reset
        """
        
        super().__init__(len(workers), duration)
        self.workers = workers
        self.worker_vocab = Vocab.IntVocab(list(range(self.A)))
        self.task_set=task_set
        self.n_tasks = len(self.task_set)
        self.task_vocab = Vocab.StrVocab(symbols=task_symbols)
        
    
        self.rng = np.random.default_rng(seed)
        self.frozen = None
        self.scale = scale

        self.obs_dim = self.T + self.T + 1
        self.obs_shape = (self.obs_dim,)
        self.obs_dtype = tt.long
        self.obs = tt.zeros(self.obs_shape, dtype=self.obs_dtype)
        self.obsS = self.obs[:self.T]
        self.obsW = self.obs[self.T:]
        self.pending = tt.tensor([0 for _ in range(self.A)])
        self.min_pending = tt.zeros_like(self.pending)
        #self.info={}

    def freeze(self, I): 
        if I<0: I = self.n_tasks+I
        self.frozen = I
        return self

    def reset(self, S=None):
        if S is None: S = self.task_set[(self.rng.integers(0, self.n_tasks) if (self.frozen is None) else self.frozen)]
        self.S = self.task_vocab.backward(self.task_vocab.forward(S))
        self.task_vocab.forward_(S, self.obsS)

        W = [self.worker_vocab.BOS] + [self.worker_vocab.UKN for _ in range(self.T)]
        self.worker_vocab.forward_(W, self.obsW) 
        
        # tracker
        self.pending[:] = 0
        self.t = 0
        return (self.obsS, self.obsW[:-1]), self.t, False, 0.0


    def step(self, action):
        self.pending-=1
        tt.clip_(self.pending, min=self.min_pending)
        self.itask = self.S[self.t]
        try:
            self.pending[action]+= self.workers[action][self.itask]
        except:
            print(f'{action=}, {type(action)}')
            raise ValueError
        self.t+=1
        self.worker_vocab.forward1_(action, self.obsW, self.t)
        
        done = self.t>=self.T
        cost = tt.max(self.pending)
        return (self.obsS, self.obsW[:-1]), self.t, done, -float(cost)*self.scale



    @staticmethod
    def get(
        task_symbols = ["R", "G", "B"], 
        n_workers=4, 
        duration=48, 
        n_states=10, 
        max_time_ratio=0.3, 
        scale=1e-3, 
        rng_seed=None, 
        env_seed=None):
        rng = np.random.default_rng(rng_seed)
        task_set=[ tuple([task_symbols[i] for i in rng.integers(0, len(task_symbols), size=duration)]) for _ in range(n_states) ]
        max_time = max(2, int(max_time_ratio*duration))
        workers =   [ {k:rng.integers(1, max_time) for k in task_symbols} for _ in range(n_workers) ]
        return __class__(task_symbols=task_symbols, duration=duration, task_set=task_set, workers=workers, scale=scale, seed=env_seed)


