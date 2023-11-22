from math import floor
import numpy as np
import torch as tt
from known.basic import load_pickle


class IntVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to int
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = -1, -2, -3, -4

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        for i,symbol in enumerate(symbols): self.vocab[int(symbol)] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[int(symbol)]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[int(symbol)]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[int(symbol)] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[int(symbol)]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]

class FloatVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to float
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = -1., -2., -3., -4.

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        for i,symbol in enumerate(symbols): self.vocab[float(symbol)] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[float(symbol)]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[float(symbol)]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[float(symbol)] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[float(symbol)]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]

class StrVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to str
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = "<UKN>", "<BOS>", "<EOS>", "<PAD>"

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        
        for i,symbol in enumerate( symbols ): self.vocab[symbol] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[symbol]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[symbol]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[symbol] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[symbol]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]


__all__ = [ 'WorkersEnv' ]

class WorkersEnv:
    r"""
    WorkersEnv:

        Given a sequence of tasks ( tokens )
        Place them among workers that have fixed consumption rate per type of task
        Reduce the total finish time
    """

    def __init__(self, task_symbols, duration, task_set, workers, scale, seed=None ) -> None:
        r"""
        # workers is a list of processing time per unit of each type of n_tasks -> list of dict
        # tasks is a list of list of task for each reset
        eg - 
        WorkersEnv(
            task_symbols = "R G B",
            duration = 10,
            task_set =  [
                            "R G R G B B R G R G",
                            "R G R G B B R G R G",
                            "R G R G B B R G R G",
                            "R G R G B B R G R G",
                            "R G R G B B R G R G",
                        ]
            workers =   [
                            dict(R=1, G=2, B=3),
                            dict(R=2, G=1, B=3),
                            dict(R=2, G=2, B=2),
                        ]
            )
        
        """
        self.idtype = tt.long

        self.task_vocab = StrVocab(symbols=task_symbols)

        self.T =duration
        self.task_set=task_set
        self.n_tasks = len(self.task_set)
        self.workers=workers
        self.A = len(self.workers)
        self.worker_vocab = IntVocab(list(range(self.A)))
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
        return self.obs, self.t, False


    def step(self, action):
        self.pending-=1
        tt.clip_(self.pending, min=self.min_pending)
        self.itask = self.S[self.t]
        try:
            self.pending[action]+= self.workers[action][self.itask]
        except:
            print(f'{action=}, {type(action)}')
            raise ValueError
        self.worker_vocab.forward1_(action, self.obsW, self.t+1)
        self.t+=1
        
        
        done = self.t>=self.T
        cost = tt.max(self.pending)
        return self.obs, self.t, done, -float(cost)*self.scale



    @staticmethod
    def get(task_symbols = ["R", "G", "B"], n_workers=3, duration=15, n_states=10, max_time_ratio=0.3, scale=1e-3, rng_seed=None, seed=None):
        rng = np.random.default_rng(rng_seed)
        task_set=[ tuple([task_symbols[i] for i in rng.integers(0, len(task_symbols), size=duration)]) for _ in range(n_states) ]
        max_time = max(2, int(max_time_ratio*duration))
        workers =   [ {k:rng.integers(1, max_time) for k in task_symbols} for _ in range(n_workers) ]
        return __class__(task_symbols=task_symbols, duration=duration, task_set=task_set, workers=workers, scale=scale, seed=seed)



# class MWorkersEnv:
#     r"""
#     WorkersEnv:

#         Given a sequence of tasks ( tokens )
#         Place them among workers that have fixed consumption rate per type of task
#         Reduce the total finish time
#     """

#     def __init__(self, task_symbolsL, duration, task_setL, workers, seed=None ) -> None:
#         r"""
#         # workers is a list of processing time per unit of each type of n_tasks -> list of dict
#         # tasks is a list of list of task for each reset
        
#         """
#         self.idtype = tt.long
        
#         self.task_vocabs = [StrVocab(symbols=task_symbols) for task_symbols in task_symbolsL] 
#         self.nT = len(task_setL)

#         self.T =duration
#         self.task_sets=task_setL
#         self.n_tasks = len(self.task_sets[0])
#         self.workers=workers
#         self.A = len(self.workers)
#         self.worker_vocab = IntVocab(list(range(self.A)))
#         self.rng = np.random.default_rng(seed)
#         self.frozen = None

#         self.obs_dim = self.nT*self.T + self.T + 1
#         self.obs_shape = (self.obs_dim,)
#         self.obs_dtype = tt.long
#         self.obs = tt.zeros(self.obs_shape, dtype=self.obs_dtype)
#         self.obsS = self.obs[:self.nT*self.T].reshape(self.nT, self.T)
#         self.obsW = self.obs[self.T:]
#         self.pending = tt.tensor([0 for _ in range(self.A)])
#         self.min_pending = tt.zeros_like(self.pending)
#         #self.info={}

#     def freeze(self, I): 
#         if I<0: I = self.n_tasks+I
#         self.frozen = I
#         return self

#     def reset(self, S=None):
#         if S is None: S = [self.task_sets[si][(self.rng.integers(0, self.n_tasks) if (self.frozen is None) else self.frozen)] for si in range(self.nT)]
#         self.S = [self.task_vocabs[si].backward(self.task_vocabs[si].forward(S[si])) for si in range(self.nT)]
#         for si in range(self.nT): self.task_vocabs[si].forward_(S[si], self.obsS[si])

#         W = [self.worker_vocab.BOS] + [self.worker_vocab.UKN for _ in range(self.T)]
#         self.worker_vocab.forward_(W, self.obsW) 
        
#         # tracker
#         self.pending[:] = 0
#         self.t = 0
#         return self.obs, self.t, False


#     def step(self, action):
#         self.pending-=1
#         tt.clip_(self.pending, min=self.min_pending)
#         self.itask = [self.S[si][self.t] for si in range(self.nT)]
#         try:
#             self.pending[action]+= sum([self.workers[action][itask] for itask in self.itask])
#         except:
#             print(f'{action=}, {type(action)}')
#             raise ValueError
#         self.worker_vocab.forward1_(action, self.obsW, self.t+1)
#         self.t+=1
        
        
#         done = self.t>=self.T
#         cost = tt.max(self.pending)
#         return self.obs, self.t, done, -float(cost)



#     @staticmethod
#     def get(
#         task_symbolsL = [["R", "G", "B", "X"], ["r", "g", "b"]], 
#         n_workers=3, 
#         duration=15, 
#         n_states=10, 
#         max_time_ratio=0.3, 
#         rng_seed=None, 
#         seed=None):
#         rng = np.random.default_rng(rng_seed)
#         task_set=[[ tuple([task_symbols[i] for i in rng.integers(0, len(task_symbols), size=duration)]) for _ in range(n_states) ]\
#                   for task_symbols in task_symbolsL]
#         max_time = max(2, int(max_time_ratio*duration))
#         all_task_syms = []
#         for task_symbols in task_symbolsL: all_task_syms.extend(task_symbols)
#         workers =   [ {k:rng.integers(1, max_time) for k in all_task_syms} for _ in range(n_workers) ]
#         return __class__(task_symbolsL=task_symbolsL, duration=duration, task_setL=task_set, workers=workers, seed=seed)


# # @tt.no_grad()
# # @tt.no_grad()
# # def predict(model, gene, n_steps,  di, do, cc, ll, bos=0, verbose=0, rng=None, device=None):
# #     model.eval()
#     pred_tgt_mask=model.generate_square_subsequent_mask(n_steps, device)
#     pred_steps = []
#     di, do, cc, ll = \
#     tt.tensor(gene.di_vocab.forward(di), dtype=tt.long, device=device).unsqueeze(0), \
#     tt.tensor(gene.do_vocab.forward(do), dtype=tt.long, device=device).unsqueeze(0), \
#     tt.tensor(gene.cc_vocab.forward(cc), dtype=tt.long, device=device).unsqueeze(0), \
#     tt.tensor(gene.ll_vocab.forward(ll), dtype=tt.long, device=device).unsqueeze(0)
#     aa = tt.zeros((1,n_steps+1), dtype=tt.long, device=device)
#     last_pred = bos
#     #print(f'{di.shape}, {ll.shape}, {aa.shape}')
#     for pos in range(n_steps):
#         aa[:, pos] = last_pred
#         aa[:, pos+1:] = bos
        
#         pp = model.forward(
#             srcs=(di, do, cc, ll),
#             tgt=aa[:,:-1],
#             src_mask=None,
#             tgt_mask=pred_tgt_mask,
#             src_key_padding_mask=None,
#             tgt_key_padding_mask=None,
#         ).view(-1, gene.n_classes)
#         tk = tt.argmax(tt.softmax(pp, dim=-1), dim=-1)
#         last_pred = tk[pos].item()
#         pred_steps.append(last_pred)
#         if verbose>1: print(f'{pos=}, {last_pred=}\ndecoder-in: {aa}\nPredictions: {tk=}\n\n')

#     placement = gene.aa_vocab.backward(pred_steps).cpu()
#     if verbose>0: print(f'\n{pred_steps=}\n{placement=}')
#     if rng is not None:
#         invalid_placements = tt.where(placement<0)[0]
#         for invalid_i in invalid_placements: placement[invalid_i] = rng.integers(0, gene.n_actions)
#         if verbose>0: print(f'Final {placement=}')
#     return pred_steps, placement.numpy()