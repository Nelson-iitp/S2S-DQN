

# ========================================================
# S2S-DQN : Sequence-to-Sequence Deep Q Network
# ========================================================
# fda.py
# Implements Flow Direction Algorithm (FDA) 
# ========================================================
# Author: Nelson.S
# ========================================================


import numpy as np
#import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now

def strA(arr, start="[", sep=",", end="]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for i in range(len(arr)):
        res+=str(arr[i])+sep
    return res + end

def l2(x): # L2 norm of difference vector (Euclidian distance b/w vectors)
    return np.sum(x**2)**0.5

def random_flow(lb, ub, ndim): # a function for generating a random solution
    return lb + np.random.uniform(0,1,size=ndim)*(ub-lb)

def optimize(MAXITER, randflowF, costF, beta, alpha, base_flows=None, seed=None):
    #W_var_history, C_history, V_history = [], [], []
    prng = np.random.default_rng(seed)

    # Initial flow population
    Flow_X = [ randflowF() for _ in range(alpha) ] 
    if base_flows is not None: 
        Flow_X.extend(base_flows)
        alpha+=len(base_flows)

    assert Flow_X, f'Expecting at least one initial flow!'
    ndim = len(Flow_X[0])

    for ITER in range(1, MAXITER): 
        Flow_newX = [] #<--- for new set of flows
        #  fitness for all flows
        Flow_fitness = np.array([costF(x) for x in Flow_X])
        #  best flow out of current flows (in Flow_X)
        best_flow_at = np.argmin(Flow_fitness)
        best_flow = Flow_X[best_flow_at]
        best_flow_cost = Flow_fitness[best_flow_at]

        # calulate 'W' which is required for calulating a 'delta' for each flow in Flow_X
        rand_bar = prng.uniform(0,1,size=ndim)
        randn = prng.normal(0,1)
        iter_ratio = ITER/MAXITER

        W =     ( 1-iter_ratio ) ** ( 2 * randn ) * \
                ( (iter_ratio * rand_bar) * rand_bar)

        for i,flow_i in enumerate(Flow_X):

            rand, x_rand = prng.uniform(0,1), randflowF()
            delta = rand*(x_rand-flow_i) * l2(best_flow-flow_i) * W

            # create beta neighbours
            Neighbour_X = [ flow_i + prng.normal(0,1)*delta \
                        for _ in range(beta) ]

            # cal neighbour fitness
            Neighbour_fitness = np.array( [costF(n) for n in Neighbour_X] )

            best_neighbour_at = np.argmin(Neighbour_fitness)
            best_neighbour = Neighbour_X[best_neighbour_at]
            best_neighbour_cost = Neighbour_fitness[best_neighbour_at]
            #print('\tbest-neighbour:', best_neighbour, ' cost:', best_neighbour_cost)


            s0=[] #<---- slopes
            for j in range(beta):
                num = Flow_fitness[i]-Neighbour_fitness[j]

                #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=
                # **[NOTE:1]
                #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=
    
                s0.append( num/np.abs(flow_i - Neighbour_X[j]))  
                #s0.append( num/l2(flow_i - Neighbour_X[j]) )  # if useing, also change line [121, 132]
                #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=
            #print(f'{len(s0)}, {s0=}')
            flow2best=False
            if best_neighbour_cost < Flow_fitness[i]:
                if not(s0[best_neighbour_at].any()==np.nan or\
                   s0[best_neighbour_at].any()==np.inf or \
                   s0[best_neighbour_at].any()==-np.inf): flow2best=True
                
            
            if flow2best:
                V = prng.normal(0,1)*s0[best_neighbour_at] #<---- flope of best neighbour 
                new_flow_i = flow_i + \
                            V * ( (flow_i-best_neighbour)/l2(flow_i-best_neighbour))
            else:
                #V = np.zeros(self.n_dim) + (1/self.n_dim**0.5) #<---- so that log(V) is zero
                r = i
                while r==i:
                    r = prng.integers(0, len(Flow_X))
                if Flow_fitness[r]<Flow_fitness[i]:
                    randn_bar=prng.normal(0,1,size=ndim)
                    new_flow_i = flow_i + randn_bar*(Flow_X[r]-flow_i) # Note, flow_i == Flow_X[i]
                else:
                    rand_n = prng.uniform(0,1)
                    new_flow_i = flow_i + 2*rand_n*(best_flow - flow_i)

            Flow_newX.append(new_flow_i)
        # end for (all flows)
        Flow_newfitness = np.array([costF(x) for x in Flow_newX])    

        for i in range(alpha):
            if Flow_newfitness[i] < Flow_fitness[i]:
                Flow_X[i] = Flow_newX[i]
                Flow_fitness[i] = Flow_newfitness[i]
        
        #print('\n')
    # end for (all iterations)
    #best_flow_at = np.argmin(Flow_fitness)
    #best_flow = Flow_X[best_flow_at]
    #best_flow_cost = Flow_fitness[best_flow_at]

    return Flow_X, Flow_fitness

