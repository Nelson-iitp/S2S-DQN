



import os
import numpy as np
from known import Remap
from tqdm import tqdm
import torch as tt
import torch.nn as nn
from .pie import RandomPie
from .env import Eval
__all__ = ['Train']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" DQN Training """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 

def Train(
    # value params [T.V]
        pie, 
        pie_opt,
        pie_lrs,
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
        explore_pies,
        explore_per_pie,
    # memory params [M]
        memory,
        memory_capacity, 
        memory_seed, 
        min_memory,
        min_memory_pies,
    # selector params
        selector_seed,
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
    from math import inf
    
    has_target = (double or tuf>0)
    val_loss = nn.MSELoss() #<-- its important that DQN uses MSELoss only (consider using huber loss)

    # checkpointing
    if save_at: os.makedirs(save_at, exist_ok=True)
    do_checkpoint = ((checkpoint_freq>0) and save_at)

    # validation
    do_validate = ((len(validations_envs)>0) and validation_freq and validation_episodes)
    mean_validation_return, mean_validation_steps = None, None
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
    rand_pie = RandomPie(Alow=0, Ahigh=env.A, seed=explore_seed) # <--- random exploration policy
    srng = np.random.default_rng(selector_seed)
    pie.eval()

    

    # fill up memory with min_memory episondes
    len_memory = len(memory)
    if len_memory < min_memory:
        min_explore = min_memory-len_memory
        if min_memory_pies is None:
            memory.extend(Eval.explore_policy(n=min_explore, env=env, pie=rand_pie))
        else:
            per_pie_explore = max(1, min_explore//len(min_memory_pies))
            for min_memory_pie in min_memory_pies:
                memory.extend(Eval.explore_policy(n=per_pie_explore, env=env, pie=min_memory_pie))


        
        print(f'[*] Explored Min-Memory [{min_explore}] Steps, Memory Size is [{len(memory)}]')

    #------------------------------------pre-training results
    if do_validate:
        mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps, validate_acts = \
            Eval.validation(validations_envs, pie, validation_episodes, validation_max_steps, validation_verbose)
            
        validation_hist.append((mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps))
        print(f' [Pre-Validation]')
        print(f' => (MEAN) :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
        print(f' => (SUM)  :: Return:{sum_validation_return}, Steps:{sum_validation_steps}')
    if do_checkpoint:
        check_point_as =  os.path.join(save_at, f'pre.pie')  
        pie.save(check_point_as)
        print(f'Checkpoint @ {check_point_as}\n')
    #------------------------------------pre-training results
    if  (explore_pies is not None) and explore_per_pie: per_pie_explore = max(1, explore_size//len(explore_pies))
    for epoch in tqdm(range(epochs)):
        epoch_ratio = epoch/epochs
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

        # eploration phase
        epsilon = epsilonF.forward(epoch_ratio)
        
        if explore_pies is None:
            memory.extend(Eval.explore_greedy(n=explore_size, env=env, pie=pie, rie=rand_pie, epsilon=epsilon, erng=erng))
        else:
            if explore_per_pie:
                for explore_pie in explore_pies:
                    memory.extend(Eval.explore_greedy(n=per_pie_explore, env=env, pie=pie, rie=explore_pie, epsilon=epsilon, erng=erng))
            else:
                memory.extend(Eval.explore_greedy(n=explore_size, env=env, pie=pie, rie=explore_pies[srng.integers(0, len(explore_pies))], epsilon=epsilon, erng=erng))



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
                
                if tt.is_tensor(S): S=S.to(dtype=tt.long, device=pie.device)
                else: S=tuple([s.to(dtype=tt.long, device=pie.device) for s in S])
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
                pie_opt.zero_grad()
                loss =  val_loss(predQ, updatedQ) 
                val_losses.append(loss.item())
                loss.backward()
                # Clip gradient norm
                if gradient_clipping>0.0: nn.utils.clip_grad_norm_(pie.theta.parameters(), gradient_clipping)
                pie_opt.step()
        # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
        
        train_hist.append((epsilon, pie_lrs.get_last_lr()[-1], np.mean(val_losses)))
        count_hist.append((learn_count, update_count, len_memory))
        pie_lrs.step()
        learn_count+=1
        if (has_target):
            if learn_count % tuf == 0:
                pie.update_target(polyak=polyak)
                update_count+=1

        pie.eval()

        if do_validate:
            if ((epoch+1)%validation_freq==0):
                mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps, validate_acts = \
                    Eval.validation(validations_envs, pie, validation_episodes, validation_max_steps, validation_verbose)
                validation_hist.append((mean_validation_return, mean_validation_steps, sum_validation_return, sum_validation_steps))
                print(f' [Validation]')
                print(f' => (MEAN) :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
                print(f' => (SUM)  :: Return:{sum_validation_return}, Steps:{sum_validation_steps}')
        if do_checkpoint:
            if ((epoch+1)%checkpoint_freq==0):
                check_point_as =  os.path.join(save_at, f'{epoch+1}.pie')  
                pie.save(check_point_as)
                print(f'Checkpoint @ {check_point_as}\n')
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
        print(f'Saved @ {save_as}\n')

    validation_hist, train_hist, count_hist = np.array(validation_hist), np.array(train_hist), np.array(count_hist)
    res = dict( train=train_hist, val=validation_hist, count=count_hist )
    if plot_results: _ = Eval.plot_training_result( validation_hist, train_hist, count_hist )
    if save_at:
        save_as = os.path.join(save_at, f'results.npz')
        np.savez( save_as, **res )

    return res
# -= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= -=-= 
