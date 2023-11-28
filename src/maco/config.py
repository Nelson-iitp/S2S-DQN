
import numpy as np
import os, warnings
from .core import ComputeNetwork, Infra

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Global Infra parameters
class GLOBAL_PARAMS:

    SCALE_B, SCALE_C = 1, 2
    # bandwidth in Mbps
    BEC = 8.0*SCALE_B     # bandwidth Edge <--> Cloud
    BEE = 400.0*SCALE_B   # bandwidth Edge <--> Edge
    BCC = 400.0*SCALE_B   # bandwidth Cloud <--> Cloud   
    BUE = 200.0*SCALE_B   # bandwidth UE <--> Edge

    # compute capacity in GHz
    CE = 10.0*SCALE_C # cpu edge
    CC = 8.0*SCALE_C # cpu cloud

    # Global task parameters 
    di_low, di_high, di_divs = 50.0, 100.0, 5     # Mega bytes
    do_low, do_high, do_divs = 50.0, 100.0, 5     # Mega bytes
    cc_low, cc_high, cc_divs = 100.0, 600.0, 10   # Giga cycles

    large_computation_limit = cc_low+(cc_high-cc_low)*0.8
    large_data_limit = di_low+(di_high-di_low)*0.8 + do_low+(do_high-do_low)*0.8 

    di_range = np.linspace(di_low, di_high, (int(di_high-di_low)//di_divs + 1), endpoint=True)
    do_range = np.linspace(do_low, do_high, (int(do_high-do_low)//do_divs + 1), endpoint=True)
    cc_range = np.linspace(cc_low, cc_high, (int(cc_high-cc_low)//cc_divs + 1), endpoint=True)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class Gene:

    @staticmethod
    def generate_state(ranges, network, steps, seeds):
        if seeds is None: seeds=(None, None, None, None)
        di, do, cc = __class__.generate_tasks(ranges, steps, *seeds[0:-1])
        ll = __class__.generate_paths(network, steps, seeds[-1])
        return np.vstack((di, do, cc, ll))

    @staticmethod
    def generate_tasks(ranges, steps, di_seed, do_seed, cc_seed):
        # generate a sequence of tasks and a sequence of locations
        di_rng=np.random.default_rng(di_seed)
        do_rng=np.random.default_rng(do_seed)
        cc_rng=np.random.default_rng(cc_seed)

        di_range, do_range, cc_range = ranges
        di = di_rng.choice(di_range, size=steps, replace=True)
        do = do_rng.choice(do_range, size=steps, replace=True)
        cc = cc_rng.choice(cc_range, size=steps, replace=True)
        return di, do, cc 

    @staticmethod
    def generate_paths(network, steps, seed):
        # generate a sequence of tasks and a sequence of locations
        ll_rng=np.random.default_rng(seed)
        # for location use moves
        stL, stH = int(0.1*steps), int(0.5*steps)
        lt = ll_rng.integers(0, network.mec.E)
        locs = []
        while len(locs)<steps:
            st = ll_rng.integers(stL, stH) # no fo repetitions
            for _ in range(st): locs.append(lt)
            neighbours = network.ngF(lt)
            if len(neighbours)>0: lt = ll_rng.choice(neighbours, size=1)[0]
        ll = np.array(locs[0:steps])
        return ll 
    
    @staticmethod
    def get_apps(params, n_apps, steps, seed):
        rng = np.random.default_rng(seed)
        min_di_select, min_do_select, min_cc_select = 0.50, 0.50, 0.50
        max_di_select, max_do_select, max_cc_select = 1.00, 1.00, 1.00
        min_seed, max_seed = 99, 999999999999999
        di_range, do_range, cc_range = params.di_range, params.do_range, params.cc_range
        di_len, do_len, cc_len = len(di_range), len(do_range), len(cc_range)
        return {
        f'app_{appid}': __class__.generate_tasks((
                                rng.choice(di_range, size=rng.integers(int(min_di_select*di_len), int(max_di_select*di_len))),
                                rng.choice(do_range, size=rng.integers(int(min_do_select*do_len), int(max_do_select*do_len))),
                                rng.choice(cc_range, size=rng.integers(int(min_cc_select*cc_len), int(max_cc_select*cc_len))),
                                ), steps, *[rng.integers(min_seed, max_seed) for _ in range(3)]) \
            for appid in range(n_apps)
        }

    @staticmethod
    def get_paths(n_paths, steps, network, seed):
        rng = np.random.default_rng(seed)
        min_seed, max_seed = 99, 999999999999999
        return {f'path_{pathid}': __class__.generate_paths(network, steps, rng.integers(min_seed, max_seed)) for pathid in range(n_paths)}

    @staticmethod
    def make_ds(path, network, n_apps, n_paths, n_steps, app_seed, path_seed):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        print(f'Create dataset @ {path}')
        #os.makedirs(path, exist_ok=True)
        appd = __class__.get_apps(GLOBAL_PARAMS, n_apps=n_apps, steps=n_steps, seed=app_seed)
        pathd = __class__.get_paths(n_paths=n_paths, steps=n_steps, network=network, seed=path_seed)    

        states = []
        for pathi, pathg in pathd.items():
            for appi, appg in appd.items():
                states.append(np.vstack((*appg, pathg)))

        #save_as = os.path.join(path,f'{name}.npy')
        print(f"{path}: {len(states)}")
        np.save(path, states)
        warnings.filterwarnings("default", category=RuntimeWarning)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class db:
    
    def __str__(self) -> str: return f'{self.__dict__}'

    @staticmethod
    def get_network(mec, default_decimals=3):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        net = ComputeNetwork( n_units=mec.A, default_decimals=default_decimals)
        net.set_resources(mec.VR, mec.DR)
        net.mec = mec
        #net.E, net.C, net.A = mec.E, mec.C, mec.A
        warnings.filterwarnings("default", category=RuntimeWarning)
        return net
            
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # pre-defined infra objects
    # #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    @staticmethod
    def X3e1c(): return __class__.get_network(Infra(3, 1, GLOBAL_PARAMS).connect_mesh())

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-




