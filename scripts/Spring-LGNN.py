################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score

from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)

# from statistics import mode


# from sympy import LM


# from torch import batch_norm_gather_stats_with_counts


MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import * # cal_graph
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *
import pdb as pdb

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def wrap_main(f):
    def fn(*args, **kwargs):
        config = (args, kwargs)
        print("Configs: ")
        print(f"Args: ")
        for i in args:
            print(i)
        print(f"KwArgs: ")
        for k, v in kwargs.items():
            print(k, ":", v)
        return f(*args, **kwargs, config=config)

    return fn


def Main(N=3, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001,
         withdata=None, datapoints=None, batch_size=1000):

    return wrap_main(main)(N=N, epochs=epochs, seed=seed, rname=rname, saveat=saveat, error_fn=error_fn,
                           dt=dt, ifdrag=ifdrag, stride=stride, trainm=trainm, grid=grid, mpass=mpass, lr=lr,
                           withdata=withdata, datapoints=datapoints, batch_size=batch_size)


def main(N=3, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001, withdata=None, datapoints=None, batch_size=1000, config=None):
    """
    Args:
        mpass: int --> number of message passing (sent to cal_graph())
         
    """

    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, ifdrag, batch_size,
    #        namespace=locals())

    randfilename = datetime.now().strftime(
        "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    PSYS = f"{N}-Spring"
    TAG = f"lgnn"
    out_dir = f"../results"

    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"{withdata}")
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def displacement(a, b):
        return a - b

    def shift(R, dR, V):
        return R+dR, V

    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func

    #loadmodel = OUT(src.models.loadmodel)
    #savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)

    savefile(f"config_{ifdrag}_{trainm}.pkl", config)

    ################################################
    ################## CONFIG ######################
    ################################################
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    try:
        dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data")[0]
    except:
        raise Exception("Generate dataset first. Use *-data.py file.")

    if datapoints is not None:
        dataset_states = dataset_states[:datapoints]

    #dataset_states --> <class 'list'>  --> len : 1000
    model_states = dataset_states[0]  # model_states.position.keys() dict_keys(['position', 'velocity', 'force', 'mass'])

    print(
        f"Total number of data points: {len(dataset_states)}x{model_states.position.shape[0]}")

    N, dim = model_states.position.shape[-2:] #
    species = jnp.zeros(N, dtype=int) #species.shape - (9,)  array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
    masses = jnp.ones(N) #  DeviceArray([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)

    Rs, Vs, Fs = States().fromlist(dataset_states).get_array() # Rs.shape :(1000, 100, 9, 2) :: 1000 initial points , 100 trajectories for each initial point, 9 number of springs, 2: x and y
    Rs = Rs.reshape(-1, N, dim) # (100000, 9, 2)
    Vs = Vs.reshape(-1, N, dim)
    Fs = Fs.reshape(-1, N, dim)

    mask = np.random.choice(len(Rs), len(Rs), replace=False) # (100000,) #shuffling the data
    allRs = Rs[mask] # (100000, 9, 2)  
    allVs = Vs[mask] # (100000, 9, 2)
    allFs = Fs[mask] # (100000, 9, 2)

    Ntr = int(0.75*len(Rs)) # 75000 ---> train set
    Nts = len(Rs) - Ntr # 25000 ----> test set

    Rs = allRs[:Ntr] # train set
    Vs = allVs[:Ntr] # train set
    Fs = allFs[:Ntr] # train set

    Rst = allRs[Ntr:] # test set
    Vst = allVs[Ntr:] # test set
    Fst = allFs[Ntr:] # test set

    ################################################
    ################## SYSTEM ######################
    ################################################

    # pot_energy_orig = PEF
    # kin_energy = partial(lnn._T, mass=masses)

    # def Lactual(x, v, params):
    #     return kin_energy(v) - pot_energy_orig(x)

    # def constraints(x, v, params):
    #     return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

    # def external_force(x, v, params):
    #     F = 0*R
    #     F = jax.ops.index_update(F, (1, 1), -1.0)
    #     return F.reshape(-1, 1)

    # def drag(x, v, params):
    #     return -0.1*v.reshape(-1, 1)

    # acceleration_fn_orig = lnn.accelerationFull(N, dim,
    #                                             lagrangian=Lactual,
    #                                             non_conservative_forces=None,
    #                                             constraints=constraints,
    #                                             external_force=None)

    # def force_fn_orig(R, V, params, mass=None):
    #     if mass is None:
    #         return acceleration_fn_orig(R, V, params)
    #     else:
    #         return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)

    # @jit
    # def forward_sim(R, V):
    #     return predition(R,  V, None, force_fn_orig, shift, dt, masses, stride=stride, runs=10)

    ################################################
    ################### ML Model ###################
    ################################################

    if grid:
        print("It's a grid?")
        a = int(np.sqrt(N))
        senders, receivers = get_connections(a, a)
        eorder = edge_order(len(senders))
    else: # default: executes
        print("It's a random?")
        # senders, receivers = get_fully_connected_senders_and_receivers(N)
        print("Creating Chain")
        _, _, senders, receivers = chain(N) # in psystems/nsprings.py <function chain at 0x2b47248579d8>
        # senders.shape--> (18,), recievers.shape --> (18,)
        eorder = edge_order(len(senders)) # in psystems/nsprings.py # shape-> (18,)

    Ef = 1  # eij dim
    Nf = dim # x and y axes--> 2 dimensions
    Oh = 1 # ? mean

    Eei = 5 # mean?
    Nei = 5 # mean?

    hidden = 5 # mean?
    nhidden = 2 # mean?



    def get_layers(in_, out_):
        """
        Args:
            in_ = 1
            out_ = 5
        
        """
        return [in_] + [hidden]*nhidden + [out_] # [1, 5, 5, 5]

    def mlp(in_, out_, key, **kwargs):
        """
        Args:
            in_ = 1
            out_ = 5
            key = DeviceArray([ 0, 42], dtype=uint32)
            kwargs = {}
        
        """
        return initialize_mlp(get_layers(in_, out_), key, **kwargs)

    fneke_params = initialize_mlp([Oh, Nei], key) # src/models.py # [Oh, Nei] -> [0, 5] # key --> DeviceArray([ 0, 42], dtype=uint32)
    """
        fneke_params:
        [(DeviceArray([[ 2.075412  ],
                    [ 0.23462065],
                    [ 0.558156  ],
                    [-1.1863935 ],
                    [ 0.882776  ]], dtype=float32), DeviceArray([-0.32761326, -0.40663472,  1.2469069 ,  1.1900423 ,
                    1.1002629 ], dtype=float32))]
    
    """
    
    fne_params = initialize_mlp([Oh, Nei], key)
    """
        fne_params:
        [(DeviceArray([[ 2.075412  ],
                    [ 0.23462065],
                    [ 0.558156  ],
                    [-1.1863935 ],
                    [ 0.882776  ]], dtype=float32), DeviceArray([-0.32761326, -0.40663472,  1.2469069 ,  1.1900423 ,
                    1.1002629 ], dtype=float32))]
    
    """


    fb_params = mlp(Ef, Eei, key)
    """
        fb_parmas:
        [(DeviceArray([[ 0.7041197 ],
                [-0.155193  ],
                [ 1.5854169 ],
                [-0.47837773],
                [-1.1402668 ]], dtype=float32), DeviceArray([-0.3681209, -1.0425483,  0.3811884, -1.4299325,  0.5978892],            dtype=float32)), (DeviceArray([[ 0.0065853 , -1.084224  , -0.90702647, -1.0937953 ,
                2.3816905 ],
                [ 0.5397879 , -0.87493896, -1.3264493 , -0.55544454,
                0.07416377],
                [ 0.40697423, -0.11366126, -0.9201015 , -0.0751691 ,
                -0.42749977],
                [ 1.9849769 ,  1.388724  , -0.12409643,  1.1750995 ,
                1.5956461 ],
                [-0.92417973, -0.9839055 , -1.8888152 , -1.0319374 ,
                -1.5857583 ]], dtype=float32), DeviceArray([ 1.7153422 , -0.36852255, -0.06004705, -0.25918266,
                0.24561109], dtype=float32)), (DeviceArray([[-0.34064734,  0.3280096 , -1.3356769 , -1.4240383 ,
                0.06739006],
                [-0.1112586 ,  1.1080816 , -0.8993317 ,  1.9280013 ,
                -0.3039846 ],
                [-0.24217628,  2.6567311 ,  1.3840958 ,  0.7180243 ,
                2.026078  ],
                [-2.2601595 ,  0.72958964,  1.6668926 , -0.6043215 ,
                -1.6963521 ],
                [ 0.23098028, -0.00758455, -1.2909052 ,  1.0563565 ,
                0.3306756 ]], dtype=float32), DeviceArray([ 1.6468883 , -0.7328745 ,  0.6258243 , -0.72865653,
                0.5730015 ], dtype=float32))]
    
    """
    fv_params = mlp(Nei+Eei, Nei, key)
    """
    fv_parmas
    [(DeviceArray([[-0.52624   , -0.8593089 , -1.9649656 , -2.6234684 ,
               0.288977  , -0.49506813, -0.10955065, -1.0215906 ,
               0.414581  ,  0.1888758 ],
             [-0.8503892 , -1.0705473 , -0.57748204,  0.93424314,
               0.08109591, -0.5390684 , -0.34153157,  0.66879934,
              -0.76888514,  0.58857256],
             [ 0.3404925 ,  0.06593669, -1.3318399 , -0.97811824,
               0.23976856,  0.09856878,  2.1316717 , -0.8348885 ,
               0.18358092, -0.37996858],
             [ 0.17217878, -0.64218897,  1.8151029 ,  1.4846883 ,
              -1.7278991 , -1.4115331 ,  0.00714499, -0.65353966,
              -0.55116683, -0.25311267],
             [-1.1598504 , -0.77897507, -1.3732234 , -0.13761204,
              -2.8188167 ,  0.18389334, -0.5832496 ,  0.9870668 ,
              -0.8391157 ,  1.1551105 ]], dtype=float32), DeviceArray([-0.3681209, -1.0425483,  0.3811884, -1.4299325,  0.5978892],            dtype=float32)), (DeviceArray([[ 0.0065853 , -1.084224  , -0.90702647, -1.0937953 ,
               2.3816905 ],
             [ 0.5397879 , -0.87493896, -1.3264493 , -0.55544454,
               0.07416377],
             [ 0.40697423, -0.11366126, -0.9201015 , -0.0751691 ,
              -0.42749977],
             [ 1.9849769 ,  1.388724  , -0.12409643,  1.1750995 ,
               1.5956461 ],
             [-0.92417973, -0.9839055 , -1.8888152 , -1.0319374 ,
              -1.5857583 ]], dtype=float32), DeviceArray([ 1.7153422 , -0.36852255, -0.06004705, -0.25918266,
              0.24561109], dtype=float32)), (DeviceArray([[-0.34064734,  0.3280096 , -1.3356769 , -1.4240383 ,
               0.06739006],
             [-0.1112586 ,  1.1080816 , -0.8993317 ,  1.9280013 ,
              -0.3039846 ],
             [-0.24217628,  2.6567311 ,  1.3840958 ,  0.7180243 ,
               2.026078  ],
             [-2.2601595 ,  0.72958964,  1.6668926 , -0.6043215 ,
              -1.6963521 ],
             [ 0.23098028, -0.00758455, -1.2909052 ,  1.0563565 ,
               0.3306756 ]], dtype=float32), DeviceArray([ 1.6468883 , -0.7328745 ,  0.6258243 , -0.72865653,
              0.5730015 ], dtype=float32))]
    
    
    """
    fe_params = mlp(Nei, Eei, key)

    ff1_params = mlp(Eei, 1, key)
    ff2_params = mlp(Nei, 1, key)
    ff3_params = mlp(dim+Nei, 1, key)
    """
    ff3_parmas:
        [(DeviceArray([[-1.473235  , -0.3015993 , -1.4777459 , -0.89958614,
                0.61375004,  0.77757746, -0.27506098],
                [ 0.56382036,  0.2546131 ,  1.0025856 ,  0.06507307,
                0.6341238 ,  0.440264  , -0.5869762 ],
                [ 1.5369172 , -0.4693531 ,  0.22295444, -1.3559561 ,
                -1.2421908 , -0.18677211, -0.14929186],
                [-0.53434426,  2.089592  , -0.7393486 , -1.9540367 ,
                -0.11162121, -0.05698226,  0.43866852],
                [ 1.7334421 ,  1.1154425 ,  0.30640864, -0.6547044 ,
                1.506656  , -0.47826478,  1.3948326 ]], dtype=float32), DeviceArray([-0.3681209, -1.0425483,  0.3811884, -1.4299325,  0.5978892],            dtype=float32)), (DeviceArray([[ 0.0065853 , -1.084224  , -0.90702647, -1.0937953 ,
                2.3816905 ],
                [ 0.5397879 , -0.87493896, -1.3264493 , -0.55544454,
                0.07416377],
                [ 0.40697423, -0.11366126, -0.9201015 , -0.0751691 ,
                -0.42749977],
                [ 1.9849769 ,  1.388724  , -0.12409643,  1.1750995 ,
                1.5956461 ],
                [-0.92417973, -0.9839055 , -1.8888152 , -1.0319374 ,
                -1.5857583 ]], dtype=float32), DeviceArray([ 1.7153422 , -0.36852255, -0.06004705, -0.25918266,
                0.24561109], dtype=float32)), (DeviceArray([[ 1.3687004 , -1.1934799 ,  1.9728705 ,  0.6608229 ,
                0.02296194]], dtype=float32), DeviceArray([0.21072027], dtype=float32))]
    
    
    """
    ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])
    """
    ke_parmas:
        [(DeviceArray([[ 0.00322629, -0.89955616,  0.62765497,  1.1348032 ,
                -0.6810764 ,  1.6817735 ],
                [-1.4994512 ,  1.455188  , -1.8684514 , -1.4111335 ,
                0.34427443,  0.3897102 ],
                [ 1.0976006 , -1.7042897 , -0.01491493,  0.21004632,
                -1.1919748 ,  0.5026559 ],
                [-0.6358833 ,  0.58948463, -0.19513735, -1.0321059 ,
                -0.12748118,  0.7405304 ],
                [-0.844815  , -0.154417  ,  1.0247767 ,  0.11422939,
                0.28967494,  1.1516122 ],
                [-1.6981575 ,  0.47032514, -0.27696633,  0.27591047,
                0.00329801, -0.45859522],
                [-1.2164251 , -0.1717648 , -0.96491796, -0.06828167,
                0.53883725,  0.8543786 ],
                [ 0.7089056 , -1.004556  ,  0.11919532,  0.65107816,
                -0.30539724, -1.2796072 ],
                [-2.6501772 , -0.6494678 ,  1.2331777 ,  0.37309495,
                0.40770796, -0.46637172],
                [-1.9542568 , -0.9474447 , -2.1213527 , -0.2898174 ,
                -0.97899985, -0.18828417]], dtype=float32), DeviceArray([ 0., -0.,  0., -0.,  0.,  0., -0., -0.,  0.,  0.], dtype=float32)), (DeviceArray([[ 1.6753788 , -0.41598487, -1.4071592 , -1.091225  ,
                -1.632567  , -0.44361305, -0.12116553,  0.5742298 ,
                0.72286385,  0.7550807 ],
                [-0.30755505,  1.9392498 , -0.20337962, -0.50147027,
                -0.6147438 ,  0.99478966,  0.72862065,  0.3497895 ,
                0.7582579 ,  0.8922674 ],
                [-0.15416694, -0.06859886, -0.74683523,  0.08216983,
                -0.31330082, -0.24141693,  0.38588148, -0.06856951,
                0.37046674,  1.4505005 ],
                [-0.54168355, -1.837251  , -1.7911594 ,  0.88038605,
                -1.5697381 ,  0.45255944,  0.4566668 , -1.8899494 ,
                0.758542  , -1.2547628 ],
                [ 0.6603963 ,  0.21346863,  1.5689476 ,  0.5235863 ,
                -0.28961483, -0.6250768 , -0.65715873, -0.00596076,
                -1.9933869 , -0.22522618],
                [ 0.01588469,  2.2302105 ,  0.68902767, -0.5325508 ,
                1.3785113 ,  0.0699767 ,  1.6464918 , -1.3147717 ,
                -0.94136727, -0.13116756],
                [-0.9063527 , -1.401032  ,  0.8697437 ,  1.1258745 ,
                -0.891545  ,  0.05492619, -1.656186  ,  0.5050847 ,
                -1.2692127 , -0.6930687 ],
                [ 0.14924382, -0.5604624 ,  1.480729  ,  0.72572625,
                -0.11553544,  0.5025322 ,  0.30464813, -0.18321568,
                0.32872042,  1.9893782 ],
                [-1.7927711 ,  1.9783881 ,  0.5101261 , -0.26589304,
                1.3015552 , -0.8845139 , -0.9819137 ,  0.04577835,
                1.9722056 , -0.44299528],
                [-0.9286923 , -0.4924002 , -1.9730735 , -1.299754  ,
                1.0746211 ,  0.25494245, -1.2052305 ,  0.9650887 ,
                1.70771   ,  0.13407326]], dtype=float32), DeviceArray([-0.,  0., -0.,  0., -0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)), (DeviceArray([[ 0.78722817, -1.2488708 ,  0.8947005 ,  2.0549572 ,
                0.9555017 , -0.13302046, -0.9986346 ,  1.0912364 ,
                1.2329265 ,  0.564678  ]], dtype=float32), DeviceArray([0.], dtype=float32))]
    
    
    """

    Lparams = dict(fb=fb_params,
                   fv=fv_params,
                   fe=fe_params,
                   ff1=ff1_params,
                   ff2=ff2_params,
                   ff3=ff3_params,
                   fne=fne_params,
                   fneke=fneke_params,
                   ke=ke_params)

    """
    Lparams.keys():
        dict_keys(['fb', 'fv', 'fe', 'ff1', 'ff2', 'ff3', 'fne', 'fneke', 'ke'])

    """

    if trainm: # trainm =1 --> true
        print("kinetic energy: learnable")

        def L_energy_fn(params, graph):
            """
            Args:
                params: dict --> params.keys() --> dict_keys(['fb', 'fv', 'fe', 'ff1', 'ff2', 'ff3', 'fne', 'fneke', 'ke'])

                
                graph:  jraph.GraphsTuple

            Returns: T - V : DeviceArray(-504066.75, dtype=float32) --> total energy - pot energy
            
            """
            g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder, # /LGNN/src/graph.py
                                useT=True, useonlyedge=True)
            return T - V

    else:
        print("kinetic energy: 0.5mv^2")

        kin_energy = partial(lnn._T, mass=masses)

        def L_energy_fn(params, graph):
            """
            Args:
                params: dict --> params.keys() --> dict_keys(['fb', 'fv', 'fe', 'ff1', 'ff2', 'ff3', 'fne', 'fneke', 'ke'])

                
                graph:  jraph.GraphsTuple

            
            
            
            """
            g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
                                useT=True, useonlyedge=True)
            return kin_energy(graph.nodes["velocity"]) - V

    R, V = Rs[0], Vs[0] # inital vel and pos--> R and v shape: (9, 2) 9springs and x and y axes


    state_graph = jraph.GraphsTuple(nodes={  # GraphsTuple treat it like a list
        "position": R,
        "velocity": V,
        "type": species,
    },
        edges={},
        senders=senders, # recall: created by chain(N)
        receivers=receivers,
        n_node=jnp.array([N]),
        n_edge=jnp.array([senders.shape[0]]),
        globals={})

    #pdb.set_trace()
    L_energy_fn(Lparams, state_graph) 


    def energy_fn(species):
        """
        Args: 
            species = DeviceArray([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
            
        """
        state_graph = jraph.GraphsTuple(nodes={
            "position": R, # device array --> shape (9,2)
            "velocity": V, # device array --> shape (9,2)
            "type": species
        },
            edges={},
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders.shape[0]]),
            globals={})

        def apply(R, V, params): # <function main.<locals>.energy_fn.<locals>.apply 
            """
            Args:
                R: device array --> shape (9,2)
                V: device array --> shape (9,2)
                params:  params: dict --> params.keys() --> dict_keys(['fb', 'fv', 'fe', 'ff1', 'ff2', 'ff3', 'fne', 'fneke', 'ke'])

            Returns: energy of state and updates the graph(not returns)
            """
            state_graph.nodes.update(position=R)
            state_graph.nodes.update(velocity=V)
            return L_energy_fn(params, state_graph)
        return apply

    apply_fn = energy_fn(species) # updates graph and returns <function main.<locals>.energy_fn.<locals>.
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0)) #<function main.<locals>.energy_fn.<locals>.apply at 0x2afcf4c52e18>


    def Lmodel(x, v, params): return apply_fn(x, v, params["L"])

    pdb.set_trace()
    params = {"L": Lparams}

    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, v, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: nn")

        def drag(x, v, params):
            return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

    params["drag"] = initialize_mlp([1, 5, 5, 1], key)

    acceleration_fn_model = jit(accelerationFull(N, dim,
                                                 lagrangian=Lmodel,
                                                 constraints=None,
                                                 non_conservative_forces=drag))
    v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

    ################################################
    ################## ML Training #################
    ################################################

    LOSS = getattr(src.models, error_fn)

    @jit
    def loss_fn(params, Rs, Vs, Fs):
        pred = v_acceleration_fn_model(Rs, Vs, params)
        return LOSS(pred, Fs)

    @jit
    def gloss(*args):
        return value_and_grad(loss_fn)(*args)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        grads_ = jax.tree_map(
            partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

    @jit
    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @ jit
    def step(i, ps, *args):
        return update(i, *ps, *args)

    def batching(*args, size=None):
        L = len(args[0])
        if size != None:
            nbatches1 = int((L - 0.5) // size) + 1
            nbatches2 = max(1, nbatches1 - 1)
            size1 = int(L/nbatches1)
            size2 = int(L/nbatches2)
            if size1*nbatches1 > size2*nbatches2:
                size = size1
                nbatches = nbatches1
            else:
                size = size2
                nbatches = nbatches2
        else:
            nbatches = 1
            size = L

        newargs = []
        for arg in args:
            newargs += [jnp.array([arg[i*size:(i+1)*size]
                                   for i in range(nbatches)])]
        return newargs

    bRs, bVs, bFs = batching(Rs, Vs, Fs,
                             size=min(len(Rs), batch_size))

    print(f"training ...")

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []
    last_loss = 1000

    larray += [loss_fn(params, Rs, Vs, Fs)]
    ltarray += [loss_fn(params, Rst, Vst, Fst)]

    def print_loss():
        print(
            f"Epoch: {epoch}/{epochs} Loss (mean of {error_fn}):  train={larray[-1]}, test={ltarray[-1]}")

    print_loss()

    for epoch in range(epochs):
        for data in zip(bRs, bVs, bFs):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)

        # optimizer_step += 1
        # opt_state, params, l_ = step(
        #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)

        if epoch % saveat == 0:
            larray += [loss_fn(params, Rs, Vs, Fs)]
            ltarray += [loss_fn(params, Rst, Vst, Fst)]
            print_loss()

        if epoch % saveat == 0:
            metadata = {
                "savedat": epoch,
                "mpass": mpass,
                "grid": grid,
                "ifdrag": ifdrag,
                "trainm": trainm,
            }
            savefile(f"lgnn_trained_model_{ifdrag}_{trainm}.dil",
                     params, metadata=metadata)
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                     (larray, ltarray), metadata=metadata)
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"lgnn_trained_model_{ifdrag}_{trainm}_low.dil",
                         params, metadata=metadata)

    fig, axs = panel(1, 1)
    plt.semilogy(larray[1:], label="Training")
    plt.semilogy(ltarray[1:], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png")) # only saving in LGNN

    metadata = {
        "savedat": epoch,
        "mpass": mpass,
        "grid": grid,
        "ifdrag": ifdrag,
        "trainm": trainm,
    }
    params = get_params(opt_state)
    savefile(f"lgnn_trained_model_{ifdrag}_{trainm}.dil",
             params, metadata=metadata)
    savefile(f"loss_array_{ifdrag}_{trainm}.dil",
             (larray, ltarray), metadata=metadata)


fire.Fire(Main)
