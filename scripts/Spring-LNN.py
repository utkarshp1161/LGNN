################################################
################## IMPORT ######################
################################################

import pdb

import json
import sys
from datetime import datetime
from functools import partial, wraps
from logging import exception

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers # load ADAM from here
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score
#from torch import mode --> not being used

from psystems.nsprings import get_init

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def wrap_main(f):
    """a higher-order function that takes another function f as an argument and returns a new function fn. 
    The fn function takes any number of arguments and keyword arguments and calls the original function f 
    with the same arguments and keyword arguments, as well as an additional argument config which 
    is a tuple containing the arguments and keyword arguments passed to fn. 
    fn first prints the arguments and keyword arguments and then calls f with all of them.
    
    
    """
    def fn(*args, **kwargs):
        config = (args, kwargs) #((), {'N': 9, 'epochs': 20, 'seed': 42, 'rname': True, 'saveat': 10, 'error_fn': 'L2error', 'dt': 0.001, 'ifdrag': 0, 'stride': 100, 'trainm': 1, 'grid': False, 'mpass': 1, 'lr': 0.001, 'withdata': None, 'datapoints': None, 'batch_size': 1000})
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
    """Main is a function that takes several optional arguments with default values and 
    returns the result of calling wrap_main(main) with those arguments. 
    The arguments and their default values are:

    args:
        N: 3
        epochs: 10000
        seed: 42
        rname: True
        saveat: 10
        error_fn: "L2error"
        dt: 1.0e-3
        ifdrag: 0
        stride: 100
        trainm: 1
        grid: False
        mpass: 1
        lr: 0.001
        withdata: None
        datapoints: None
        batch_size: 1000
    
    
    """

    return wrap_main(main)(N=N, epochs=epochs, seed=seed, rname=rname, saveat=saveat, error_fn=error_fn,
                           dt=dt, ifdrag=ifdrag, stride=stride, trainm=trainm, grid=grid, mpass=mpass, lr=lr,
                           withdata=withdata, datapoints=datapoints, batch_size=batch_size)


def main(N=3, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001, withdata=None, datapoints=None, batch_size=1000, config=None):
    """main is the original function that is being wrapped by wrap_main.
    It takes several optional arguments with default values and is 
    responsible for implementing the main logic of the script. 
    The arguments and their default values are the same as in Main. 
    It is called with all the arguments passed to Main, 
    as well as the additional config argument.

    """

    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, batch_size,
    #        namespace=locals())

    randfilename = datetime.now().strftime(
        "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}" #02-07-2023_15-46-44_None

    PSYS = f"{N}-Spring" #'9-Spring'
    TAG = f"lnn" # 'lnn'
    out_dir = f"../results" #'../results'

    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data")) else "0"
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func

    #loadmodel = OUT(src.models.loadmodel) # sends output of _filename(file, tag=tag), *args, **kwargs
    #savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile) # <function loadfile at 0x2af04be36840>
    savefile = OUT(src.io.savefile) # <function savefile at 0x2af04be368c8>
    #save_ovito = OUT(src.io.save_ovito)

    savefile(f"config_{ifdrag}_{trainm}.pkl", config) # ((), {'N': 9, 'epochs': 20, 'seed': 42, 'rname': True, 'saveat': 10, 'error_fn': 'L2error', 'dt': 0.001, 'ifdrag': 0, 'stride': 100, 'trainm': 1, 'grid': False, 'mpass': 1, 'lr': 0.001, 'withdata': None, 'datapoints': None, 'batch_size': 1000})
    # this pickle file contains info about system 


    ################################################
    ################## CONFIG ######################
    ################################################
    np.random.seed(seed) # 42
    key = random.PRNGKey(seed) #DeviceArray([ 0, 42], dtype=uint32), A PRNG key, consumable by random functions as well as split and fold_in.

    try:
        dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data")[0] # loads the /results/9-Spring-data/0/model_states_0.pkl --> 
    except:
        raise Exception("Generate dataset first. Use *-data.py file.")

    if datapoints is not None: # not executed--> datapoints = None
        dataset_states = dataset_states[:datapoints]

    #dataset_states --> <class 'list'>  --> len : 1000
    model_states = dataset_states[0] #--> <class 'jax_md.simulate.NVEState'>

    print(
        f"Total number of data points: {len(dataset_states)}x{model_states.position.shape[0]}")

    N, dim = model_states.position.shape[-2:] # 9, 2 # model_states.position.keys() dict_keys(['position', 'velocity', 'force', 'mass'])
    masses = model_states.mass[0].flatten() # array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
    species = jnp.zeros(N, dtype=int) # DeviceArray([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)

    Rs, Vs, Fs = States().fromlist(dataset_states).get_array()  #  Rs.shape :(1000, 100, 9, 2) :: 1000 initial points , 100 trajectories for each initial point, 9 number of springs, 2: x and y
    Rs = Rs.reshape(-1, N, dim) # (100000, 9, 2)
    Vs = Vs.reshape(-1, N, dim) # (100000, 9, 2)
    Fs = Fs.reshape(-1, N, dim) # (100000, 9, 2)
    #pdb.set_trace()

    mask = np.random.choice(len(Rs), len(Rs), replace=False) # (100000,) # shuffling the data
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

    #pdb.set_trace()

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

    def MLP(in_dim, out_dim, key, hidden=256, nhidden=2):
        """Initialzes a mlp from src.models
        Args:
            in_dim: int
            out_dim: int
            key: int
            hidden: int
            nhidden: int
        

        Returns: Initialize the weights of all layers of a linear layer network
        """
        return initialize_mlp([in_dim]+[hidden]*nhidden+[out_dim], key)

    lnn_params_pe = MLP(N*dim, 1, key) # <class 'list'>, defining the neural network weights --> parameters
    lnn_params_ke = jnp.array(np.random.randn(N)) # (9,) 
    #pdb.set_trace()

    def Lmodel(x, v, params):
        if trainm:
            print("KE: 0.5mv2")
            KE = (jnp.abs(params["lnn_ke"]) * jnp.square(v).sum(axis=1)).sum()
        else:
            print("KE: learned")
            KE = (masses * jnp.square(v).sum(axis=1)).sum()
        return (KE -
                forward_pass(params["lnn_pe"], x.flatten(), activation_fn=SquarePlus)[0])

    params = {"lnn_pe": lnn_params_pe, "lnn_ke": lnn_params_ke} # defined params

    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, v, params):
            """Returns 0
            """
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, v, params):
            """
            """
            return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

    params["drag"] = initialize_mlp([1, 5, 5, 1], key) # dict_keys(['lnn_pe', 'lnn_ke', 'drag'])

    acceleration_fn_model = src.lnn.accelerationFull(N, dim,       
                                                     lagrangian=Lmodel,
                                                     constraints=None,
                                                     non_conservative_forces=drag)
    #acceleration_fn_model <function accelerationFull.<locals>.fn at 0x2ac84ff7c158>
    v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None)) # <function accelerationFull.<locals>.fn at 0x2b904ff06c80>

    ################################################
    ################## ML Training #################
    ################################################

    LOSS = getattr(src.models, error_fn) #<function L2error at 0x2b904753e268> # get attribute error_fn of the object src.models # why not load directly?
    #pdb.set_trace()

    @jit
    def loss_fn(params, Rs, Vs, Fs):
        pred = v_acceleration_fn_model(Rs, Vs, params)
        return LOSS(pred, Fs) # loss(y_hat, y) --> 

    @jit
    def gloss(*args):
        return value_and_grad(loss_fn)(*args) # jax.value_and_grad --> Create a function that evaluates both fun and the gradient of fun.

    @jit
    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters 
        Args:
            i:
            opt_state:
            params:
            loss__:
            *data: R, v, F :-> pos, vel, forces?

        
        """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @ jit
    def step(i, ps, *args):
        return update(i, *ps, *args)

    opt_init, opt_update_, get_params = optimizers.adam(lr) # optimizer load: (init_fun, update_fun, get_params) triple. 
    """https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adam"""

    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        # grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

    def batching(Rs, Vs, Fs, size=None):
        """Returns : batches of entire data 

        Args:
            size: int --> batch size
        
        """
        args = [Rs, Vs, Fs]
        L = len(args[0]) # args = [Rs, Vs, Fs]
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
        for arg in args: # arg = [Rs, Vs, Fs]
            newargs += [jnp.array([arg[i*size:(i+1)*size]
                                   for i in range(nbatches)])]
        return newargs

    bRs, bVs, bFs = batching(Rs, Vs, Fs, size=batch_size) # b means batched, data batched

    print(f"training ...")

    opt_state = opt_init(params) # initialize adam optimizer with params
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []

    part = f"{ifdrag}_{datapoints}" # '0_None'

    last_loss = 1000

    larray += [loss_fn(params, Rs, Vs, Fs)] # [DeviceArray(65490.94438603, dtype=float64)]
    ltarray += [loss_fn(params, Rst, Vst, Fst)] # t means test, [DeviceArray(65221.98776716, dtype=float64)]

    def print_loss():
        """Prints at every: epoch % saveat == 0 --> 10 iterations
        Args: 
            epoch: int
            epochs: int default: 1000
            error_fn: str default: "L2error"
            larray: array
            ltarray: array


        
        """
        print(
            f"Epoch: {epoch}/{epochs} Loss (mean of {error_fn}):  train={larray[-1]}, test={ltarray[-1]}")

    print_loss() # Epoch: 0/10000 Loss (mean of L2error):  train=65490.94438602682, test=65221.98776715841

    for epoch in range(epochs):
        for data in zip(bRs, bVs, bFs): # (1000, 9, 2), (1000, 9, 2), (1000, 9, 2)--> batchsize =1000, 9 nodes, 2: x and y axes values of pos, vel and forces
            #pdb.set_trace()
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data) # opt_state: <class 'jax.experimental.optimizers.OptimizerState'>, step: DeviceArray(66073.8246356, dtype=float64)

        # optimizer_step += 1
        # opt_state, params, l_ = step(
        #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)

        if epoch % saveat == 0: # saveat == 10
            larray += [loss_fn(params, Rs, Vs, Fs)]
            ltarray += [loss_fn(params, Rst, Vst, Fst)] # loss test array
            print_loss()

        if epoch % saveat == 0:
            metadata = {
                "savedat": epoch,
                "mpass": mpass,
                "grid": grid,
                "ifdrag": ifdrag,
                "trainm": trainm,
            }
            savefile(f"{TAG}_trained_model_{ifdrag}_{trainm}.dil",
                     params, metadata=metadata)
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                     (larray, ltarray), metadata=metadata)
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"{TAG}_trained_model_{ifdrag}_{trainm}_low.dil", # best model?
                         params, metadata=metadata)

    fig, axs = panel(1, 1)
    plt.plot(larray, label="Training")
    plt.plot(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss_{part}.png"))

    params = get_params(opt_state)
    savefile(f"lnn_trained_model_{part}.dil",
             params, metadata={"savedat": epoch})  # best model?
    savefile(f"loss_array_{part}.dil",
             (larray, ltarray), metadata={"savedat": epoch})

    


fire.Fire(Main)
