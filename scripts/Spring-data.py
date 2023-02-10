################################################
################## IMPORT ######################
################################################

"""

This code is an implementation of a Molecular Dynamics simulation using the Jax library. 
It has several helper functions to save and load data, and functions for calculating the energies and forces of the system. 
It implements a 2D system of point masses connected by springs, and the motion of the masses is calculated using the equations of motion. 
The code allows for changing the number of masses, their initial positions and velocities, and other parameters like the time step size, 
the number of iterations, and the learning rate for an optimizer. The results of the simulation can be saved in the form of data files and 
visualized using the Ovito visualization tool.

"""

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
                               get_fully_edge_order, get_init, get_init_spring)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import predition
from src.models import MSE, initialize_mlp
from src.nve import NVEStates, nve
from src.utils import *

from src.spring.utils_data import  displacement, shift, external_force

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
import pdb
# jax.config.update('jax_platform_name', 'gpu')


def main(N1=3, N2=None, dim=2, grid=False, saveat=100, runs=100, nconfig=1000, ifdrag=0, rname=False):

    if N2 is None:
        N2 = N1

    N = N1*N2 #9

    tag = f"{N}-Spring-data" # 9-spring-data
    seed = 42
    out_dir = f"../results" 
    rstring = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") if rname else "0" # false
    filename_prefix = f"{out_dir}/{tag}/{rstring}/"

    def _filename(name):
        """
        Args:
            name: string
            filename_prefix: string

        Return: string
        """
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename


    def OUT(f):
        """
        
        Args:
            f: string


        Return: 
        """
        @wraps(f)
        def func(file, *args, **kwargs):
            return f(_filename(file), *args, **kwargs) # calls f on output of _filename function
        return func



    #loadmodel = OUT(src.models.loadmodel) # loadmodel
    #savemodel = OUT(src.models.savemodel) # save model

    #loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)

    ################################################
    ################## CONFIG ######################
    ################################################

    np.random.seed(seed)
    key = random.PRNGKey(seed)

    init_confs = [chain(N)[:2]
                  for i in range(nconfig)] # nconfig = 10000

    _, _, senders, receivers = chain(N)

    # if grid:
    #     senders, receivers = get_connections(N1, N2)
    # else:
    #     # senders, receivers = get_fully_connected_senders_and_receivers(N)
    #     print("Creating Chain")

    R, V = init_confs[0] # len(init_config) = nconfig, 

    print("Saving init configs...")
    savefile(f"initial-configs_{ifdrag}.pkl",
             init_confs, metadata={"N1": N1, "N2": N2})

    species = jnp.zeros(N, dtype=int) # species initialized with zeros -> N
    masses = jnp.ones(N) # masses intitalized with ones len -> N

    dt = 1.0e-3
    stride = 100
    lr = 0.001

    kin_energy = partial(lnn._T, mass=masses)

    def pot_energy_orig(x):
        """
        Args:
            x:
            senders: int
            recievers: int

        
        Return: 
        """
        dr = jnp.square(x[senders, :] - x[receivers, :]).sum(axis=1) # k(delta_x **2)
        return vmap(partial(lnn.SPRING, stiffness=1.0, length=1.0))(dr).sum()

    def Lactual(x, v, params):
        """ Calculate Langrangian: K.E - P.E
        """
        return kin_energy(v) - pot_energy_orig(x)


    def force_fn_orig(R, V, params, mass=None):
        """ Returns 

        Args:


        Return:
        """
        if mass is None:
            return acceleration_fn_orig(R, V, params)
        else:
            return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)


    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, v, params):
            """ drag = 0 
            """
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, v, params):
            """ drag = -gamma*v ; gamma = 0.1
            """
            return -0.1*v.reshape(-1, 1)

    acceleration_fn_orig = lnn.accelerationFull(N, dim,
                                                lagrangian=Lactual,
                                                non_conservative_forces=drag,
                                                constraints=None,
                                                external_force=None)


    @jit
    def forward_sim(R, V):
        """Returns state 
        """
        return predition(R,  V, None, force_fn_orig, shift, dt, masses, stride=stride, runs=runs)

    @jit
    def v_forward_sim(init_conf):
        return vmap(lambda x: forward_sim(x[0], x[1]))(init_conf)

    ################################################
    ############### DATA GENERATION ################
    ################################################

    print("Data generation ...")
    ind = 0
    dataset_states = []
    for R, V in init_confs:
        ind += 1
        print(f"{ind}/{len(init_confs)}", end='\r')
        model_states = forward_sim(R, V) #   
        dataset_states += [model_states] # appending the states
        if ind % saveat == 0:
            print(f"{ind} / {len(init_confs)}")
            print("Saving datafile...")
            savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    print("Saving datafile...")
    savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    def cal_energy(states):
        KE = vmap(kin_energy)(states.velocity)
        PE = vmap(pot_energy_orig)(states.position)
        L = vmap(Lactual, in_axes=(0, 0, None))(
            states.position, states.velocity, None)
        return jnp.array([PE, KE, L, KE+PE]).T

    print("plotting energy...")
    ind = 0
    for states in dataset_states:
        ind += 1
        Es = cal_energy(states)

        fig, axs = panel(1, 1, figsize=(20, 5))
        plt.plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("Energy")
        plt.xlabel("Time step")

        title = f"{N}-Spring random state {ind}"
        plt.title(title)
        plt.savefig(
            _filename(title.replace(" ", "_")+".png"), dpi=300)
        save_ovito(f"dataset_{ind}.ovito", [
            state for state in NVEStates(states)], lattice="")

        if ind > 10: # only 9 pendulum system
            break

    #pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
