import jax
from jax import  vmap
import jax.numpy as jnp
from .. import lnn
from functools import partial, wraps
import os




#global filename_prefix = 




def displacement(a, b):
    """

    Args:
        a: float
        b: float
    
    Return: float
    """
    return a - b

def shift(R, dR, V):
    """

    Args:
        R: float
        dR: float
        V: float
    
    Return: final position, velocity
    """
    return R+dR, V








def external_force(R):
    """Calculate external force

    Args:


    Return:
    """
    F = 0*R
    F = jax.ops.index_update(F, (1, 1), -1.0)
    return F.reshape(-1, 1)

