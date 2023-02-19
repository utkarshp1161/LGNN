import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def get_count(s, i):
    c = 0
    for item in s:
        if item == i:
            c += 1
    return c



def get_init_ab(a, b, L=1, dim=2):
    R = jnp.array(np.random.rand(a, dim))*L*2
    V = jnp.array(np.random.rand(*R.shape)) / 10
    V = V - V.mean(axis=0)
    
    def check(i, j, senders, receivers):
        bool = True
        for it1, it2 in zip(senders, receivers):
            if (it1 == i) and (it2 == j):
                bool = False
                break
            if (it2 == i) and (it1 == j):
                bool = False
                break
        return 
        
    def get_count(s, i):
        c = 0
        for item in s:
            if item == i:
                c += 1
        return c

    senders = []
    receivers = []
    for i in range(a):
        c = get_count(senders, i)
        if c >= b:
            pass
        else:
            neigh = b-c
            s = ((R - R[i])**2).sum(axis=1)
            ind = np.argsort(s)
            new = []
            for j in ind:
                if check(i, j, senders, receivers) and (neigh > 0) and j != i:
                    new += [j]
                    neigh -= 1

            senders += new + [i]*len(new)
            receivers += [i]*len(new) + new
            print(i, senders, receivers)

    return R, V, jnp.array(senders, dtype=int), jnp.array(receivers, dtype=int)

def chain(N, L=2, dim=2):
    """Returns R, V, senders, recievers

    Args: 
        N: int
        L: int
        dim: int

    Return: jnp_array, jnp_array, jnp_array, jnp_array
    
    """

    R = jnp.array(np.random.rand(N, dim))*L # shape (N, dim) ==> (9, 2)
    V = jnp.array(np.random.rand(*R.shape)) / 10  # shape (N, dim) ==. (9, 2)
    V = V - V.mean(axis=0) # ==> (9, 2)

    senders = [N-1] + list(range(0, N-1)) # [N-1, 0, 1, 2, ..N-2] #[8, 0, 1, 2, 3, 4, 5, 6, 7]
    receivers = list(range(N)) # [0, 1, 2, 3, 4, ...., N-1] # [0, 1, 2, 3, 4, 5, 6, 7, 8]

    return R, V, jnp.array(senders+receivers, dtype=int), jnp.array(receivers+senders, dtype=int)
    # jnp.array(senders+receivers, dtype=int) --> DeviceArray([8, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
    # jnp.array(receivers+senders, dtype=int) --> DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)

if __name__ == "__main__":
    import pdb as pdb
    pdb.set_trace()