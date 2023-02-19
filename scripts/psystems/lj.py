"""
For lj system: each particle is connected to each particle--> 

"""



def chain(N, L=2, dim=1):
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
