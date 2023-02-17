
"""A library of Graph Neural Network models."""
"""https://github.com/deepmind/jraph"""

import functools
import sunau
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree # used at line 215 in this file
import numpy as np
from frozendict import frozendict
from jax import vmap
from jraph._src import graph as gn_graph
from jraph._src import utils

from .models import SquarePlus, forward_pass
import pdb as pdb
jax.tree_util.register_pytree_node(
    frozendict,
    flatten_func=lambda s: (tuple(s.values()), tuple(s.keys())),
    unflatten_func=lambda k, xs: frozendict(zip(k, xs)))

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
# V: Potential energy of edge
GN_to_V_Fn = Callable[[EdgeFeatures, NodeFeatures], float]
GN_to_T_Fn = Callable[[NodeFeatures], float]


def GNNet(
    V_fn: GN_to_V_Fn,
    initial_edge_embed_fn: Optional[GNUpdateEdgeFn],
    initial_node_embed_fn: Optional[GNUpdateEdgeFn],
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    T_fn: GN_to_T_Fn = None,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
        attention_reduce_fn: Optional[AttentionReduceFn] = None,
        N=1,):
    """Returns a method that applies a configured GraphNetwork.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Example usage::

      gn = GraphNetwork(update_edge_function,
      update_node_function, **kwargs)
      # Conduct multiple rounds of message passing with the same parameters:
      for _ in range(num_message_passing_steps):
        graph = gn(graph)

    Args:
      update_edge_fn: function used to update the edges or None to deactivate edge
        updates.
      update_node_fn: function used to update the nodes or None to deactivate node
        updates.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
      attention_logit_fn: function used to calculate the attention weights or
        None to deactivate attention mechanism.
      attention_normalize_fn: function used to normalize raw attention logits or
        None if attention mechanism is not active.
      attention_reduce_fn: function used to apply weights to the edge features or
        None if attention mechanism is not active.

    Returns:
      A method that applies the configured GraphNetwork.
    """
    """
    Args:
        V_fn = <function cal_graph.<locals>.edge_node_to_V_fn at 0x2b48b3fe0840>
        initial_edge_embed_fn = <function cal_graph.<locals>.initial_edge_emb_fn at 0x2b492fe2d378>
        initial_node_embed_fn = <function cal_graph.<locals>.initial_node_emb_fn at 0x2b492fe2dbf8>
        update_edge_fn = <function cal_graph.<locals>.update_edge_fn at 0x2b48b3ffa6a8>
        update_node_fn = <function cal_graph.<locals>.update_node_fn at 0x2b48b3ffa7b8>
        T_fn = <function cal_graph.<locals>.node_to_T_fn at 0x2b48b3fe08c8>
        update_global_fn = None
        aggregate_nodes_for_globals_fn = <function segment_sum at 0x2b47af860840>
        aggregate_edges_for_globals_fn = <function segment_sum at 0x2b47af860840>
        attention_logit_fn = None
        attention_normalize_fn = <function segment_softmax at 0x2b47af860d08>
        attention_reduce_fn = None
        N = 1
    
    
    """
    def not_both_supplied(x, y): 
        """
        Args:
            x: attention_logit_fn = None
            y: attention_reduce_fn = None
        
        Returns: Boolean
        """
        return  (x != y) and ((x is None) or (y is None))
    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                          ' supplied.'))

    def _ApplyGraphNet(graph):
        """Applies a configured GraphNetwork to a graph.

        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.

        In addition this implementation supports softmax attention over incoming
        edge features.

        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.
        """
        """
        Args:
          graph: a `GraphsTuple` containing the graph.

        Returns:
          Updated `GraphsTuple`.


        """
        # pylint: disable=g-long-lambda
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        # Equivalent to jnp.sum(n_node), but jittable

        # calculate number of nodes in graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0] # 9

        # calculate number of edges in graph
        sum_n_edge = senders.shape[0] # 18

        # check if all all node array are of same length = number of nodes
        if not tree.tree_all(
                tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError(
                'All node arrays in nest must contain the same number of nodes.')

        # Initial sent info
        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)

        # Initial received info
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        # i.e create an array per edge for global attributes
        global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        # i.e create an array per node for global attributes
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
        """ Node and edge embedding initialize------------------------below:"""
        # apply initial edge embeddings
        if initial_edge_embed_fn: # <function cal_graph.<locals>.initial_edge_emb_fn at 0x2b1cfd8707b8>
            edges = initial_edge_embed_fn(edges, sent_attributes, received_attributes,
                                          global_edge_attributes)
            """edges ==> dict_keys(['edge_embed', 'eij'])--> shape (18,5) and (18,1)"""
        # apply initial node embeddings #--------------------------------------->
        if initial_node_embed_fn: # <function cal_graph.<locals>.initial_node_emb_fn at 0x2b1cfd862e18>
            nodes = initial_node_embed_fn(nodes, sent_attributes,
                                          received_attributes, global_attributes)
            """nodes: dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])"""
        """message passing ------------------------------------------below"""
        # Now perform message passing for N times #------------------------------>
        for pass_i in range(N): # N times message passing
            if attention_logit_fn: # false usually
                logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                            global_edge_attributes)
                tree_calculate_weights = functools.partial(
                    attention_normalize_fn,
                    segment_ids=receivers,
                    num_segments=sum_n_node)
                weights = tree.tree_map(tree_calculate_weights, logits)
                edges = attention_reduce_fn(edges, weights)


            """update node dict"""
            if update_node_fn: # <function cal_graph.<locals>.update_node_fn at 0x2ac741250488>
                nodes = update_node_fn(
                    nodes, edges, senders, receivers,
                    global_attributes, sum_n_node)


            """update edge dict"""
            if update_edge_fn:
                senders_attributes = tree.tree_map( # DeviceArray([8, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
                    lambda n: n[senders], nodes)
                receivers_attributes = tree.tree_map( # DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
                    lambda n: n[receivers], nodes)
                #pdb.set_trace()
                edges = update_edge_fn(edges, senders_attributes, receivers_attributes, # <function cal_graph.<locals>.update_edge_fn 
                                       global_edge_attributes, pass_i == N-1)

        if update_global_fn: # not executed initially
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
            edge_gr_idx = jnp.repeat(
                graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
                lambda n: aggregate_nodes_for_globals_fn(
                    n, node_gr_idx, n_graph),
                nodes)
            edge_attribtutes = tree.tree_map(
                lambda e: aggregate_edges_for_globals_fn(
                    e, edge_gr_idx, n_graph),
                edges)
            # These pooled nodes are the inputs to the global update fn.
            globals_ = update_global_fn(
                node_attributes, edge_attribtutes, globals_)

        V = 0.0
        if V_fn is not None: # <function cal_graph.<locals>.edge_node_to_V_
            V += V_fn(edges, nodes) # DeviceArray(504125.16, dtype=float32)

        T = 0.0
        if T_fn is not None: # <function cal_graph.<locals>.node_to_T_fn
            T += T_fn(nodes) # DeviceArray(58.41084, dtype=float32)

        # pylint: enable=g-long-lambda
        return gn_graph.GraphsTuple(  # <module 'jraph._src.graph' 
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge), V, T

    return _ApplyGraphNet


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]


def get_fully_connected_senders_and_receivers(
    num_particles: int, self_edges: bool = False,
):
    """Returns senders and receivers for fully connected particles."""
    particle_indices = np.arange(num_particles)
    senders, receivers = np.meshgrid(particle_indices, particle_indices)
    senders, receivers = senders.flatten(), receivers.flatten()
    if not self_edges:
        mask = senders != receivers
        senders, receivers = senders[mask], receivers[mask]
    return senders, receivers


def cal_graph(params, graph, eorder=None, mpass=1,
              useT=True, useonlyedge=False, act_fn=SquarePlus):
    """
    Args:
        params: dict --> params.keys() --> dict_keys(['fb', 'fv', 'fe', 'ff1', 'ff2', 'ff3', 'fne', 'fneke', 'ke'])
        graph:  jraph.GraphsTuple
        eorder: DeviceArray([ 9, 10, 11, 12, 13, 14, 15, 16, 17,  0,  1,  2,  3,  4,  5, 6,  7,  8], dtype=int32)
        mpass: int -> = 1
        useT: Boolean --> True
        useonlyedge: Boolean  ---> True
        act_fn: activation fn --> <function SquarePlus at 0x2b47b42e7ea0>
        


    Returns: what GNNet(graph) returns
    """
    fb_params = params["fb"]
    fne_params = params["fne"]
    fneke_params = params["fneke"]
    fv_params = params["fv"]
    fe_params = params["fe"]
    ff1_params = params["ff1"]
    ff2_params = params["ff2"]
    ff3_params = params["ff3"]
    ke_params = params["ke"]

    num_species = 1 # means?

    def onehot(n):
        def fn(n):
            out = jax.nn.one_hot(n, num_species)
            return out
        out = vmap(fn)(n.reshape(-1,))
        return out

    def fne(n): # called by  initial_node_emb_fn
        """
        Args:
            n = DeviceArray([[1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.]], dtype=float32)
                
        Returns: DeviceArray,  shape  -> (9,5)        
        """
        def fn(ni):
            out = forward_pass(fne_params, ni, activation_fn=lambda x: x) # in src.models.py
            return out
        out = vmap(fn, in_axes=(0))(n)
        return out

    def fneke(n):
        def fn(ni):
            out = forward_pass(fneke_params, ni, activation_fn=lambda x: x)
            return out
        out = vmap(fn, in_axes=(0))(n)
        return out

    def fb(e): # called by initial_edge_emb_fn
        """
        Args:
            e: Device array -> shape (18,1)
        
        """
        def fn(eij):
            out = forward_pass(fb_params, eij, activation_fn=act_fn)  # in src.models.py
            return out
        out = vmap(fn, in_axes=(0))(e) # shape (18, 5)
        return out

    def fv(n, e, s, r, sum_n_node): # called by update_node_fn
        """
        Args:
            nodes: frozendict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
            edges: frozendict --> dict_keys(['edge_embed', 'eij'])
            senders = DeviceArray([8, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
            receivers = DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
            globals_: dict = {}
            sum_n_node: int = 9
        
        Returns: DeviceArray: (9, 5)
        
        """
        c1ij = jnp.hstack([n[r], e]) # shape: (18, 10)
        out = vmap(lambda x: forward_pass(fv_params, x))(c1ij) # shape (18, 5)
        return n + jax.ops.segment_sum(out, r, sum_n_node) # shape: (9, 5)

    def fe(e, s, r): # called by update_edge_fn
        """
        Args:
            e: devicearray --> shape : (18, 5)
            s: devicearray --> shape : (18, 5)

        Returns: DeviceArray --> shape : (18, 5)
        """
        def fn(hi, hj):
            c2ij = hi * hj
            out = forward_pass(fe_params, c2ij, activation_fn=act_fn)
            return out
        out = e + vmap(fn, in_axes=(0, 0))(s, r)
        return out

    def ff1(e): # called by edge_node_to_V_fn
        """
        Args:
            e -> device array --> shape --> (18, 5)
        
        Returns: device array --> shape --> (18, 1)
        """
        def fn(eij):
            out = forward_pass(ff1_params, eij, activation_fn=act_fn)
            return out
        out = vmap(fn)(e) # shape --> (18, 1)
        return out

    def ff2(n):
        def fn(ni):
            out = forward_pass(ff2_params, ni, activation_fn=act_fn)
            return out
        out = vmap(fn)(n)
        return out

    def ff3(n):
        def fn(ni):
            out = forward_pass(ff3_params, ni, activation_fn=act_fn)
            return out
        out = vmap(fn)(n)
        return out

    def ke(n):# called by node_to_T_fn 
        """
        Args:
            n: device array --> shape (9, 6)

        Returns: device array --> shape --> (9, 1)
        """
        def fn(ni):
            out = forward_pass(ke_params, ni, activation_fn=act_fn) # <function forward_pass at 0x2afce4e300d0>
            return out
        out = vmap(fn)(n)
        return out

    # ================================================================================

    def initial_edge_emb_fn(edges, senders, receivers, globals_):
        """Returns 
        Args:
            edges: dict --> {}
            globals: 
            senders: dict --> dict_keys(['position', 'type', 'velocity'])
            receivers: dict --> dict_keys(['position', 'type', 'velocity'])



        
        Return: Frozendict : keys:{edge_embed, eij}
        """

        del edges, globals_ # deleting since empty initially ?
        dr = (senders["position"] - receivers["position"]) # r2 - r2 => dr, dr.shape: (18, 2)
        # eij = dr
        eij = jnp.sqrt(jnp.square(dr).sum(axis=1, keepdims=True)) # edge ijth? --> (18, 1)
        emb = fb(eij) # emb means embedding? shape --> (18, 5)
        return frozendict({"edge_embed": emb, "eij": eij})

    def initial_node_emb_fn(nodes, sent_edges, received_edges, globals_):
        """
        Args:
            nodes: dict: dict_keys(['position', 'velocity', 'type'])
            sent_edges: dict_keys(['position', 'type', 'velocity'])
            received_edges: dict_keys(['position', 'type', 'velocity'])
            globals_: <built-in function globals>

        
        Return: Frozendict:  keys : {node_embed, node_pos_embed, node_vel_embed}
        """
        del sent_edges, received_edges, globals_
        type_of_node = nodes["type"] # DeviceArray([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
        ohe = onehot(type_of_node) # DeviceArray([1, 1, 1, 1, 1, 1, 1, 1, 1].T
        emb = fne(ohe) # shape -- (9, 5) # <function cal_graph.<locals>.fne at 0x2ac7410c0488>
        emb_pos = jnp.hstack([emb, nodes["position"]]) # (9, 7)
        emb_vel = jnp.hstack(     # (9, 6)
            [fneke(ohe), jnp.sum(jnp.square(nodes["velocity"]), axis=1, keepdims=True)])
        return frozendict({"node_embed": emb,
                           "node_pos_embed": emb_pos,
                           "node_vel_embed": emb_vel,
                           })

    def update_node_fn(nodes, edges, senders, receivers, globals_, sum_n_node):
        """
        Args:
            nodes: frozendict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
            edges: frozendict --> dict_keys(['edge_embed', 'eij'])
            senders = DeviceArray([8, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
            receivers = DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
            globals_ = {}
            sum_n_node = 9
        
        Returns: frozendict: dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
        
        """

        del globals_
        emb = fv(nodes["node_embed"], edges["edge_embed"], # <function cal_graph.<locals>.fv at 0x2ac7410c0158>
                 senders, receivers, sum_n_node)
        #emb.shape --> (9, 5)
        n = dict(nodes)
        """ updating node embedding"""
        n.update({"node_embed": emb}) # dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
        return frozendict(n)

    def update_edge_fn(edges, senders, receivers, globals_, last_step):
        """
        Args;
            nodes: frozendict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
            edges: frozendict --> dict_keys(['edge_embed', 'eij'])
            senders: frozendict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
            receivers: dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
            globals_ = {}
            last_step = True
        
        
        Returns: dict:  dict_keys(['edge_embed', 'eij'])
        """
        del globals_
        emb = fe(edges["edge_embed"], senders["node_embed"], # <function cal_graph.<locals>.fe at 0x2afcf4c09f28>
                 receivers["node_embed"])
        # emb.shape --> (18, 5)
        if last_step: # true
            if eorder is not None: # DeviceArray([ 9, 10, 11, 12, 13, 14, 15, 16, 17,  0,  1,  2,  3,  4,  5, 6,  7,  8], dtype=int32)
                emb = (emb + fe(edges["edge_embed"][eorder], # <function cal_graph.<locals>.fe
                       receivers["node_embed"], senders["node_embed"])) / 2
        return frozendict({"edge_embed": emb, "eij": edges["eij"]})

    if useonlyedge: # True
        def edge_node_to_V_fn(edges, nodes):
            """
            Args:
                nodes: frozendict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
                edges: frozendict --> dict_keys(['edge_embed', 'eij'])

            Returns: <bound method sum of DeviceArray
            """
            vij = ff1(edges["edge_embed"])  # <function cal_graph.<locals>.ff1 
            # vij --> shape --> (18,1)
            # print(vij, edges["eij"])
            return vij.sum() # <bound method sum of DeviceArray(
    else:
        def edge_node_to_V_fn(edges, nodes):
            vij = ff1(edges["edge_embed"]).sum()
            vi = 0
            vi = vi + ff2(nodes["node_embed"]).sum()
            vi = vi + ff3(nodes["node_pos_embed"]).sum()
            return vij + vi

    def node_to_T_fn(nodes):
        """
        Args:
            nodes: dict --> dict_keys(['node_embed', 'node_pos_embed', 'node_vel_embed'])
        
        Returns : device arrray --> 1 value
        """
        return ke(nodes["node_vel_embed"]).sum() # <function cal_graph.<locals>.ke # DeviceArray(58.41084, dtype=float32)

    if not(useT): # false
        node_to_T_fn = None

    Net = GNNet(N=mpass, # N : message passing?   #/src/graph.py(88)GNNet()
                V_fn=edge_node_to_V_fn,
                T_fn=node_to_T_fn,
                initial_edge_embed_fn=initial_edge_emb_fn,
                initial_node_embed_fn=initial_node_emb_fn,
                update_edge_fn=update_edge_fn,
                update_node_fn=update_node_fn)

    return Net(graph) # returns what Net(graph) returns
