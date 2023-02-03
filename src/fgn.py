"""

This code is a Python implementation of a deep neural network architecture for graph representation learning. 
The architecture consists of a GraphEncodeNet which has three components:

GraphMapFeatures: this component encodes the graph structure and node/edge features into node/edge embeddings.

GraphNetwork: this component uses the node/edge embeddings as input and applies multiple rounds of 
message passing among nodes and edges to update the node/edge embeddings.

GraphNetwork: this component applies final processing to the node/edge embeddings to produce the final output.

The neural network is trained with a loss function which is the sum of the energy of all nodes in the graph. 
The output of the network can be used to calculate the acceleration of each node in the graph.


"""


from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, lax, random
from jax_md.nn import GraphNetEncoder
from jraph import GraphMapFeatures, GraphNetwork, GraphsTuple

from src.models import SquarePlus, forward_pass, initialize_mlp


class GraphEncodeNet():
    def __init__(self, N, embedding_fn, model_fn, final_fn):
        self.N = N
        self._encoder = GraphMapFeatures(
            embedding_fn('EdgeEncoder'),
            embedding_fn('NodeEncoder'),
            embedding_fn('GlobalEncoder'))
        self._propagation_network = GraphNetwork(
            model_fn('EdgeFunction'),
            model_fn('NodeFunction'),
            model_fn('GlobalFunction'), aggregate_edges_for_globals_fn=lambda *x: jnp.array([0.0]))
        self._final = GraphNetwork(
            final_fn('EdgeFunction'),
            final_fn('NodeFunction'),
            final_fn('GlobalFunction'), aggregate_edges_for_globals_fn=lambda *x: jnp.array([0.0]))

    def __call__(self, graph):
        output = self._encoder(graph)
        for _ in range(self.N):
            output = self._propagation_network(output)
        output = self._final(output)
        return output


def cal(params, graph, mpass=1):
    ee_params = params["ee_params"]
    ne_params = params["ne_params"]
    e_params = params["e_params"]
    n_params = params["n_params"]
    g_params = params["g_params"]

    def node_em(nodes):
        out = jnp.hstack([v for k, v in nodes.items()])

        def fn(out):
            return forward_pass(ne_params, out, activation_fn=SquarePlus)
        out = jax.vmap(fn)(out)
        return {"embed": out}

    def edge_em(edges):
        out = edges["dij"]
        out = jax.vmap(lambda p, x: forward_pass(p, x.reshape(-1)),
                       in_axes=(None, 0))(ee_params, out)
        return {"embed": out}

    embedding = {
        "EdgeEncoder": edge_em,
        "NodeEncoder": node_em,
        "GlobalEncoder": None,
    }

    def embedding_fn(arg): return embedding[arg]

    def edge_fn(edges, sent_attributes, received_attributes, global_):
        out = jnp.hstack([edges["embed"], sent_attributes["embed"],
                          received_attributes["embed"]])
        out = jax.vmap(forward_pass, in_axes=(None, 0))(e_params, out)
        return {"embed": out}

    def node_fn(nodes, sent_attributes, received_attributes, global_):
        out = jnp.hstack([nodes["embed"], sent_attributes["embed"],
                          received_attributes["embed"]])
        out = jax.vmap(forward_pass, in_axes=(None, 0))(n_params, out)
        return {"embed": out}

    model = {
        "EdgeFunction": edge_fn,
        "NodeFunction": node_fn,
        "GlobalFunction": None,
    }

    def model_fn(arg): return model[arg]

    final = {
        "EdgeFunction": lambda *x: x[0],
        "NodeFunction": lambda *x: x[0],
        "GlobalFunction": lambda node_attributes, edge_attribtutes, globals_:
        forward_pass(g_params, node_attributes["embed"].reshape(-1)),
        # "GlobalFunction": lambda node_attributes, edge_attribtutes, globals_:
        #     node_attributes["embed"].sum()
    }

    def final_fn(arg): return final[arg]

    net = GraphEncodeNet(mpass, embedding_fn, model_fn, final_fn)

    graph = net(graph)

    return graph


def cal_energy(params, graph, **kwargs):
    graph = cal(params, graph, **kwargs)
    return graph.globals.sum()


def cal_acceleration(params, graph, **kwargs):
    graph = cal(params, graph, **kwargs)
    acc_params = params["acc_params"]
    out = jax.vmap(forward_pass, in_axes=(None, 0))(
        acc_params, graph.nodes["embed"])
    return out
