import networkx
import community
from qbindiff.features.visitor import FunctionFeatureExtractor, Environment
from qbindiff.loader.function import Function


class GraphNbBlock(FunctionFeatureExtractor):
    """Number of basic blocks in the function"""
    name = "graph_nblock"
    key = "Gnb"

    def call(self, env: Environment, fun: Function):
        n_node = len(fun.graph)
        env.add_feature("N_BLOCK", n_node)


class GraphMeanInstBlock(FunctionFeatureExtractor):
    """Mean of instruction per basic blocks in the function"""
    name = "graph_mean_inst_block"
    key = "Gmib"

    def call(self, env: Environment, fun: Function):
        n_node = len(fun.graph)
        metric = sum(map(len, fun.values())) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_INST_P_BLOCK', metric)


class GraphMeanDegree(FunctionFeatureExtractor):
    """Mean degree of the function"""
    name = "graph_mean_degree"
    key = "Gmd"

    def call(self, env: Environment, fun: Function):
        n_node = len(fun.graph)
        metric = sum(x for a, x in fun.graph.degree()) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_DEGREE', metric)


class GraphDensity(FunctionFeatureExtractor):
    """Density of the function flow graph"""
    name = "graph_density"
    key = "Gd"

    def call(self, env:Environment, fun: Function):
        env.add_feature('DENSITY', networkx.density(fun.graph))


class GraphNbComponents(FunctionFeatureExtractor):
    """Number of components in the function (non-connected flow graphs)"""
    name = "graph_num_components"
    key = "Gnc"

    def call(self, env: Environment, fun: Function):
        components = list(networkx.connected_components(fun.graph.to_undirected()))
        env.add_feature("N_COMPONENTS", len(components))


class GraphDiameter(FunctionFeatureExtractor):
    """Diamater of the function flow graph"""
    name = "graph_diameter"
    key = "Gdi"

    def call(self, env: Environment, fun: Function):
        components = list(networkx.connected_components(fun.graph.to_undirected()))
        if components:
            max_dia = max(networkx.diameter(networkx.subgraph(fun.graph, x).to_undirected()) for x in components)
        else:
            max_dia = 0
        env.add_feature("MAX_DIAMETER", max_dia)


class GraphTransitivity(FunctionFeatureExtractor):
    """Transitivity of the function flow graph"""
    name = "graph_transitivity"
    key = "Gt"

    def call(self, env: Environment, fun: Function):
        env.add_feature('TRANSITIVITY', networkx.transitivity(fun.graph))


class GraphCommunities(FunctionFeatureExtractor):
    """Number of graph communities (Louvain modularity)"""
    name = "graph_community"
    key = "Gcom"

    def call(self, env: Environment, fun: Function):
        partition = community.best_partition(fun.graph.to_undirected())
        if len(fun) > 1:
            metric = max(x for x in partition.values() if x != fun.addr)
        else:
            metric = 0
        env.add_feature('COMMUNITIES', metric)
