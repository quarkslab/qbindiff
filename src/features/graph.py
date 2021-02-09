import networkx
from qbindiff.features.visitor import FunctionFeature, Environment
from qbindiff.loader.function import Function


class GraphNbBlock(FunctionFeature):
    """Number of basic blocks in the function"""
    name = "graph_nblock"
    key = "Gnb"

    def visit_function(self, fun: Function, env: Environment):
        n_node = len(fun.graph)
        env.add_feature("N_BLOCK", n_node)


class GraphMeanInstBlock(FunctionFeature):
    """Mean of instruction per basic blocks in the function"""
    name = "graph_mean_inst_block"
    key = "Gmib"

    def visit_function(self, fun: Function, env: Environment):
        n_node = len(fun.graph)
        metric = sum(map(len, fun.values())) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_INST_P_BLOCK', metric)


class GraphMeanDegree(FunctionFeature):
    """Mean degree of the function"""
    name = "graph_mean_degree"
    key = "Gmd"

    def visit_function(self, fun: Function, env: Environment):
        n_node = len(fun.graph)
        metric = sum(x for a, x in fun.graph.degree()) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_DEGREE', metric)


class GraphDensity(FunctionFeature):
    """Density of the function flow graph"""
    name = "graph_density"
    key = "Gd"

    def visit_function(self, fun: Function, env: Environment):
        env.add_feature('DENSITY', networkx.density(fun.graph))


class GraphNbComponents(FunctionFeature):
    """Number of components in the function (non-connected flow graphs)"""
    name = "graph_num_components"
    key = "Gnc"

    def visit_function(self, fun: Function, env: Environment):
        components = list(networkx.connected_components(fun.graph.to_undirected()))
        env.add_feature("N_COMPONENTS", len(components))


class GraphDiameter(FunctionFeature):
    """Diamater of the function flow graph"""
    name = "graph_diameter"
    key = "Gdi"

    def visit_function(self, fun: Function, env: Environment):
        components = list(networkx.connected_components(fun.graph.to_undirected()))
        if components:
            max_dia = max(networkx.diameter(networkx.subgraph(fun.graph, x).to_undirected()) for x in components)
        else:
            max_dia = 0
        env.add_feature("MAX_DIAMETER", max_dia)


class GraphTransitivity(FunctionFeature):
    """Transitivity of the function flow graph"""
    name = "graph_transitivity"
    key = "Gt"

    def visit_function(self, fun: Function, env: Environment):
        env.add_feature('TRANSITIVITY', networkx.transitivity(fun.graph))

