import networkx
import community
from qbindiff.features.visitor import FunctionFeatureExtractor


class GraphNbBlock(FunctionFeatureExtractor):
    name = "graph_nblock"
    key = "Gnb"

    def call(self, env, function):
        n_node = len(function.graph)
        env.add_feature("N_BLOCK", n_node)


class GraphMeanInstBlock(FunctionFeatureExtractor):
    name = "graph_mean_inst_block"
    key = "Gmib"

    def call(self, env, function):
        n_node = len(function.graph)
        metric = sum(map(len, function.values())) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_INST_P_BLOCK', metric)


class GraphMeanDegree(FunctionFeatureExtractor):
    name = "graph_mean_degree"
    key = "Gmd"

    def call(self, env, function):
        n_node = len(function.graph)
        metric = sum(x for a,x in function.graph.degree()) / n_node if n_node != 0 else 0
        env.add_feature('MEAN_DEGREE', metric)


class GraphDensity(FunctionFeatureExtractor):
    name = "graph_density"
    key = "Gd"

    def call(self, env, function):
        env.add_feature('DENSITY', networkx.density(function.graph))


class GraphNbComponents(FunctionFeatureExtractor):
    name = "graph_num_components"
    key = "Gnc"

    def call(self, env, function):
        components = list(networkx.connected_components(function.graph.to_undirected()))
        env.add_feature("N_COMPONENTS", len(components))


class GraphDiameter(FunctionFeatureExtractor):
    name = "graph_diameter"
    key = "Gdi"

    def call(self, env, function):
        components = list(networkx.connected_components(function.graph.to_undirected()))
        if components:
            max_dia = max(networkx.diameter(networkx.subgraph(function.graph, x).to_undirected()) for x in components)
        else:
            max_dia = 0
        env.add_feature("MAX_DIAMETER", max_dia)


class GraphTransitivity(FunctionFeatureExtractor):
    name = "graph_transitivity"
    key = "Gt"

    def call(self, env, function):
        env.add_feature('TRANSITIVITY', networkx.transitivity(function.graph))


class GraphCommunities(FunctionFeatureExtractor):
    name = "graph_community"
    key = "Gcom"

    def call(self, env, function):
        partition = community.best_partition(function.graph.to_undirected())
        if len(function) > 1:
            metric = max(x for x in partition.values() if x != function.addr)
        else:
            metric = 0
        env.add_feature('COMMUNITIES', metric)
