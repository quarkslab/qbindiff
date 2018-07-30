import networkx
import community
from qbindiff.features.visitor import FeatureExtractor


class GraphNbBlock(FeatureExtractor):
    name = "graph_nblock"
    key = "Gnb"

    def call(self, env, function):
        n_node = len(function.graph)
        env.add_feature("N_BLOCK", n_node)

class GraphMeanInstBlock(FeatureExtractor):
    name = "graph_mean_inst_block"
    key = "Gmib"

    def call(self, env, function):
        n_node = len(function.graph)
        n_elements = map(len, function.values())
        metric = sum(n_elements) / n_node
        env.add_feature('MEAN_INST_P_BLOCK', metric)

class GraphMeanDegree(FeatureExtractor):
    name = "graph_mean_degree"
    key = "Gmd"

    def call(self, env, function):
        n_node = len(function.graph)
        metric = sum(x for a,x in function.graph.degree()) / n_node
        env.add_feature('MEAN_DEGREE', metric)

class GraphDensity(FeatureExtractor):
    name = "graph_density"
    key = "Gd"

    def call(self, env, function):
        env.add_feature('DENSITY', networkx.density(function.graph))

class GraphNbComponents(FeatureExtractor):
    name = "graph_num_components"
    key = "Gnc"

    def call(self, env, function):
        components = list(networkx.connected_components(function.graph.to_undirected()))
        env.add_feature("N_COMPONENTS", len(components))

class GraphDiameter(FeatureExtractor):
    name = "graph_diameter"
    key = "Gdi"

    def call(self, env, function):
        components = list(networkx.connected_components(function.graph.to_undirected()))
        max_dia = max(networkx.diameter(networkx.subgraph(function.graph, x).to_undirected()) for x in components)
        env.add_feature("MAX_DIAMETER", max_dia)

class GraphTransitivity(FeatureExtractor):
    name = "graph_transitivity"
    key = "Gt"

    def call(self, env, function):
        env.add_feature('TRANSITIVITY', networkx.transitivity(function.graph))

class GraphCommunities(FeatureExtractor):
    name = "graph_community"
    key = "Gcom"

    def call(self, env, function):
        partition = community.best_partition(function.graph.to_undirected())
        if len(function) > 1:
            metric = max(x for x in partition.values() if x != function.addr)
        else:
            metric = 0
        env.add_feature('COMMUNITIES', metric)
