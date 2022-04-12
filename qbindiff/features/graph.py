import networkx
from qbindiff.features.visitor import FunctionFeature, FeatureCollector
from qbindiff.loader.function import Function
import community


class BBlockNb(FunctionFeature):
    """Number of basic blocks in the function"""

    name = "bblock_nb"
    key = "bnb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(function.flowgraph.nodes)
        collector.add_feature(self.key, value)


class JumpNb(FunctionFeature):
    """Number of jumps in the function"""

    name = "jump_nb"
    key = "jnb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(function.flowgraph.edges)
        collector.add_feature(self.key, value)


class MaxParentNb(FunctionFeature):
    """Maximum number of parent of a bblock in the function"""

    name = "max_parent_nb"
    key = "maxp"

    def visit_function(self, function: Function, collector: FeatureCollector):
        # FIXME: Change to use binexport module
        value = max(
            len(
                function.flowgraph.predecessors(bblock) for bblock in function.flowgraph
            )
        )  # WRONG!
        # value = max(len(bb.parents) for bb in function)
        collector.add_feature(self.key, value)


class MaxChildNb(FunctionFeature):
    """Maximum number of children of a bblock in the function"""

    name = "max_child_nb"
    key = "maxc"

    def visit_function(self, function: Function, collector: FeatureCollector):
        # FIXME: Change to use binexport module
        value = max(
            len(function.flowgraph.successors(bblock) for bblock in function.flowgraph)
        )  # WRONG!
        # value = max(len(bb.children) for bb in function)
        collector.add_feature(self.key, value)


class MaxInsNB(FunctionFeature):
    """Max number of instructions per basic blocks in the function"""

    name = "max_ins_nb"
    key = "maxins"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = max(len(bblock) for bblock in function)
        collector.add_feature(self.key, value)


class MeanInsNB(FunctionFeature):
    """Mean number of instructions per basic blocks in the function"""

    name = "mean_ins_nb"
    key = "meanins"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = sum(len(bblock) for bblock in function) / len(function)
        collector.add_feature(self.key, value)


class InstNB(FunctionFeature):
    """Number of instructions per basic blocks in the function"""

    name = "ins_nb"
    key = "inb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = sum(len(bblock) for bblock in function)
        collector.add_feature(self.key, value)


class GraphMeanDegree(FunctionFeature):
    """Mean degree of the function"""

    name = "graph_mean_degree"
    key = "Gmd"

    def visit_function(self, function: Function, collector: FeatureCollector):
        n_node = len(function.flowgraph)
        value = (
            sum(x for a, x in function.flowgraph.degree) / n_node if n_node != 0 else 0
        )
        collector.add_feature(self.key, value)


class GraphDensity(FunctionFeature):
    """Density of the function flow graph"""

    name = "graph_density"
    key = "Gd"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = networkx.density(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphNbComponents(FunctionFeature):
    """Number of components in the function (non-connected flow graphs)"""

    name = "graph_num_components"
    key = "Gnc"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(
            list(networkx.connected_components(function.flowgraph.to_undirected()))
        )
        collector.add_feature(self.key, value)


class GraphDiameter(FunctionFeature):
    """Diamater of the function flow graph"""

    name = "graph_diameter"
    key = "Gdi"

    def visit_function(self, function: Function, collector: FeatureCollector):
        components = list(
            networkx.connected_components(function.flowgraph.to_undirected())
        )
        if components:
            value = max(
                networkx.diameter(
                    networkx.subgraph(function.flowgraph, x).to_undirected()
                )
                for x in components
            )
        else:
            value = 0
        collector.add_feature(self.key, value)


class GraphTransitivity(FunctionFeature):
    """Transitivity of the function flow graph"""

    name = "graph_transitivity"
    key = "Gt"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = networkx.transitivity(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphCommunities(FunctionFeature):
    """Number of graph communities (Louvain modularity)"""

    name = "graph_community"
    key = "Gcom"

    def visit_function(self, function: Function, collector: FeatureCollector) -> None:
        partition = community.best_partition(function.flowgraph.to_undirected())
        if len(function) > 1:
            value = max(x for x in partition.values() if x != function.addr)
        else:
            value = 0
        collector.add_feature(self.key, value)
