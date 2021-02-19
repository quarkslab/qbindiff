import networkx
from qbindiff.features.visitor import FunctionFeature, Environment
from qbindiff.loader.function import Function
import community


class BBlockNb(FunctionFeature):
    """Number of basic blocks in the function"""
    name = "bblock_nb"
    key = "bnb"

    def visit_function(self, fun: Function, env: Environment):
        value = len(fun.flowgraph.nodes)
        env.add_feature(self.key, value)


class JumpNb(FunctionFeature):
    """Number of jumps in the function"""
    name = "jump_nb"
    key = "jnb"

    def visit_function(self, fun: Function, env: Environment):
        value = len(fun.flowgraph.edges)
        env.add_feature(self.key, value)


class MaxParentNb(FunctionFeature):
    """Maximum number of parent of a bblock in the function"""
    name = "max_parent_nb"
    key = "maxp"

    def visit_function(self, fun: Function, env: Environment):
        # FIXME: Change to use binexport module
        value = max(len(fun.flowgraph.predecessors(bblock) for bblock in fun.flowgraph))  # WRONG!
        # value = max(len(bb.parents) for bb in fun)
        env.add_feature(self.key, value)


class MaxChildNb(FunctionFeature):
    """Maximum number of children of a bblock in the function"""
    name = "max_child_nb"
    key = "maxc"

    def visit_function(self, fun: Function, env: Environment):
        # FIXME: Change to use binexport module
        value = max(len(fun.flowgraph.successors(bblock) for bblock in fun.flowgraph))  # WRONG!
        # value = max(len(bb.children) for bb in fun)
        env.add_feature(self.key, value)


class MaxInsNB(FunctionFeature):
    """Max number of instructions per basic blocks in the function"""
    name = "max_ins_nb"
    key = "maxins"

    def visit_function(self, fun: Function, env: Environment):
        value = max(len(bblock) for bblock in fun)
        env.add_feature(self.key, value)


class MeanInsNB(FunctionFeature):
    """Mean number of instructions per basic blocks in the function"""
    name = "mean_ins_nb"
    key = "meanins"

    def visit_function(self, fun: Function, env: Environment):
        value = sum(len(bblock) for bblock in fun) / len(fun)
        env.add_feature(self.key, value)


class InstNB(FunctionFeature):
    """Number of instructions per basic blocks in the function"""
    name = "ins_nb"
    key = "inb"

    def visit_function(self, fun: Function, env: Environment):
        value = sum(len(bblock) for bblock in fun)
        env.add_feature(self.key, value)


class GraphMeanDegree(FunctionFeature):
    """Mean degree of the function"""
    name = "graph_mean_degree"
    key = "Gmd"

    def visit_function(self, fun: Function, env: Environment):
        n_node = len(fun.flowgraph)
        value = sum(x for a, x in fun.flowgraph.degree) / n_node if n_node != 0 else 0
        env.add_feature(self.key, value)


class GraphDensity(FunctionFeature):
    """Density of the function flow graph"""
    name = "graph_density"
    key = "Gd"

    def visit_function(self, fun: Function, env: Environment):
        value = networkx.density(fun.flowgraph)
        env.add_feature(self.key, value)


class GraphNbComponents(FunctionFeature):
    """Number of components in the function (non-connected flow graphs)"""
    name = "graph_num_components"
    key = "Gnc"

    def visit_function(self, fun: Function, env: Environment):
        value = len(list(networkx.connected_components(fun.flowgraph.to_undirected())))
        env.add_feature(self.key, value)


class GraphDiameter(FunctionFeature):
    """Diamater of the function flow graph"""
    name = "graph_diameter"
    key = "Gdi"

    def visit_function(self, fun: Function, env: Environment):
        components = list(networkx.connected_components(fun.flowgraph.to_undirected()))
        if components:
            value = max(networkx.diameter(networkx.subgraph(fun.flowgraph, x).to_undirected()) for x in components)
        else:
            value = 0
        env.add_feature(self.key, value)


class GraphTransitivity(FunctionFeature):
    """Transitivity of the function flow graph"""
    name = "graph_transitivity"
    key = "Gt"

    def visit_function(self, fun: Function, env: Environment):
        value = networkx.transitivity(fun.flowgraph)
        env.add_feature(self.key, value)


class GraphCommunities(FunctionFeature):
    """Number of graph communities (Louvain modularity)"""
    name = "graph_community"
    key = "Gcom"

    def visit_function(self, function: Function, env: Environment) -> None:
        partition = community.best_partition(function.flowgraph.to_undirected())
        if len(function) > 1:
            value = max(x for x in partition.values() if x != function.addr)
        else:
            value = 0
        env.add_feature(self.key, value)
