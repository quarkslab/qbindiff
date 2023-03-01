import networkx
import numpy as np
from qbindiff.features.extractor import FunctionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Function


class BBlockNb(FunctionFeatureExtractor):
    """Number of basic blocks in the function"""

    key = "bnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(function.flowgraph.nodes)
        collector.add_feature(self.key, value)


class CyclomaticComplexity(FunctionFeatureExtractor):
    """ Cyclomatic complexity of the function """

    key='cc'
    
    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        e = len(function.edges)
        n = len([n for n in function.nodes])
        components = len([c for c in networkx.weakly_connected_components(function.callgraph)])
        value = e - n + 2*components
        collector.add_feature(self.key, value)

class MDIndex(FunctionFeatureExtractor):
    """ MD-Index of the function, based on : https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-IST-091/MP-IST-091-26.pdf.
    A slightly modified version of it : notice the topological sort is only available for DAG graphs (which may not always be the case)."""

    key='mdidx'

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):

        try :
            topological_sort = list(networkx.topological_sort(function.flowgraph))
            sort_ok = True
        else :
            sort_ok = False
    
        if sort_ok : 
            value = np.sum([1/math.sqrt(topological_sort.index(src) + math.sqrt(2)*function.flowgraph.in_degree(src) + math.sqrt(3)*function.flowgraph.out_degree(src) + math.sqrt(5)*function.flowgraph.in_degree(dst) + math.sqrt(7)*function.flowgraph.out_degree(dst)) for (src, dst) in function.edges])
            
        else :
            value = np.sum([1/math.sqrt(math.sqrt(2)*function.flowgraph.in_degree(src) + math.sqrt(3)*function.flowgraph.out_degree(src) + math.sqrt(5)*function.flowgraph.in_degree(dst) + math.sqrt(7)*function.flowgraph.out_degree(dst)) for (src, dst) in function.edges])

        collector.add_feature(self.key, value)

class JumpNb(FunctionFeatureExtractor):
    """Number of jumps in the function"""

    key = "jnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(function.flowgraph.edges)
        collector.add_feature(self.key, value)


class MaxParentNb(FunctionFeatureExtractor):
    """Maximum number of parent of a bblock in the function"""

    key = "maxp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = max(
            len(function.flowgraph.predecessors(bblock))
            for bblock in function.flowgraph
        )
        # value = max(len(bb.parents) for bb in function)
        collector.add_feature(self.key, value)


class MaxChildNb(FunctionFeatureExtractor):
    """Maximum number of children of a bblock in the function"""

    key = "maxc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = max(
            len(function.flowgraph.successors(bblock)) for bblock in function.flowgraph
        )
        # value = max(len(bb.children) for bb in function)
        collector.add_feature(self.key, value)


class MaxInsNB(FunctionFeatureExtractor):
    """Max number of instructions per basic blocks in the function"""

    key = "maxins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = max(len(bblock) for bblock in function)
        collector.add_feature(self.key, value)


class MeanInsNB(FunctionFeatureExtractor):
    """Mean number of instructions per basic blocks in the function"""

    key = "meanins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = sum(len(bblock) for bblock in function) / len(function)
        collector.add_feature(self.key, value)


class InstNB(FunctionFeatureExtractor):
    """Number of instructions in the function"""

    key = "totins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = sum(len(bblock) for bblock in function)
        collector.add_feature(self.key, value)


class GraphMeanDegree(FunctionFeatureExtractor):
    """Mean degree of the function"""

    key = "Gmd"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        n_node = len(function.flowgraph)
        value = (
            sum(d for _, d in function.flowgraph.degree) / n_node if n_node != 0 else 0
        )
        collector.add_feature(self.key, value)


class GraphDensity(FunctionFeatureExtractor):
    """Density of the function flow graph"""

    key = "Gd"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = networkx.density(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphNbComponents(FunctionFeatureExtractor):
    """Number of components in the function (non-connected flow graphs)"""

    key = "Gnc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(
            list(networkx.connected_components(function.flowgraph.to_undirected()))
        )
        collector.add_feature(self.key, value)


class GraphDiameter(FunctionFeatureExtractor):
    """Diamater of the function flow graph"""

    key = "Gdi"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
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


class GraphTransitivity(FunctionFeatureExtractor):
    """Transitivity of the function flow graph"""

    key = "Gt"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = networkx.transitivity(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphCommunities(FunctionFeatureExtractor):
    """Number of graph communities (Louvain modularity)"""

    key = "Gcom"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        import community

        partition = community.best_partition(function.flowgraph.to_undirected())
        if len(function) > 1:
            value = max(x for x in partition.values() if x != function.addr)
        else:
            value = 0
        collector.add_feature(self.key, value)
