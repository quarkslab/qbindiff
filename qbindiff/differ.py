# third-party imports
import numpy as np
import scipy.io
import json
import scipy.spatial.distance
from networkx import DiGraph
from collections.abc import Generator, Iterator

from qbindiff.abstract import GenericGraph
from qbindiff.loader import Program, Function
from qbindiff.matcher import Matcher
from qbindiff.mapping import Mapping
from qbindiff.features.extractor import FeatureCollector
from qbindiff.visitor import Visitor, NoVisitor, ProgramVisitor
from typing import Any
from qbindiff.types import (
    Anchors,
    RawMapping,
    PathLike,
    Positive,
    Ratio,
    Graph,
    AdjacencyMatrix,
)


class Differ:
    """
    Abstract class that perform the NAP diffing between two generic graphs.
    
    :param distance: the distance function used when comparing the feature vector
                     extracted from the graphs
    :param sparsity_ratio: the sparsity ratio enforced to the similarity matrix
    :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0)
                     and edge similarity (tradeoff=0.0)
    :param epsilon: perturbation parameter to enforce convergence and speed up computation.
                    The greatest the fastest, but least accurate
    :param maxiter: maximum number of message passing iterations
    """

    DTYPE = np.float32

    def __init__(self, primary: Graph, secondary: Graph, 
        distance: str = "canberra",
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,visitor: Visitor = None):
        
        self.distance = distance
        self.sparsity_ratio = sparsity_ratio
        self.tradeoff = tradeoff
        self.epsilon = epsilon
        self.maxiter = maxiter
        
        self.primary = primary
        self.secondary = secondary
        self._visitor = NoVisitor() if visitor is None else visitor

        (
            self.primary_adj_matrix,
            self.primary_i2n,
            self.primary_n2i,
        ) = self.extract_adjacency_matrix(primary)
        (
            self.secondary_adj_matrix,
            self.secondary_i2n,
            self.secondary_n2i,
        ) = self.extract_adjacency_matrix(secondary)
        self.sim_matrix = None

        self.mapping = None

    def _convert_mapping(self, mapping: RawMapping) -> Mapping:
        primary_idx, secondary_idx = mapping
        get_node = lambda idx, graph, map_i2n: graph[map_i2n[idx]]
        get_node_primary = lambda idx: get_node(idx, self.primary, self.primary_i2n)
        get_node_secondary = lambda idx: get_node(
            idx, self.secondary, self.secondary_i2n
        )

        # Get the matching nodes
        primary_matched = map(get_node_primary, primary_idx)
        secondary_matched = map(get_node_secondary, secondary_idx)

        # Get the unmatched nodes
        primary_unmatched = set(
            map(
                get_node_primary,
                np.setdiff1d(range(len(self.primary_adj_matrix)), primary_idx),
            )
        )
        secondary_unmatched = set(
            map(
                get_node_secondary,
                np.setdiff1d(range(len(self.secondary_adj_matrix)), secondary_idx),
            )
        )

        # Get the similiarity scores
        similarities = self.sim_matrix[primary_idx, secondary_idx]

        # Get the number of squares for each matching pair. We are counting both squares
        # in which the pair is a starting pair and the ones in which is a ending pair.
        #   (n1) <----> (n2) (starting pair)
        #    |           |
        #    v           v
        #   (n3) <----> (n4) (ending pair)
        common_subgraph = self.primary_adj_matrix[np.ix_(primary_idx, primary_idx)]
        common_subgraph &= self.secondary_adj_matrix[
            np.ix_(secondary_idx, secondary_idx)
        ]
        squares = common_subgraph.sum(0) + common_subgraph.sum(1)

        return Mapping(
            zip(primary_matched, secondary_matched, similarities, squares),
            primary_unmatched,
            secondary_unmatched,
        )

    def extract_adjacency_matrix(
        self, graph: Graph
    ) -> (AdjacencyMatrix, dict[int, Any], dict[Any, int]):
        """Returns the adjacency matrix for the graph and the mappings"""

        map_i2l = {}  # Map index to label
        map_l2i = {}  # Map label to index
        for i, node in enumerate(graph.node_labels):
            map_l2i[node] = i
            map_i2l[i] = node

        matrix = np.zeros((len(map_i2l), len(map_i2l)), bool)
        for node_a, node_b in graph.edges:
            matrix[map_l2i[node_a], map_l2i[node_b]] = True

        return (matrix, map_i2l, map_l2i)

    def register_feature_extractor(self, extractorClass: type, weight: Positive = 1.0):
        """
        Register a feature extractor class.
        The class will be called when the visitor will traverse the graph.
        """
        extractor = extractorClass(weight)
        self._visitor.register_feature_extractor(extractor)

    def compute_similarity(self) -> None:
        """
        Populate the self.sim_matrix similarity matrix by computing the pairwise
        similarity between the nodes of the two graphs to diff.
        """
        # Extract the features
        key_fun = lambda *args: args[0][0]  # ((label, node) iteration)
        primary_features = self._visitor.visit(self.primary, key_fun=key_fun)
        secondary_features = self._visitor.visit(self.secondary, key_fun=key_fun)

        # Get the weights of each feature
        f_weights = {}
        for extractor in self._visitor.feature_extractors:
            f_weights[extractor.key] = extractor.weight

        # Get all the keys and subkeys of the features
        # features_keys is a dict: {main_key: set(subkeys), ...}
        features_keys = {}
        for features in (primary_features, secondary_features):
            for f_collector in features.values():
                for main_key, subkey_list in f_collector.full_keys().items():
                    features_keys.setdefault(main_key, set())
                    if subkey_list:
                        features_keys[main_key].update(subkey_list)

        # Build the weights vector
        weights = []
        for main_key, subkey_list in features_keys.items():
            if subkey_list:
                dim = len(subkey_list)
                weights.extend(f_weights[main_key] / dim for _ in range(dim))
            else:
                weights.append(f_weights[main_key])

        # Build the feature matrix
        primary_feature_matrix = np.zeros(
            (len(primary_features), len(weights)), dtype=Differ.DTYPE
        )
        secondary_feature_matrix = np.zeros(
            (len(secondary_features), len(weights)), dtype=Differ.DTYPE
        )
        for node_label, feature in primary_features.items():
            node_index = self.primary_n2i[node_label]
            primary_feature_matrix[node_index] = feature.to_vector(features_keys)
        for node_label, feature in secondary_features.items():
            node_index = self.secondary_n2i[node_label]
            secondary_feature_matrix[node_index] = feature.to_vector(features_keys)

        # Generate the similarity matrix
        self.sim_matrix = scipy.spatial.distance.cdist(
            primary_feature_matrix, secondary_feature_matrix, self.distance, w=weights
        ).astype(Differ.DTYPE)

        # Normalize
        if self.sim_matrix.max() != 0:
            self.sim_matrix /= self.sim_matrix.max()
        self.sim_matrix[:] = 1 - self.sim_matrix

    def run_filters(self) -> None:
        """
        Custom filters that can edit the self.sim_matrix similarity matrix.
        This method is meant to be overriden by a sub-class.
        """
        pass

    def process(self) -> None:
        """Initialize all the variables for the NAP algorithm"""
        # Perform the initialization only once
        if self.sim_matrix is not None:
            return

        self.compute_similarity()
        self.run_filters()

    def compute_matching(
        self,
    ) -> Mapping:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is returned as a Mapping object.
        """
        for _ in self.matching_iterator():
            pass
        return self.mapping

    def matching_iterator(
        self,
    ) -> Generator[int]:
        """
        Run the belief propagation algorithm. This method returns a generator the yields
        the iteration number until the algorithm either converges or reaches `self.maxiter`
        """

        self.process()

        matcher = Matcher(
            self.sim_matrix, self.primary_adj_matrix, self.secondary_adj_matrix
        )
        matcher.process(self.sparsity_ratio)

        yield from matcher.compute(self.tradeoff, self.epsilon, self.maxiter)

        self.mapping = self._convert_mapping(matcher.mapping)


class DiGraphDiffer(Differ):
    """
    Differ implementation for two generic networkx.DiGraph
    """

    class DiGraphWrapper(GenericGraph):
        """A wrapper for DiGraph. It has no distinction between node labels and nodes"""

        def __init__(self, graph: DiGraph):
            self._graph = graph

        def items(self) -> Iterator[tuple[Any, Any]]:
            """Return an iterator over the items. Each item is {node_label: node} but since"""
            for node in self._graph.nodes:
                yield (node, node)

        @property
        def node_labels(self) -> Iterator[Any]:
            """Return an iterator over the node labels"""
            return self._graph.nodes

        @property
        def nodes(self) -> Iterator[Any]:
            """Return an iterator over the nodes"""
            return self._graph.nodes

        @property
        def edges(self) -> Iterator[tuple[Any, Any]]:
            """
            Return an iterator over the edges.
            An edge is a pair (node_label_a, node_label_b)
            """
            return self._graph.edges

    def __init__(self, primary: DiGraph, secondary: DiGraph):
        super(DiGraphDiffer, self).__init__(
            self.DiGraphWrapper(primary), self.DiGraphWrapper(secondary)
        )


class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    DTYPE = np.float32

    def __init__(self, primary: Program, secondary: Program):
        super(QBinDiff, self).__init__(primary, secondary, ProgramVisitor())

        # Aliases
        self.primary_f2i = self.primary_n2i
        self.primary_i2f = self.primary_i2n
        self.secondary_f2i = self.secondary_n2i
        self.secondary_i2f = self.secondary_i2n

    def match_import_functions(self) -> None:
        primary_import = {}
        for addr, func in self.primary.items():
            if func.is_import():
                primary_import[func.name] = addr
                self.sim_matrix[self.primary_f2i[addr]] = 0
        for addr, func in self.secondary.items():
            if func.is_import():
                s_idx = self.secondary_f2i[addr]
                self.sim_matrix[:, s_idx] = 0

                if func.name in primary_import:
                    p_idx = self.primary_f2i[primary_import[func.name]]
                    self.sim_matrix[p_idx, s_idx] = 1

    def run_filters(self) -> None:
        self.match_import_functions()

    def save(self, filename: str):
        with open(filename, "w") as file:
            json.dump(
                {
                    "matched": [(x[0].addr, x[1].addr) for x in self.mapping],
                    "unmatched": [
                        [x.addr for x in self.mapping.primary_unmatched],
                        [x.addr for x in self.mapping.secondary_unmatched],
                    ],
                },
                file,
                indent=2,
            )

    def load(self):
        pass

    def initialize_from_file(self, filename: PathLike):
        data = scipy.io.loadmat(str(filename))
        self.primary_affinity = data["A"].astype(bool)
        self.secondary_affinity = data["B"].astype(bool)
        self.sim_matrix = data["C"].astype(QBinDiff.DTYPE)
        # Initialize lookup dict Item -> idx
        self._make_indexes(
            range(len(self.primary_affinity)), range(len(self.secondary_affinity))
        )

    def diff_program(
        self,
        distance: str = "canberra",
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
        anchors: Anchors = None,
    ) -> Mapping:
        # Convert networkx callgraphs to numpy array
        primary_affinity = self._get_affinity_matrix(
            self.primary.callgraph, self.primary
        )
        secondary_affinity = self._get_affinity_matrix(
            self.secondary.callgraph, self.secondary
        )

        self.compute_similarity(
            self.primary,
            self.secondary,
            primary_affinity,
            secondary_affinity,
            self._visitor,
            distance,
        )
        if anchors:
            self.set_anchors(anchors)
        return self.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    def diff_function(
        self,
        primary: Function,
        secondary: Function,
        anchors: Anchors = None,
    ) -> Mapping:
        # Convert networkx callgraphs to numpy array
        primary_affinity = self._get_affinity_matrix(primary.flowgraph, primary)
        secondary_affinity = self._get_affinity_matrix(secondary.flowgraph, secondary)

        self.compute_similarity(
            primary,
            secondary,
            primary_affinity,
            secondary_affinity,
            self._visitor,
            distance,
        )
        if anchors:
            self.set_anchors(anchors)
        return self.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    def save_sqlite(self, filename: PathLike):
        self.mapping.save_sqlite(filename)
