import logging
import numpy as np
from datasketch import MinHash
from networkx import DiGraph
from collections.abc import Generator, Iterator
from typing import Any, Callable, Optional

from qbindiff.abstract import GenericGraph
from qbindiff.loader import Program
from qbindiff.matcher import Matcher
from qbindiff.mapping import Mapping
from qbindiff.features.extractor import FeatureExtractor
from qbindiff.passes import FeaturePass, ZeroPass
from qbindiff.types import (
    RawMapping,
    Positive,
    Ratio,
    Graph,
    AdjacencyMatrix,
    SimMatrix,
)


class Differ:
    """
    Abstract class that perform the NAP diffing between two generic graphs.

    :param sparsity_ratio: the sparsity ratio enforced to the similarity matrix
    :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0)
                     and edge similarity (tradeoff=0.0)
    :param epsilon: perturbation parameter to enforce convergence and speed up computation.
                    The greatest the fastest, but least accurate
    :param maxiter: maximum number of message passing iterations
    :param sparse_row: Whether to build the sparse similarity matrix considering its
                       entirety or processing it row per row
    """

    DTYPE = np.float32

    def __init__(
        self,
        primary: Graph,
        secondary: Graph,
        *,
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
        normalize: bool = False,
        sparse_row: bool = False,
    ):

        # NAP parameters
        self.sparsity_ratio = sparsity_ratio
        self.tradeoff = tradeoff
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.sparse_row = sparse_row

        self.primary = primary
        self.secondary = secondary
        self._pre_passes = []
        self._post_passes = []
        self._already_processed = False  # Flag to perfom the processing only once

        if normalize:
            self.primary = self.normalize(primary)
            self.secondary = self.normalize(secondary)

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

        # Dimension of the graphs
        self.primary_dim = len(self.primary_i2n)
        self.secondary_dim = len(self.secondary_i2n)

        # Similarity matrix filled with -1 (unknown value)
        self.sim_matrix = np.full(
            (self.primary_dim, self.secondary_dim), -1, dtype=Differ.DTYPE
        )
        self.mapping = None

    def get_similarities(
        self, primary_idx: list[int], secondary_idx: list[int]
    ) -> list[float]:
        """
        Returns the similarity scores between the nodes specified as parameter.
        By default it uses the similarity matrix.
        This method is meant to be overridden by subclasses to give more meaningful
        scores
        """

        return self.sim_matrix[primary_idx, secondary_idx]

    def _convert_mapping(self, mapping: RawMapping, confidence: list[float]) -> Mapping:
        """
        Return the result of the diffing as a Mapping object.

        :param mapping: The raw mapping between the nodes
        :param confidence: The confidence score for each match
        """

        logging.debug("Wrapping raw diffing output in a Mapping object")
        primary_idx, secondary_idx = mapping
        get_node_primary = lambda idx: self.primary.get_node(self.primary_i2n[idx])
        get_node_secondary = lambda idx: self.secondary.get_node(
            self.secondary_i2n[idx]
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
        similarities = self.get_similarities(primary_idx, secondary_idx)

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
            zip(primary_matched, secondary_matched, similarities, confidence, squares),
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

    def register_prepass(self, pass_func: Callable, **extra_args):
        """
        Register a new pre-pass that will operate on the similarity matrix.
        The passes will be called in the same order as they are registered and each one
        of them will operate on the output of the previous one.
        WARNING: a prepass should assign values to the full row or the full column, it
        should never assign single entries in the matrix
        """
        self._pre_passes.append((pass_func, extra_args))

    def register_postpass(self, pass_func: Callable, **extra_args):
        """
        Register a new post-pass that will operate on the similarity matrix.
        The passes will be called in the same order as they are registered and each one
        of them will operate on the output of the previous one.
        """
        self._post_passes.append((pass_func, extra_args))

    def normalize(self, graph: Graph) -> Graph:
        """
        Custom function that normalizes the input graph.
        This method is meant to be overriden by a sub-class.
        """
        return graph

    def run_passes(self) -> None:
        """Run all the passes that have been previously registered"""

        for pass_func, extra_args in self._pre_passes:
            pass_func(
                self.sim_matrix,
                self.primary,
                self.secondary,
                self.primary_n2i,
                self.secondary_n2i,
                **extra_args,
            )
        for pass_func, extra_args in self._post_passes:
            pass_func(
                self.sim_matrix,
                self.primary,
                self.secondary,
                self.primary_n2i,
                self.secondary_n2i,
                **extra_args,
            )

    def process(self) -> None:
        """Initialize all the variables for the NAP algorithm"""
        # Perform the initialization only once
        if self._already_processed:
            return
        self._already_processed = True

        self.run_passes()  # User registered passes

    def compute_matching(self) -> Mapping:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is returned as a Mapping object.
        """
        for _ in self.matching_iterator():
            pass
        return self.mapping

    def matching_iterator(self) -> Generator[int]:
        """
        Run the belief propagation algorithm. This method returns a generator the yields
        the iteration number until the algorithm either converges or reaches `self.maxiter`
        """

        self.process()

        matcher = Matcher(
            self.sim_matrix, self.primary_adj_matrix, self.secondary_adj_matrix
        )
        matcher.process(self.sparsity_ratio, self.sparse_row)

        yield from matcher.compute(self.tradeoff, self.epsilon, self.maxiter)

        self.mapping = self._convert_mapping(matcher.mapping, matcher.confidence_score)


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

        def get_node(self, node_label: Any):
            """Returns the node identified by the `node_label`"""
            return node_label

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

    def __init__(self, primary: DiGraph, secondary: DiGraph, **kwargs):
        super(DiGraphDiffer, self).__init__(
            self.DiGraphWrapper(primary), self.DiGraphWrapper(secondary), **kwargs
        )

        self.register_prepass(self.gen_sim_matrix)

    def gen_sim_matrix(self, sim_matrix: SimMatrix, *args, **kwargs):
        sim_matrix[:] = 1


class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.

    :param distance: the distance function used when comparing the feature vector
                     extracted from the graphs
    """

    DTYPE = np.float32

    def __init__(
        self, primary: Program, secondary: Program, distance: str = "canberra", **kwargs
    ):
        super(QBinDiff, self).__init__(primary, secondary, **kwargs)

        # Aliases
        self.primary_f2i = self.primary_n2i
        self.primary_i2f = self.primary_i2n
        self.secondary_f2i = self.secondary_n2i
        self.secondary_i2f = self.secondary_i2n

        # Register the feature extraction pass
        self._feature_pass = FeaturePass(distance)
        self.register_postpass(self._feature_pass)
        self.register_postpass(ZeroPass)
        self.register_prepass(self.match_import_functions)

    def register_feature_extractor(
        self,
        extractorClass: type[FeatureExtractor],
        weight: Optional[Positive] = 1.0,
        distance: Optional[str] = None,
        **extra_args,
    ):
        """Register a feature extractor class"""
        extractor = extractorClass(weight, **extra_args)
        self._feature_pass.register_extractor(extractor, distance=distance)

    def match_import_functions(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict,
        secondary_mapping: dict,
    ) -> None:
        primary_import = {}
        for addr, func in primary.items():
            if func.is_import():
                primary_import[func.name] = addr
                sim_matrix[primary_mapping[addr]] = 0
        for addr, func in secondary.items():
            if func.is_import():
                s_idx = secondary_mapping[addr]
                sim_matrix[:, s_idx] = 0

                if func.name in primary_import:
                    p_idx = primary_mapping[primary_import[func.name]]
                    sim_matrix[p_idx, s_idx] = 1

    def normalize(self, program: Program) -> Program:
        """Normalize the input Program"""

        for addr, func in list(program.items()):
            if not func.is_thunk():
                continue
            # ~ print(addr, len(func.children), func.children)
            assert len(func.children) == 1, "Thunk function has multiple children"

            # Replace all the callers with the called function
            # { callers } --> thunk --> called
            import_func_addr = next(iter(func.children))
            program.follow_through(addr, import_func_addr)

        return program

    def get_similarities(
        self, primary_idx: list[int], secondary_idx: list[int]
    ) -> list[float]:
        """
        Returns the similarity scores between the nodes specified as parameter.
        Uses MinHash fuzzy hash at basic block level to give a similarity score.
        """

        # Utils functions
        get_func_primary = lambda idx: self.primary[self.primary_i2f[idx]]
        get_func_secondary = lambda idx: self.secondary[self.secondary_i2f[idx]]

        # Get the matching nodes
        primary_matched = map(get_func_primary, primary_idx)
        secondary_matched = map(get_func_secondary, secondary_idx)

        similarities = []
        for f1, f2 in zip(primary_matched, secondary_matched):
            h1, h2 = MinHash(), MinHash()
            for _, bb in f1.items():
                h1.update(b"".join(instr.mnemonic.encode("utf8") for instr in bb))
            for _, bb in f2.items():
                h2.update(b"".join(instr.mnemonic.encode("utf8") for instr in bb))
            similarities.append(h1.jaccard(h2))
        return similarities

    def export_to_bindiff(self, filename: str):
        from qbindiff.mapping.bindiff import BinDiffFormat

        bindiff = BinDiffFormat(filename, self.primary, self.secondary, self.mapping)
        bindiff.save()
