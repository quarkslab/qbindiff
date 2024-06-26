# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary and generic differs

This module contains the implementations of a generic differ (class Differ)
that can be subclassed for specific use cases.
It also contains the standard implementation for binary diffing (class QBinDiff)
and for directed graphs diffing (class DiGraphDiffing)
"""

from __future__ import annotations
import logging
import numpy as np
import networkx
from datasketch import MinHash  # type: ignore[import-untyped]
from collections.abc import Generator, Iterator
from typing import cast, TYPE_CHECKING

# third-party imports
from bindiff import BindiffFile  # type: ignore[import-untyped]

# local imports
# from qbindiff import __version__
from qbindiff.abstract import GenericGraph
from qbindiff.loader import Program, Function
from qbindiff.matcher import Matcher
from qbindiff.mapping import Mapping
from qbindiff.features.extractor import FeatureExtractor
from qbindiff.passes import FeaturePass, ZeroPass
from qbindiff.utils import is_debug, wrapper_iter
from qbindiff.exceptions import *
from qbindiff.types import RawMapping, Positive, Ratio, Graph, AdjacencyMatrix, Distance
from qbindiff.mapping.bindiff import export_to_bindiff

if TYPE_CHECKING:
    from typing import Any
    from qbindiff.types import (
        GenericPrePass,
        GenericPostPass,
        NodeLabel,
        SimMatrix,
        FeatureValue,
        Addr,
        Idx,
        ArrayLike1D,
    )
    from qbindiff.features.extractor import FeatureCollector


class Differ:
    DTYPE = np.float32

    def __init__(
        self,
        primary: Graph,
        secondary: Graph,
        *,
        sparsity_ratio: Ratio = 0.6,
        tradeoff: Ratio = 0.8,
        epsilon: Positive = 0.9,
        maxiter: int = 1000,
        sparse_row: bool = False,
    ):
        """
        Abstract class that perform the NAP diffing between two generic graphs.

        :param primary: primary graph
        :param secondary: secondary graph
        :param sparsity_ratio: the sparsity ratio enforced to the similarity matrix
            of type :py:class:`qbindiff.types.Ratio`
        :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0)
            and edge similarity (tradeoff=0.0) of type :py:class:`qbindiff.types.Ratio`
        :param epsilon: perturbation parameter to enforce convergence and speed up computation,
            of type :py:class:`qbindiff.types.Positive`. The greatest the fastest, but least accurate
        :param maxiter: maximum number of message passing iterations
        :param sparse_row: Whether to build the sparse similarity matrix considering its
            entirety or processing it row per row
        """

        # Checks for empty graph
        if len(primary) == 0:
            raise Exception("Primary graph does not contains any nodes")
        if len(secondary) == 0:
            raise Exception("Secondary graph does not contains any nodes")

        # NAP parameters
        self.sparsity_ratio = sparsity_ratio
        self.tradeoff = tradeoff
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.sparse_row = sparse_row

        #: Primary graph
        self.primary = primary
        #: Secondary graph
        self.secondary = secondary
        self._pre_passes: list = []
        self._already_processed: bool = False  # Flag to perform the processing only once

        self.primary_adj_matrix, self.primary_i2n, self.primary_n2i = self.extract_adjacency_matrix(
            primary
        )

        (
            self.secondary_adj_matrix,
            self.secondary_i2n,
            self.secondary_n2i,
        ) = self.extract_adjacency_matrix(secondary)

        # Dimension of the graphs
        self.primary_dim: int = len(self.primary_i2n)
        self.secondary_dim: int = len(self.secondary_i2n)

        # Similarity matrix filled with -1 (unknown value)
        self.sim_matrix: np.ndarray = np.full(
            (self.primary_dim, self.secondary_dim), -1, dtype=Differ.DTYPE
        )
        self.mapping: Mapping | None = None

    def get_similarities(self, primary_idx: list[Idx], secondary_idx: list[Idx]) -> ArrayLike1D:
        """
        Returns the similarity scores between the nodes specified as parameter.
        By default, it uses the similarity matrix.
        This method is meant to be overridden by subclasses to give more meaningful
        scores

        :param primary_idx: the List of integers that represent nodes inside the primary graph
        :param secondary_idx: the List of integers that represent nodes inside the primary graph
        :returns: A sequence with the corresponding similarities of the given nodes
        """
        return self.sim_matrix[primary_idx, secondary_idx]

    def _convert_mapping(self, mapping: RawMapping, confidence: list[float]) -> Mapping:
        """
        Return the result of the diffing as a Mapping object.

        :param mapping: The raw mapping between the nodes
        :param confidence: The confidence score for each match
        :return: mapping given a raw mapping with confidence scores
        """

        logging.debug("Wrapping raw diffing output in a Mapping object")
        primary_idx, secondary_idx = mapping
        get_node_primary = lambda idx: self.primary.get_node(self.primary_i2n[idx])
        get_node_secondary = lambda idx: self.secondary.get_node(self.secondary_i2n[idx])

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

        # Get the similarity scores
        similarities = self.get_similarities(primary_idx, secondary_idx)

        # Get the number of squares for each matching pair. We are counting both squares
        # in which the pair is a starting pair and the ones in which is a ending pair.
        #   (n1) <----> (n2) (starting pair)
        #    |           |
        #    v           v
        #   (n3) <----> (n4) (ending pair)
        common_subgraph = self.primary_adj_matrix[np.ix_(primary_idx, primary_idx)]
        common_subgraph &= self.secondary_adj_matrix[np.ix_(secondary_idx, secondary_idx)]
        squares = common_subgraph.sum(0) + common_subgraph.sum(1)

        return Mapping(
            zip(primary_matched, secondary_matched, similarities, confidence, squares),
            primary_unmatched,
            secondary_unmatched,
        )

    def extract_adjacency_matrix(
        self, graph: Graph
    ) -> tuple[AdjacencyMatrix, dict[Idx, NodeLabel], dict[NodeLabel, Idx]]:
        """
        Returns the adjacency matrix for the graph and the mappings

        :param graph: Graph whose adjacency matrix should be extracted
        :returns: A tuple containing in this order: the adjacency matrix of the
            graph, the map between index to label, the map between label to index.
        """

        map_i2l = {}  # Map index to label
        map_l2i = {}  # Map label to index
        # Node labels are node function addresses (in decimal)
        for i, node in enumerate(graph.node_labels):
            map_l2i[node] = i
            map_i2l[i] = node

        matrix = np.zeros((len(map_i2l), len(map_i2l)), bool)
        for node_a, node_b in graph.edges:
            matrix[map_l2i[node_a], map_l2i[node_b]] = True

        return matrix, map_i2l, map_l2i

    def register_prepass(self, pass_func: GenericPrePass, **extra_args) -> None:
        """
        Register a new pre-pass that will operate on the similarity matrix.
        The passes will be called in the same order as they are registered and each one
        of them will operate on the output of the previous one.
        .. warning:: A prepass should assign values to the full row or the full column, it
        should never assign single entries in the matrix

        :param pass_func: Pass method to apply on the similarity matrix. Example : a Pass that first matches import
            functions.
        """

        self._pre_passes.append((pass_func, extra_args))

    def _run_passes(self) -> Iterator[int]:
        """
        Run all the passes that have been previously registered.
        It returns an iterator that can be used for tracking the progress.

        :returns: An iterator of values in the range [0, 1000] used for tracking progress.
                  It might contain more than 1000 elements.
        """

        # If no passess have been registered, returns immediately 1000
        if len(self._pre_passes) == 0:
            yield 1000

        for pass_func, extra_args in self._pre_passes:
            for s in wrapper_iter(
                pass_func(
                    self.sim_matrix,
                    self.primary,
                    self.secondary,
                    self.primary_n2i,
                    self.secondary_n2i,
                    **extra_args,
                )
            ):
                # The single steps are within [0, 1000]. Normalize them.
                # Due to rounding errors, clip it to 1000
                yield min(1000, round(s / len(self._pre_passes)))

    def process_iterator(self) -> Iterator[int]:
        """
        Initialize all the variables for the NAP algorithm in an iterative way.
        It returns an iterator that can be used for tracking the progress.

        :returns: An iterator of values in the range [0, 1000] used for tracking progress.
                  It might contain more than 1000 elements.
        """

        # Perform the initialization only once
        if self._already_processed:
            return
        self._already_processed = True

        yield from self._run_passes()  # User registered passes

    def process(self) -> None:
        """
        Initialize all the variables for the NAP algorithm.
        """

        for _ in self.process_iterator():
            pass

    def compute_matching(self) -> Mapping | None:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is returned as a Mapping object.

        :return: Mapping between items of the primary and items of the secondary
        """

        for _ in self.matching_iterator():
            pass
        return self.mapping

    def matching_iterator(self) -> Generator[int, None, None]:
        """
        Run the belief propagation algorithm.

        :returns: A generator the yields the iteration number until the algorithm
                  either converges or reaches ``self.maxiter``
        """

        self.process()

        matcher = Matcher(self.sim_matrix, self.primary_adj_matrix, self.secondary_adj_matrix)
        matcher.process(self.sparsity_ratio, self.sparse_row)

        yield from matcher.compute(self.tradeoff, self.epsilon, self.maxiter)

        self.mapping = self._convert_mapping(matcher.mapping, matcher.confidence_score)


class GraphDiffer(Differ):
    """
    Differ implementation for two generic networkx.Graph instances (undirected graphs)
    """

    class GraphWrapper(GenericGraph):
        def __init__(self, graph: networkx.Graph):
            """
            A wrapper for networkx.Graph. It has no distinction between node labels and nodes
            """
            self._graph = graph

        def __len__(self) -> int:
            return len(self._graph)

        def items(self) -> Iterator[tuple[Addr, Any]]:
            """
            Return a iterator over the items. Each item is {node_label: node}
            """
            for node in self._graph.nodes:
                yield node, node

        def get_node(self, node_label: Any) -> Any:
            """
            Returns the node identified by the `node_label`
            """
            return node_label

        @property
        def node_labels(self) -> Iterator[Any]:
            """
            Return an iterator over the node labels
            """
            yield from self._graph.nodes

        @property
        def nodes(self) -> Iterator[Any]:
            """
            Return an iterator over the nodes
            """
            yield from self._graph.nodes

        @property
        def edges(self) -> Iterator[tuple[Any, Any]]:
            """
            Return an iterator over the edges.
            An edge is a pair (node_label_a, node_label_b)
            """
            yield from self._graph.edges

    def __init__(self, primary: networkx.Graph, secondary: networkx.Graph, **kwargs):
        super(GraphDiffer, self).__init__(
            self.GraphWrapper(primary), self.GraphWrapper(secondary), **kwargs
        )

        self.register_prepass(self.gen_sim_matrix)

    def gen_sim_matrix(self, sim_matrix: SimMatrix, *args, **kwargs) -> None:
        """
        Initialize the similarity matrix

        :param sim_matrix: The similarity matrix of type :py:class:`qbindiff.types.SimMatrix`
        """

        sim_matrix[:] = 1


class DiGraphDiffer(Differ):
    """
    Differ implementation for two generic networkx.DiGraph
    """

    class DiGraphWrapper(GenericGraph):
        def __init__(self, graph: networkx.DiGraph):
            """
            A wrapper for DiGraph. It has no distinction between node labels and nodes

            :param graph: Graph to initialize the differ
            """
            self._graph = graph

        def __len__(self) -> int:
            return len(self._graph)

        def items(self) -> Iterator[tuple[Addr, Any]]:
            """
            Return an iterator over the items. Each item is {node_label: node}
            """
            for node in self._graph.nodes:
                yield node, node

        def get_node(self, node_label: Any) -> Any:
            """
            Returns the node identified by the `node_label`
            """
            return node_label

        @property
        def node_labels(self) -> Iterator[Any]:
            """
            Return an iterator over the node labels
            """
            yield from self._graph.nodes

        @property
        def nodes(self) -> Iterator[Any]:
            """
            Return an iterator over the nodes
            """
            return self._graph.nodes

        @property
        def edges(self) -> Iterator[tuple[Any, Any]]:
            """
            Return an iterator over the edges.
            An edge is a pair (node_label_a, node_label_b)
            """
            yield from self._graph.edges

    def __init__(self, primary: networkx.DiGraph, secondary: networkx.DiGraph, **kwargs):
        super(DiGraphDiffer, self).__init__(
            self.DiGraphWrapper(primary), self.DiGraphWrapper(secondary), **kwargs
        )

        self.register_prepass(self.gen_sim_matrix)

    def gen_sim_matrix(self, sim_matrix: SimMatrix, *args, **kwargs) -> None:
        """
        Initialize the similarity matrix

        :param sim_matrix: The similarity matrix of type :py:class:`qbindiff.types.SimMatrix`
        :return: None
        """

        sim_matrix[:] = 1


class QBinDiff(Differ):
    DTYPE = np.float32

    def __init__(
        self,
        primary: Program,
        secondary: Program,
        distance: Distance = Distance.haussmann,
        normalize: bool = False,
        **kwargs,
    ):
        """
        QBinDiff class that provides a high-level interface to trigger a diff between two binaries.

        :param primary: The primary binary of type :py:class:`qbindiff.loader.Program`
        :param secondary: The secondary binary of type :py:class:`qbindiff.loader.Program`
        :param distance: the distance function used when comparing the feature vector
            extracted from the graphs.
        :param normalize: Normalize the two programs Call Graphs with a series of heuristics. Look at
            :py:meth:`normalize` for more information.
        """

        if normalize:  # Optional normalization step
            primary = self.normalize(primary)
            secondary = self.normalize(secondary)

        super().__init__(primary, secondary, **kwargs)

        # Type casting. Does nothing at runtime.
        self.primary = cast(Program, self.primary)  # type: Program
        self.secondary = cast(Program, self.secondary)  # type: Program

        # Aliases
        self.primary_f2i = self.primary_n2i
        self.primary_i2f = self.primary_i2n
        self.secondary_f2i = self.secondary_n2i
        self.secondary_i2f = self.secondary_i2n

        self._post_passes: list = []

        # Register the import function mapping and feature extraction pass
        self._feature_pass = FeaturePass(distance)
        self.register_postpass(ZeroPass)
        self.register_prepass(self.match_import_functions)

    def register_postpass(self, pass_func: GenericPostPass, **extra_args) -> None:
        """
        Register a new post-pass that will operate on the similarity matrix.
        The passes will be called in the same order as they are registered and each one
        of them will operate on the output of the previous one.

        :param pass_func: Pass method to apply on the similarity matrix. Example: a Pass that enforces
                          the matches considering certain features extracted.
        """

        self._post_passes.append((pass_func, extra_args))

    def register_feature_extractor(
        self,
        extractor_class: type[FeatureExtractor],
        weight: Positive | None = 1.0,
        distance: Distance | None = None,
        **extra_args,
    ) -> None:
        """
        Register a feature extractor class. This will include the corresponding feature in the similarity matrix
        computation

        :param extractor_class: A feature extractor of type :py:class:`qbindiff.features.extractor`
        :param weight: Weight associated to the corresponding feature. Default is 1.
        :param distance: Distance used only for this feature. It does not make sense to use it with bnb feature,
            but it can be useful for the WeisfeilerLehman feature.
        """

        extractor = extractor_class(weight, **extra_args)

        # Check that the capabilities required by the feature are supported by the Programs
        supported_cap = self.primary.capabilities & extractor.required_capabilities
        if supported_cap != extractor.required_capabilities:
            missing = supported_cap ^ extractor.required_capabilities
            raise UnsupportedFeatureException(
                f"The capability {missing.name}, required by the feature"
                f" {extractor.key} is not supported by the primary backend"
            )
        supported_cap = self.secondary.capabilities & extractor.required_capabilities
        if supported_cap != extractor.required_capabilities:
            missing = supported_cap ^ extractor.required_capabilities
            raise UnsupportedFeatureException(
                f"The capability {missing.name}, required by the feature"
                f" {extractor.key} is not supported by the secondary backend"
            )

        self._feature_pass.register_extractor(extractor, distance=distance)

    def _run_passes(self) -> Iterator[int]:
        """
        Run all the pre passes and post passes that have been previously registered.
        It returns an iterator that can be used for tracking the progress.

        :returns: An iterator of values in the range [0, 1000] used for tracking progress.
                  It might contain more than 1000 elements.
        """

        # Dirty hack to force the progress bar to display only the feature pass iterator
        for _ in super()._run_passes():
            pass

        # Call feature pass
        yield from self._feature_pass(
            self.sim_matrix, self.primary, self.secondary, self.primary_n2i, self.secondary_n2i
        )

        for pass_func, extra_args in self._post_passes:
            for _ in wrapper_iter(
                pass_func(
                    self.sim_matrix,
                    self.primary,
                    self.secondary,
                    self.primary_n2i,
                    self.secondary_n2i,
                    self._feature_pass.primary_features,
                    self._feature_pass.secondary_features,
                    **extra_args,
                )
            ):
                pass

        # Dirty hack to still provide a last step
        yield 1000

    def match_import_functions(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict[Addr, Idx],
        secondary_mapping: dict[Addr, Idx],
    ) -> None:
        """
        Anchoring phase. This phase considers import functions as anchors to the matching and set these functions
        similarity to 1. This anchoring phase is necessary to obtain a good match.

        :param sim_matrix: The similarity matrix of between the primary and secondary, of
            type :py:class:`qbindiff.types:SimMatrix`
        :param primary: The primary binary of type :py:class:`qbindiff.loader.Program`
        :param secondary: The secondary binary of type :py:class:`qbindiff.loader.Program`
        :param primary_mapping: Mapping between the primary function addresses and their corresponding index
        :param secondary_mapping: Mapping between the secondary function addresses and their corresponding index
        """

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
        """
        Normalize the input Program. In some cases, this can create an exception, caused by a thunk function.

        :param program: the program of type :py:class:`qbindiff.loader.Program` to normalize.
        :returns: the normalized program
        """

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

    def get_similarities(self, primary_idx: list[int], secondary_idx: list[int]) -> ArrayLike1D:
        """
        Returns the similarity scores between the nodes specified as parameter.
        Uses MinHash fuzzy hash at basic block level to give a similarity score.

        :param primary_idx: List of node indexes inside the primary
        :param secondary_idx: List of node indexes inside the secondary
        :return: A sequence with the corresponding similarities of the given nodes
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

    def export_to_bindiff(self, filename: str) -> None:
        """
        Exports diffing results inside the BinDiff format

        :param filename: Name of the output diffing file
        """

        if self.mapping is None:
            raise ValueError("No mapping yet. Compute the mapping first.")
        export_to_bindiff(filename, self.primary, self.secondary, self.mapping)
