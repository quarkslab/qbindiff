# third-party imports
import numpy as np
import scipy.io
import json
import scipy.spatial.distance
from collections import defaultdict
from networkx import DiGraph

from qbindiff.loader import Program, Function
from qbindiff.matcher import Matcher
from qbindiff.mapping import Mapping
from qbindiff.features.visitor import (
    FeatureCollector,
    Visitor,
    ProgramVisitor,
    FeatureExtractor,
)
from qbindiff.types import (
    Generator,
    Union,
    Iterable,
    Tuple,
    List,
    Dict,
    Anchors,
    Item,
    Idx,
)
from qbindiff.types import Optional, Any, RawMapping
from qbindiff.types import PathLike, Positive, Ratio, Addr, Dtype
from qbindiff.types import FeatureVectors, AffinityMatrix, SimMatrix


class QBinDiff:
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    DTYPE = np.float32

    def __init__(self, primary: Program, secondary: Program):
        self.primary = primary
        self.secondary = secondary
        self._visitor = ProgramVisitor()
        self.sim_matrix = None
        self.mapping = None

        self.primary_f2i = {}  # function address to index in sim_matrix
        self.primary_i2f = {}  # sim_matrix index to function address
        self.secondary_f2i = {}
        self.secondary_i2f = {}

    @staticmethod
    def diff(
        primary: Iterable,
        secondary: Iterable,
        primary_affinity: AffinityMatrix,
        secondary_affinity: AffinityMatrix,
        visitor: Visitor,
        distance: str = "canberra",
        anchors: Anchors = None,
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
    ) -> Mapping:
        differ = QBinDiff()
        differ.compute_similarity(
            primary, secondary, primary_affinity, secondary_affinity, visitor, distance
        )
        if anchors:
            differ.set_anchors(anchors)
        return differ.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    def compute_similarity(self, distance: str = "canberra") -> None:
        """
        Initialize the diffing instance by computing the pairwise similarity between the
        nodes of the two graphs to diff.

        :param distance: distance metric to use (will be later converted into a similarity metric)
        """
        # Extract the features
        primary_features = self._visitor.visit(self.primary)
        secondary_features = self._visitor.visit(self.secondary)

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
            (len(primary_features), len(weights)), dtype=QBinDiff.DTYPE
        )
        secondary_feature_matrix = np.zeros(
            (len(secondary_features), len(weights)), dtype=QBinDiff.DTYPE
        )
        for (i, (func_addr, feature)) in enumerate(primary_features.items()):
            self.primary_f2i[func_addr] = i
            self.primary_i2f[i] = func_addr
            primary_feature_matrix[i] = feature.to_vector(features_keys)
        for (i, (func_addr, feature)) in enumerate(secondary_features.items()):
            self.secondary_f2i[func_addr] = i
            self.secondary_i2f[i] = func_addr
            secondary_feature_matrix[i] = feature.to_vector(features_keys)

        # Generate the similarity matrix
        self.sim_matrix = scipy.spatial.distance.cdist(
            primary_feature_matrix, secondary_feature_matrix, distance, w=weights
        ).astype(QBinDiff.DTYPE)
        self.sim_matrix /= self.sim_matrix.max()
        self.sim_matrix[:] = 1 - self.sim_matrix

    def compute_matching(
        self,
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
    ) -> Mapping:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then converted into a binary-based format.
        :param sparsity_ratio: ratio of most probable correspondences to consider during the matching
        :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0) and edge similarity (tradeoff=0.0)
        :param epsilon: perturbation parameter to enforce convergence and speed up computation. The greatest the fastest, but least accurate
        :param maxiter: maximum number of message passing iterations
        """
        for _ in self.matching_iterator(sparsity_ratio, tradeoff, epsilon, maxiter):
            pass
        return self.mapping

    def matching_iterator(
        self,
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
    ) -> Generator[int, None, None]:
        matcher = Matcher(
            self.sim_matrix, self.primary_affinity, self.secondary_affinity
        )
        matcher.process(sparsity_ratio)

        yield from matcher.compute(tradeoff, epsilon, maxiter)

        self.mapping = self._convert_mapping(matcher.mapping)

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

    def _extract_feature_keys(
        self,
        primary_envs: List[FeatureCollector],
        secondary_envs: List[FeatureCollector],
    ) -> Tuple[List[str], List[float]]:
        feature_keys = defaultdict(set)
        for envs in (primary_envs, secondary_envs):
            for env in envs:
                for key, values in env.features.items():
                    if isinstance(values, dict):
                        feature_keys[key].update(values.keys())
                    else:
                        feature_keys[key].add(key)

        features_weights = dict()
        for ft_key, ft_sub_keys in feature_keys.items():
            for k in ft_sub_keys:
                features_weights[k] = self._visitor.feature_weight(ft_key) / len(
                    ft_sub_keys
                )

        feature_keys = sorted(
            {key for keys in feature_keys.values() for key in keys}
        )  # [Mnemonic, NbChild .. ]
        feature_weights = [features_weights[key] for key in feature_keys]
        return feature_keys, feature_weights

    @staticmethod
    def _vectorize_features(
        features: List[FeatureCollector], feature_keys: List[str]
    ) -> FeatureVectors:
        feature_index = {key: idx for idx, key in enumerate(feature_keys)}
        feature_matrix = np.zeros(
            (len(features), len(feature_index)), dtype=QBinDiff.DTYPE
        )
        for idx, env in enumerate(features):
            for key, value in env.features.items():
                if isinstance(value, dict):
                    idy, value = zip(
                        *((feature_index[key], value) for key, value in value.items())
                    )
                    feature_matrix[idx, list(idy)] = value
                else:
                    feature_matrix[idx, feature_index[key]] = value
        return feature_matrix

    def _compute_statistics(self, mapping: RawMapping) -> Tuple[List[float], List[int]]:
        idx, idy = mapping
        similarities = self.sim_matrix[idx, idy]
        common_subgraph = self.primary_affinity[np.ix_(idx, idx)]
        common_subgraph &= self.secondary_affinity[np.ix_(idy, idy)]
        squares = common_subgraph.sum(0) + common_subgraph.sum(1)
        return similarities, squares

    def _convert_mapping(self, mapping: RawMapping) -> Mapping:
        idx, idy = mapping
        similarities, squares = self._compute_statistics(mapping)

        # Get reverse indexes: idx->obj
        primary_idx_to_item, secondary_idx_to_item = self.__rev_indexes()

        # Retrieve matched items from indexes
        primary_matched, secondary_matched = [primary_idx_to_item[x] for x in idx], [
            secondary_idx_to_item[x] for x in idy
        ]

        # Retrieve unmatched by doing some substractions
        if isinstance(self._primary_items_to_idx, dict):  # just do set substraction
            primary_unmatched = (
                self._primary_items_to_idx.keys() - primary_matched
            )  # All items mines ones that have been matched
            secondary_unmatched = (
                self._secondary_items_to_idx.keys() - secondary_matched
            )
        else:  # plays with list indexes to retrieve unmatched
            primary_unmatched = [
                primary_idx_to_item[i]
                for i in range(len(self._primary_items_to_idx))
                if i not in idx
            ]
            secondary_unmatched = [
                secondary_idx_to_item[i]
                for i in range(len(self._secondary_items_to_idx))
                if i not in idx
            ]

        return Mapping(
            list(zip(primary_matched, secondary_matched, similarities, squares)),
            (primary_unmatched, secondary_unmatched),
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

    @staticmethod
    def _get_affinity_matrix(graph: DiGraph, items: Iterable):
        item_index = {item.addr: idx for idx, item in enumerate(items)}
        affinity_matrix = np.zeros((len(item_index), len(item_index)), dtype=bool)
        for src, dst in graph.edges:
            affinity_matrix[item_index[src], item_index[dst]] += 1
        return affinity_matrix

    def diff_function(
        self,
        primary: Function,
        secondary: Function,
        distance: str = "canberra",
        sparsity_ratio: Ratio = 0.75,
        tradeoff: Ratio = 0.75,
        epsilon: Positive = 0.5,
        maxiter: int = 1000,
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

    def register_feature_extractor(self, extractorClass: type, weight: Positive = 1.0):
        extractor = extractorClass(weight)
        self._visitor.register_feature_extractor(extractor)
