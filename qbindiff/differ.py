# third-party imports
import numpy as np
import scipy.io
import scipy.spatial.distance
from collections import defaultdict
from networkx import DiGraph

from qbindiff.loader import Program, Function
from qbindiff.matcher import Matcher
from qbindiff.mapping import Mapping
from qbindiff.features.visitor import Environment, Visitor, ProgramVisitor, Feature
from qbindiff.types import Generator, Union, Iterable, Tuple, List, Dict, Anchors, Item, Idx
from qbindiff.types import Optional, Any, RawMapping
from qbindiff.types import PathLike, Positive, Ratio, Addr, Dtype
from qbindiff.types import FeatureVectors, AffinityMatrix, SimMatrix


class Differ(object):

    DTYPE = np.float32

    def __init__(self):
        # All fields are computed dynamically
        self._primary_items_to_idx = None
        self._secondary_items_to_idx = None
        self.primary_affinity = None  # initialized in compute_similarity
        self.secondary_affinity = None
        self.sim_matrix = None
        self.mapping = None

    @staticmethod
    def diff(primary: Iterable,
             secondary: Iterable,
             primary_affinity: AffinityMatrix,
             secondary_affinity: AffinityMatrix,
             visitor: Visitor,
             distance: str='canberra',
             anchors: Anchors=None,
             sparsity_ratio: Ratio=.75,
             tradeoff: Ratio=.75,
             epsilon: Positive=.5,
             maxiter: int=1000) -> Mapping:
        differ = Differ()
        differ.compute_similarity(primary, secondary, primary_affinity, secondary_affinity, visitor, distance)
        if anchors:
            differ.set_anchors(anchors)
        return differ.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    def compute_similarity(self, primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix,
             visitor: Visitor, distance: str='canberra'):
        """
        Initialize the diffing instance by computing the pairwise similarity between the nodes
        of the two graphs to diff.
        :param primary: iterable to extract features from
        :param secondary: iterable to extract features from
        :param primary_affinity: primary affinity matrix
        :param secondary_affinity: secondary affinity matrix
        :param visitor: list of features extractors to apply
        :param distance: distance metric to use (will be later converted into a similarity metric)
        :param anchors: user defined mapping correspondences
        """
        # Compute lookup and reverse lookup tables
        # TODO: Creating a BasicBlock object / items_to_idx as list
        self._primary_items_to_idx = {x: i for i, x in enumerate(primary)}
        self._secondary_items_to_idx = {x: i for i, x in enumerate(secondary)}

        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_keys = self._extract_feature_keys(primary_features, secondary_features)
        primary_matrix = self._vectorize_features(primary_features, feature_index)
        secondary_matrix = self._vectorize_features(secondary_features, feature_index)

        self.sim_matrix = self._compute_similarity(primary_matrix, secondary_matrix, distance)
        self.primary_affinity = primary_affinity
        self.secondary_affinity = secondary_affinity

    def set_anchors(self, anchors: Anchors) -> None:
        idx = [self._primary_items_to_idx[x[0]] for x in anchors]  # Convert items to indexes
        idy = [self._secondary_items_to_idx[x[1]] for x in anchors]
        data = self.sim_matrix[idx, idy]
        self.sim_matrix[idx] = 0
        self.sim_matrix[:, idy] = 0
        self.sim_matrix[idx, idy] = data

    def compute_matching(self, sparsity_ratio: Ratio=.75, tradeoff: Ratio=.75, epsilon: Positive=.5, maxiter: int=1000) -> Mapping:
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

    def matching_iterator(self, sparsity_ratio: Ratio=.75, tradeoff: Ratio=.75, epsilon: Positive=.5, maxiter: int=1000) -> Generator[int, None, None]:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then converted into a binary-based format.
        :param sparsity_ratio: ratio of most probable correspondences to consider during the matching
        :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0) and edge similarity (tradeoff=0.0)
        :param epsilon: perturbation parameter to enforce convergence and speed up computation. The greatest the fastest, but least accurate
        :param maxiter: maximum number of message passing iterations
        """
        matcher = Matcher(self.sim_matrix, self.primary_affinity, self.secondary_affinity)
        matcher.process(sparsity_ratio)

        yield from matcher.compute(tradeoff, epsilon, maxiter)

        self.mapping = self._convert_mapping(matcher.mapping)

    def save(self, filename: str):
        self.mapping.save(filename)

    def initialize_from_file(self, filename: PathLike):
        data = scipy.io.loadmat(str(filename))
        self.primary_affinity = data['A'].astype(bool)
        self.secondary_affinity = data['B'].astype(bool)

        # Initialize lookup dict Item -> idx
        self._primary_items_to_idx = {i: i for i in range(len(self.primary_affinity))}
        self._secondary_items_to_idx = {i: i for i in range(len(self.secondary_affinity))}

        self.sim_matrix = data['C'].astype(Differ.DTYPE)

    @staticmethod
    def _extract_feature_keys(primary_features: List[Environment], secondary_features: List[Environment]) -> List[str]:
        feature_keys = set()
        for features in (primary_features, secondary_features):
            for env in features:
                for key, value in env.features.items():
                    if isinstance(value, dict):
                        feature_keys.update(value.keys())
                    else:
                        feature_keys.update(key)
        return sorted(feature_keys)

    @staticmethod
    def _vectorize_features(features: List[Environment], feature_keys: List[str]) -> FeatureVectors:
        feature_index = {key: idx for idx, key in enumerate(feature_keys)}
        feature_matrix = np.zeros((len(features), len(feature_index)), dtype=Differ.DTYPE)
        for idx, env in enumerate(features):
            for key, value in env.features.items():
                if isinstance(value, dict):
                    idy, value = zip(*((feature_index[key], value) for key, value in value.items()))
                    feature_matrix[idx, list(idy)] 
                else:
                    feature_matrix[idx, feature_index[key]] = value
        return feature_matrix

    @staticmethod
    def _compute_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: str='canberra', weights: List[Positive]=1.0) -> SimMatrix:
        matrix = scipy.spatial.distance.cdist(primary_matrix, secondary_matrix, distance, w=weights).astype(Differ.DTYPE)
        matrix /= matrix.max()
        matrix[:] = 1 - matrix
        return matrix

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
        primary_idx_to_item = {v: k for k, v in self._primary_items_to_idx.items()}
        secondary_idx_to_item = {v: k for k, v in self._secondary_items_to_idx.items()}

        items_x, items_y = [primary_idx_to_item[x] for x in idx], [secondary_idx_to_item[x] for x in idy]
        primary_unmatched = self._primary_items_to_idx.keys() - items_x  # All items mines ones that have been matched
        secondary_unmatched = self._secondary_items_to_idx.keys() - items_y
        return Mapping(list(zip(items_x, items_y, similarities, squares)), (primary_unmatched, secondary_unmatched))


class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        super(QBinDiff, self).__init__()
        self.primary = primary
        self.secondary = secondary
        self._visitor = ProgramVisitor()

    def diff_program(self, distance: str='canberra',
                           sparsity_ratio: Ratio=.75,
                           tradeoff: Ratio= .75,
                           epsilon: Positive= .5,
                           maxiter: int=1000,
                           anchors: Anchors=None) -> Mapping:
        # Convert networkx callgraphs to numpy array
        primary_affinity = self._get_affinity_matrix(self.primary.callgraph, self.primary)
        secondary_affinity = self._get_affinity_matrix(self.secondary.callgraph, self.secondary)

        self.compute_similarity(self.primary, self.secondary, primary_affinity, secondary_affinity, self._visitor, distance)
        if anchors:
            self.set_anchors(anchors)
        return self.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    @staticmethod
    def _get_affinity_matrix(graph: DiGraph, items: Iterable):
        tmp = {v.addr: i for i, v in enumerate(items)}
        affinity_matrix = np.zeros((len(tmp), len(tmp)), dtype=bool)
        for src, dst in graph.edges:
            affinity_matrix[tmp[src], tmp[dst]] += 1

    def diff_function(self, primary: Function,
                            secondary: Function,
                            distance: str='canberra',
                            sparsity_ratio: Ratio=.75,
                            tradeoff: Ratio= .75,
                            epsilon: Positive= .5,
                            maxiter: int=1000,
                            anchors: Anchors=None) -> Mapping:
        # Convert networkx callgraphs to numpy array
        primary_affinity = self._get_affinity_matrix(primary.flowgraph, primary)
        secondary_affinity = self._get_affinity_matrix(secondary.flowgraph, secondary)

        self.compute_similarity(primary, secondary, primary_affinity, secondary_affinity, self._visitor, distance)
        if anchors:
            self.set_anchors(anchors)
        return self.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter)

    def save_sqlite(self, filename: PathLike):
        self.mapping.save_sqlite(filename)

    def register_feature(self, feature: Union[type, Feature], weight: Positive = 1.0):
        if not isinstance(feature, Feature):
            feature = feature(weight)
        self._visitor.register_feature(feature, weight)

    def compute_similarity(self, primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix,
             visitor: Visitor, distance: str='canberra'):
        # WARNING: C'est quoi la diff ?
        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_keys, feature_weights = self._extract_feature_keys(primary_features, secondary_features)
        primary_matrix = self._vectorize_features(primary_features, feature_keys)
        secondary_matrix = self._vectorize_features(secondary_features, feature_keys)

        self.sim_matrix = self._compute_similarity(primary_matrix, secondary_matrix, distance, feature_weights)
        self.primary_affinity = primary_affinity
        self.secondary_affinity = secondary_affinity

    def _extract_feature_keys(self, primary_features: List[Environment], secondary_features: List[Environment]) -> Tuple[List[str], List[float]]:
        feature_keys = defaultdict(set)
        for program_features in (primary_features, secondary_features):
            for function_features in program_features:
                for key, values in function_features.features.items():
                    if isinstance(values, dict):
                        feature_keys[key].update(values.keys())
                    else:
                        feature_keys[key].update(key)
        
        features_weights = dict()
        for key, keys in feature_keys.items():
            for k in keys:
                features_weights[k] = self._visitor.get_feature(key).weight / len(keys)

        feature_keys = sorted([key for keys in feature_keys.values() for key in keys])  # [Mnemonic, NbChild .. ]
        feature_weights = [features_weights[key] for key in feature_keys]
        return feature_keys, feature_weights