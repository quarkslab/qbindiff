from __future__ import absolute_import
import logging

from qbindiff.matcher.matcher import Matcher
from qbindiff.mapping.mapping import Mapping

# Import for types
from typing import Optional, Iterable, Tuple, List, Dict, Any, Str, Int , Float
from qbindiff.types import PathLike, FeatureVector, AffinityMatrix, SimMatrix, Idx, Addr, Ratio
from qbindiff.loader.program import Program
from qbindiff.loader.function import Function
from qbindiff.features.visitor import FeatureExtractor, Environment, ProgramVisitor, Visitor
from qbindiff.mapping import Mapping, ProgramMapping, BasicBlockMapping, FunctionMapping



class Differ(object):

    def diff(self, primary: Iterable, secondary: Iterable,
             visitor: Visitor, distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Idx, Idx]=[],
             sparsity_ratio: Ratio=.0, tradeoff: Ratio: .75, epsilon: Float= .5, maxiter: Int=1000,
             filename: PathLike=''):
        self.initialize(visitor, distance, dtype, anchors)
        self.compute(sparsity_ratio, tradeoff, epsilon, maxiter)
        self.save(filename)

    def initialize(self, primary: Iterable, secondary: Iterable,
             visitor: Visitor, distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Idx, Idx]=[]):
        """
        Initialize the diffing instance by computing the pairwise similarity between the nodes
        of the two graphs to diff.
        :param primary: iterable to extract features from
        :param secondary: iterable to extract features from
        :param visitor: list of features extractors to apply
        :param distance: distance metric to use (will be later converted into a similarity metric)
        :param dtype: datatype of the similarity measures
        :param anchors: user defined mapping correspondences
        """
        self.diffing = '_vs_'.join((primary.name, secondary.name))
        self.primary_affinity = primary.graph
        self.secondary_affinity = secondary.graph

        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_index, feature_weights = self._build_feature_index(primary_features, secondary_features, visitor)
        primary_matrix = self._vectorize_features(primary_features, feature_index, dtype)
        secondary_matrix = self._vectorize_features(secondary_features, feature_index, dtype)

        self.sim_matrix = self._compute_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        self._apply_anchors(self.sim_matrix, anchors)
        # Todo build graph attribute for any iterable

    def compute(self, sparsity_ratio: Ratio=.9, tradeoff: Ratio=.75, epsilon: Float=.5, maxiter: Int=1000):
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then converted into a binary-based format.
        :param sparsity_ratio: ratio of most probable correspondences to consider during the matching
        :param tradeoff: tradeoff ratio bewteen node similarity (tradeoff=1.0) and edge similarity (tradeoff=0.0)
        :param epsilon: perturbation parameter to enforce convergence and speed up computation. The greatest the fastest, but least accurate
        :maxiter: maximum number of message passing iterations
        """
        matcher = Matcher(self.sim_matrix, self.primary_affinity, self.secondary_affinity)
        matcher.process(sparsity_ratio)

        yield from matcher.compute(tradeoff, epsilon, maxiter)

        logging.info(matcher.display_statistics())

        self.mapping = self._convert_mapping(matcher.format_mappping())

    def save(self, filename: PathLike=''):
        filename = str(filename)
        if not filename:
            filename = str(self.diffing)
        if not filename.endswith('.qbindiff'):
            filename += '.qbindiff'
        self.mapping.save(filename)

    def initialize_from_file(self, filename: PathLike):
        data = scipy.io.loadmat(str(filename))
        self.primary_affinity = data['A'].astype(bool)
        self.secondary_affinity = data['B'].astype(bool)
        self.sim_matrix = data['C'].astype(dtype)
        self.diffing = Path(filename).name

    @staticmethod
    def _build_feature_index(primary_features: List[Environment], secondary_features: List[Environment], visitor: Visitor) -> Tuple[Dict, List]:
        feature_keys = dict()
        for program_features in (primary_features, secondary_features):
            for function_features in program_features:
                for key, value in function_features.items():
                    if isinstance(value: list):
                        feature_keys.update(dict.from_keys(value.keys(), visitor[key].weight))
                    else:
                        feature_keys[key] = visitor[key].weight
        feature_index = {key: idx for idx, key in enumerate(feature_keys)}
        feature_weights = {feature_index[key]: weight for key, weight in feature_keys.items()}
        feature_weights = [weight for _, weight in  sorted(feature_weights.items())]
        return feature_index, feature_weights

    @staticmethod
    def _vectorize_features(iterable_features: List[Environment], feature_index: Dict, dtype: np.dtype) -> FeatureVector:
        feature_matrix = np.zerors((len(iterable_features), len(feature_index)), dtype=dtype)
        for idx, features in enumerate(iterable_features.values()):
            if features:  # if the node has features (otherwise array cells are already zeros)
                idy, value = zip(*((feature_index[key], value) for key, value in features.items()))
                feature_matrix[idx, ft_idx] += value
        return feature_matrix

     @staticmethod
    def _compute_similarity(primary_matrix: FeatureVector, secondary_matrix: FeatureVector, distance: Str='canberra', dtype: np.dtype=np.float32, feature_weights: List[Float]=1) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        matrix /= matrix.max()
        return matrix

    @staticmethod
    def _compute_feature_similarity(primary_matrix: FeatureVector, secondary_matrix: FeatureVector, distance: Str, dtype: np.dtype, feature_weights: List[Float]) -> SimMatrix:
        matrix = cdist(primary_matrix, secondary_matrix, distance, w=feature_weights).astype(dtype)
        matrix /= matrix.max()
        matrix[:] = 1 - matrix
        return matrix

    @staticmethod
    def _compute_address_similarity(nb_primary_nodes: Int, nb_secondary_nodes: Int, dtype: np.dtype) -> SimMatrix:
        matrix = np.zeros((nb_primary_nodes, nb_secondary_nodes), dtype)
        primary_idx = np.arange(nb_primary_nodes, dtype=dtype) / np.maximum(nb_primary_nodes, nb_secondary_nodes)
        secondary_idx = np.arange(nb_secondary_nodes, dtype=dtype) / np.maximum(nb_primary_nodes, nb_secondary_nodes)
        matrix = np.absolute(np.subtract.outer(primary_idx, secondary_idx))
        matrix[:] = 1 - matrix
        return matrix

    @staticmethod
    def _compute_constant_similarity(primary_constants: List[Tuple[Str]], secondary_constants: List[Tuple[Str]], weight=.5, dtype: np.dtype) -> SimMatrix:
        matrix = np.zeros((len(primary_constants), len(secondary_constants)), dtype)
        for constant in set(primary_constants).intersection(secondary_constants):
            idx, idy = zip(*product(primary_constants[constant], secondary_constants[constant]))
            np.add.at(matrix, (idx, idy), weight / len(idx))
        return matrix

    @staticmethod
    def _apply_anchors(matrix:SimMatrix, anchors:Tuple[Idx, Idx]) -> SimMatrix:
        if anchors:
            idx, idy = anchors
            data = matrix[idx, idy]
            matrix[idx] = 0
            matrix[:, idy] = 0
            matrix[idx, idy] = data

    @staticmethod
    def _convert_mapping(matcher_mapping):
        return Mapping(matcher_mapping)


class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        self.primary = primary
        self.secondary = secondary

    def diff_program(self, features: List[FeatureExtractor], distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Addr, Addr]=[],
                     sparsity_ratio: Ratio=.0, tradeoff: Ratio: .75, epsilon: Float= .5, maxiter: Int=1000,
                     filename: PathLike=''):
        visitor = self._build_visitor(features)
        anchors = self.extract_function_anchors(self.primary, self.secondary, anchors)
        self.diff(self.primary, self.secondary, visitor, distance, dtype, anchors, sparsity_ratio, tradeoff, epsilon, maxiter, filename)

    def diff_function(self, primary: Function, secondary: Function,
                      features: List[FeatureExtractor], distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Addr, Addr]=[],
                      sparsity_ratio: Ratio=.0, tradeoff: Ratio: .75, epsilon: Float= .5, maxiter: Int=1000,
                      filename: PathLike=''):
        visitor = self._build_visitor(features)
        anchors = self.extract_bblock_anchors(primary, secondary, anchors)
        mapping = self.diff(primary, secondary, visitor, distance, dtype, anchors, sparsity_ratio, tradeoff, epsilon, maxiter, filename)

    def save(self, filename: PathLike=''):
        filename = str(filename)
        if not filename:
            filename = str(self.diffing)
        if not filename.endswith('.qbindiff'):
            filename += '.qbindiff'
        self.mapping.save_sqlite(filename)

    @staticmethod
    def _build_visitor(features: List[FeatureExtractor]): -> ProgramVisitor
        visitor = ProgramVisitor()
        for feature, weight in zip(features, weights):
            visitor.register_feature(feature, weight)
        #Todo: deal with weights
        return visitor

    @staticmethod
    def extract_function_anchors(primary: Program, secondary: Program, anchors:Tuple[Addr, Addr]=[]) -> Tuple[Idx, Idx]:
        if anchors:
            primary_index = {function.addr: idx for idx, function in enumerate(self.primary)}
            secondary_index = {function.addr: idx for idx, function in enumerate(self.secondary)}
            addrx, addry = anchors
            return [primary_index[addr] for addr in addrx], [secondary_index[addr] for addr in addry]
        primary_names = {function.name: idx for idx, function in enumerate(self.primary)}
        secondary_names = {function.name: idx for idx, function in enumerate(self.secondary)}
        anchors = []
        for name in set(primary_names).intersection(secondary_names):
            anchors.append((primary_names[name], secondary_names[name]))
        return zip(*anchors)

    @staticmethod
    def extract_bblock_anchors(primary: Function, secondary: Function, anchors:Tuple[Addr, Addr]=[]) -> Tuple[Idx, Idx]:
        raise NotImplementedError('Not Implemented')

     @staticmethod
    def _compute_similarity(primary_matrix: FeatureVector, secondary_matrix: FeatureVector, distance: Str='canberra', dtype: np.dtype=np.float32, feature_weights: List[Float]=1) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        matrix += .01 * Differ._compute_address_similarity(len(primary_matrix), len(secondary_matrix), dtype)
        matrix /= matrix.max()
        return matrix

    @staticmethod
    def _convert_mapping(matcher_mapping):
        return FunctionMapping(matcher_mapping)



