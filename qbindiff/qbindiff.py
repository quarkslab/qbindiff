# coding: utf-8
import logging
import numpy as np

from qbindiff.loader.visitor import ProgramVisitor
from qbindiff.matcher.matcher import Matcher
from qbindiff.mapping.mapping import Mapping, AddressMapping

# Import for types
from qbindiff.types import Iterable, Tuple, List, Dict
from qbindiff.types import Optional, Any, Int, Float, Str
from qbindiff.types import PathLike, Positive, Ratio, Idx, Addr, Dtype
from qbindiff.types import FeatureVectors, AffinityMatrix, SimMatrix

from qbindiff.features.visitor import FeatureExtractor, Environment, Visitor
from qbindiff.loader.program import Program
from qbindiff.loader.function import Function


class Differ(object):

    def diff(self, primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix,
             visitor: Visitor, distance: Str='canberra', dtype: Dtype=np.float32, anchors: Anchors=None,
             sparsity_ratio: Ratio=.75, tradeoff: Ratio: .75, epsilon: Positive= .5, maxiter: Int=1000,
             output: PathLike=''):
        self.initialize(visitor, distance, dtype)
        self.compute(sparsity_ratio, tradeoff, epsilon, maxiter)
        self.save(output)

    def initialize(self, primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix,
             visitor: Visitor, distance: Str='canberra', dtype: Dtype=np.float32, anchors: Anchors=None):
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
        self.primary_affinity = primary_affinity
        self.secondary_affinity = secondary_affinity

        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_index, feature_weights = self._build_feature_index(primary_features, secondary_features, visitor)
        primary_matrix = self._vectorize_features(primary_features, feature_index, dtype)
        secondary_matrix = self._vectorize_features(secondary_features, feature_index, dtype)

        self.sim_matrix = self._compute_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        anchors = self._convert_anchors(primary, secondary, anchors)
        self._apply_anchors(self.sim_matrix, anchors)

    def compute(self, sparsity_ratio: Ratio=.75, tradeoff: Ratio=.75, epsilon: Positive=.5, maxiter: Int=1000):
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

        self.mapping = self._convert_mapping(matcher.mapping)

        logging.info(self.display_statistics())

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
    def _vectorize_features(iterable_features: List[Environment], feature_index: Dict, dtype: Dtype) -> FeatureVectors:
        feature_matrix = np.zerors((len(iterable_features), len(feature_index)), dtype=dtype)
        for idx, features in enumerate(iterable_features.values()):
            if features:  # if the node has features (otherwise array cells are already zeros)
                idy, value = zip(*((feature_index[key], value) for key, value in features.items()))
                feature_matrix[idx, idy] += value
        return feature_matrix

     @staticmethod
    def _compute_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: Str='canberra', dtype: Dtype=np.float32, weights: List[Positive]=1.0) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, dtype, weights)
        matrix /= matrix.max()
        return matrix

    @staticmethod
    def _compute_feature_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: Str, dtype: Dtype, weights: List[Positive]) -> SimMatrix:
        matrix = cdist(primary_matrix, secondary_matrix, distance, w=weights).astype(dtype)
        matrix /= matrix.max()
        matrix[:] = 1 - matrix
        return matrix

    @staticmethod
    def _compute_address_similarity(nb_primary_nodes: Int, nb_secondary_nodes: Int, dtype: Dtype) -> SimMatrix:
        matrix = np.zeros((nb_primary_nodes, nb_secondary_nodes), dtype)
        primary_idx = np.arange(nb_primary_nodes, dtype=dtype) / np.maximum(nb_primary_nodes, nb_secondary_nodes)
        secondary_idx = np.arange(nb_secondary_nodes, dtype=dtype) / np.maximum(nb_primary_nodes, nb_secondary_nodes)
        matrix = np.absolute(np.subtract.outer(primary_idx, secondary_idx))
        matrix[:] = 1 - matrix
        return matrix

    @staticmethod
    def _compute_constant_similarity(primary_constants: List[Tuple[Str]], secondary_constants: List[Tuple[Str]], weight: Positive=.5, dtype: Dtype) -> SimMatrix:
        matrix = np.zeros((len(primary_constants), len(secondary_constants)), dtype)
        for constant in set(primary_constants).intersection(secondary_constants):
            idx, idy = zip(*product(primary_constants[constant], secondary_constants[constant]))
            np.add.at(matrix, (idx, idy), weight / len(idx))
        return matrix

    @staticmethod
    def _convert_anchors(primary: Iterable, secondary: Iterable, anchors: Anchors) -> Anchors:
        return anchors

    @staticmethod
    def _apply_anchors(matrix:SimMatrix, anchors:Anchors):
        if anchors:
            idx, idy = anchors
            data = matrix[idx, idy]
            matrix[idx] = 0
            matrix[:, idy] = 0
            matrix[idx, idy] = data

    def _convert_mapping(self, mapping: RawMapping) -> Mapping:
        idx, idy = mapping
        similarities, squares = self._compute_statistics(mapping)
        return Mapping(idx, idy, similarities, squares)

    def _compute_statistics(self, mapping: RawMapping=None) -> Tuple[Vector, AffinityMatrix]:
        if mapping is None:
            mapping = self.mapping
        idx, idy = mapping
        similarities = self.sim_matrix[idx, idy]
        common_subgraph = self.primary_affinity[np.ix_(idx, idx)]
        common_subgraph &= self.secondary_affinity[np.ix_(idy, idy)]
        squares = common_subgraph.sum(0) + common_subgraph.sum(1)
        return similarities, squares

    def display_statistics(self, mapping: RawMapping=None) -> Str:
        similarities, squares = self._compute_statistics(mapping)
        nb_matches = len(similarities)
        similarity = similarities.sum()
        nb_squares = squares.sum()

        output = 'Score: {:.4f} | '\
                 'Similarity: {:.4f} | '\
                 'Squares: {:.0f} | '\
                 'Nb matches: {}\n'.format(similarity + nb_squares, similarity, nb_squares, nb_matches)
        output += 'Node cover:  {:.3f}% / {:.3f}% | '\
                  'Edge cover:  {:.3f}% / {:.3f}%\n'.format(100 * nb_matches / len(self.primary_affinity),
                                                            100 * nb_matches / len(self.secondary_affinity),
                                                            100 * nb_squares / self.primary_affinity.sum(),
                                                            100 * nb_squares / self.secondary_affinity.sum())
        return output


class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        self.primary = primary
        self.secondary = secondary
        self._visitor = ProgramVisitor()

    def diff_program(self, visitor: ProgramVisitor, distance: Str='canberra', dtype: Dtype=np.float32, anchors: AddrAnchors=None,
                     sparsity_ratio: Ratio=.75, tradeoff: Ratio: .75, epsilon: Positive= .5, maxiter: Int=1000,
                     output: PathLike=''):
        self.diff(self.primary, self.secondary, self.primary.callgraph, self.secondary.callgraph, self._visitor, distance, dtype, anchors, sparsity_ratio, tradeoff, epsilon, maxiter, output)

    def diff_function(self, primary: Function, secondary: Function,
                      visitor: ProgramVisitor, distance: Str='canberra', dtype: Dtype=np.float32, anchors: AddrAnchors=None,
                      sparsity_ratio: Ratio=.75, tradeoff: Ratio: .75, epsilon: Positive= .5, maxiter: Int=1000,
                      output: PathLike=''):
        self.diff(primary, secondary, primary.flowgraph, secondary.flowgraph, self._visitor, distance, dtype, anchors, sparsity_ratio, tradeoff, epsilon, maxiter, output)

    def save(self, filename: PathLike=''):
        filename = str(filename)
        if not filename:
            filename = str(self.diffing)
        if not filename.endswith('.qbindiff'):
            filename += '.qbindiff'
        self.mapping.save_sqlite(filename)

    def register_feature(feature: FeatureExtractor, weight: Positive=1.0):
        self._visitor.register_feature(feature, weight)

     @staticmethod
    def _compute_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: Str='canberra', dtype: Dtype=np.float32, feature_weights: List[Positive]=1.0) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        matrix += .01 * Differ._compute_address_similarity(len(primary_matrix), len(secondary_matrix), dtype)
        matrix /= matrix.max()
        return matrix

    @staticmethod
    def _convert_anchors(primary: Iterable, secondary: Iterable, anchors: AddrAnchors) -> Anchors:
        if anchors:
            primary_index = {item.addr: idx for idx, item in enumerate(primary)}
            secondary_index = {item.addr: idx for idx, item in enumerate(secondary)}
            addrx, addry = anchors
            return [primary_index[addr] for addr in addrx], [secondary_index[addr] for addr in addry]
        return anchors

    def _convert_mapping(self, mapping: RawMapping) -> Mapping:
        idx, idy = mapping
        similarities, squares = self._compute_statistics(mapping)
        return AddressMapping(idx, idy, similarities, squares)

