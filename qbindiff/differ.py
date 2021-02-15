# coding: utf-8
import logging
from pathlib import Path

# third-party imports
import numpy as np
import scipy.io
import scipy.spatial.distance
from typing import Generator

from qbindiff.features.visitor import ProgramVisitor, Feature
from qbindiff.matcher.matcher import Matcher
from qbindiff.mapping.mapping import Mapping, AddressMapping

# Import for types
from qbindiff.types import Iterable, Tuple, List, Dict, Anchors
from qbindiff.types import Optional, Any, RawMapping, AddrAnchors
from qbindiff.types import PathLike, Positive, Ratio, Idx, Addr, Dtype
from qbindiff.types import FeatureVectors, AffinityMatrix, SimMatrix

from qbindiff.features.visitor import Environment, Visitor
from qbindiff.loader.program import Program
from qbindiff.loader.function import Function



class Differ(object):

    DTYPE = np.float32

    def __init__(self):
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
        for _ in differ.compute_matching(sparsity_ratio, tradeoff, epsilon, maxiter):
            pass
        return differ.mapping

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
        self.primary_affinity = primary_affinity
        self.secondary_affinity = secondary_affinity

        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_index, feature_weights = self._build_feature_index(primary_features, secondary_features, visitor)
        primary_matrix = self._vectorize_features(primary_features, feature_index)
        secondary_matrix = self._vectorize_features(secondary_features, feature_index)

        self.sim_matrix = self._compute_similarity(primary_matrix, secondary_matrix, distance, feature_weights)


    def set_anchors(self, anchors: Anchors) -> None:
        idx, idy = zip(*anchors)
        data = self.sim_matrix[idx, idy]
        self.sim_matrix[idx] = 0
        self.sim_matrix[:, idy] = 0
        self.sim_matrix[idx, idy] = data

    def compute_matching(self, sparsity_ratio: Ratio=.75, tradeoff: Ratio=.75, epsilon: Positive=.5, maxiter: int=1000) -> Generator[int, None, None]:
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
        self.sim_matrix = data['C'].astype(Differ.DTYPE)

    @staticmethod
    def _build_feature_index(primary_features: List[Environment], secondary_features: List[Environment], visitor: ProgramVisitor) -> Tuple[List, List]:
        feature_keys = dict()

        # TODO: Pour les features dict => merger les items pour avoir la taille => appliquer weight / len(set)
        # TODO: Deplacer fonction dans QBindiff + réécrire fonction simple pour Differ

        for program_features in (primary_features, secondary_features):
            for function_features in program_features:
                for key, value in function_features.features.items():
                    if isinstance(value, dict):
                        feature_keys.update(dict.from_keys(value.keys(), visitor.get_feature(key).weight))
                    else:
                        feature_keys[key] = visitor[key].weight
        feature_list = sorted(feature_keys)
        feature_weights = [feature_keys[key] for key in feature_list]
        return feature_list, feature_weights

    @staticmethod
    def _vectorize_features(iterable_features: List[Environment], feature_list: List[str]) -> FeatureVectors:
        feature_index = {key: idx for idx, key in enumerate(feature_list)}
        feature_matrix = np.zeros((len(iterable_features), len(feature_index)), dtype=Differ.DTYPE)
        for idx, env in enumerate(iterable_features):
            if env.features:  # if the node has features (otherwise array cells are already zeros)
                idy, value = zip(*((feature_index[key], value) for key, value in env.features.items()))
                feature_matrix[idx, list(idy)] += value
        return feature_matrix

    @staticmethod
    def _compute_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: str='canberra', weights: List[Positive]=1.0) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, weights)
        matrix /= matrix.max()
        return matrix

    @staticmethod
    def _compute_feature_similarity(primary_matrix: FeatureVectors, secondary_matrix: FeatureVectors, distance: str, weights: List[Positive]) -> SimMatrix:
        matrix = scipy.spatial.distance.cdist(primary_matrix, secondary_matrix, distance, w=weights).astype(Differ.DTYPE)
        matrix /= matrix.max()
        matrix[:] = 1 - matrix
        return matrix

    def _convert_mapping(self, mapping: RawMapping) -> Mapping:
        idx, idy = mapping
        similarities, squares = self._compute_statistics(mapping)

        # Compute the unmatched as the difference from all idx to matched ones
        unmatch_p = list(set(range(len(self.primary_affinity))) - set(idx))
        unmatch_s = list(set(range(len(self.secondary_affinity))) - set(idy))

        return Mapping(list(zip(idx, idy, similarities, squares)), (unmatch_p, unmatch_s))

    def _compute_statistics(self, mapping: RawMapping) -> Tuple[List[float], List[int]]:
        idx, idy = mapping
        similarities = self.sim_matrix[idx, idy]
        common_subgraph = self.primary_affinity[np.ix_(idx, idx)]
        common_subgraph &= self.secondary_affinity[np.ix_(idy, idy)]
        squares = common_subgraph.sum(0) + common_subgraph.sum(1)
        return similarities, squares



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


    @staticmethod
    def diff(primary: Iterable,
             secondary: Iterable,
             primary_affinity: AffinityMatrix,
             secondary_affinity: AffinityMatrix,
             visitor: Visitor,
             distance: str='canberra',
             anchors: AddrAnchors=None,
             sparsity_ratio: Ratio=.75,
             tradeoff: Ratio=.75,
             epsilon: Positive=.5,
             maxiter: int=1000) -> AddressMapping:

        # Compute reverse index to add anchor
        reverse_primary = {x.address: i for i,x in enumerate(primary)}
        reverse_secondary = {x.address: i for i,x in enumerate(secondary)}
        if anchors:
            anchors = [(reverse_primary[addr1], reverse_secondary[addr2]) for addr1, addr2 in anchors]

        # Compute the diff using indexes
        idx_mapping = super().diff(primary, secondary, primary_affinity, secondary_affinity, visitor, distance, anchors, sparsity_ratio, tradeoff, epsilon, maxiter)

        # Convert back indexes to Addresses
        primary_lookup = {i: x.address for i, x in enumerate(primary)}
        secondary_lookup = {i: x.address for i, x in enumerate(secondary)}
        addr_mapping = [(primary_lookup[idx1], secondary_lookup[idx2], sim, sq) for idx1, idx2, sim, sq in idx_mapping]

        # Compute unmatched addresses
        unmatched_p = [primary_lookup[idx] for idx in idx_mapping.primary_unmatched]
        unmatched_s = [secondary_lookup[idx] for idx in idx_mapping.secondary_unmatched]

        # Recreate an AddressMapping
        return AddressMapping(primary, secondary, addr_mapping, (unmatched_p, unmatched_s))

    def diff_program(self, distance: str='canberra',
                           sparsity_ratio: Ratio=.75,
                           tradeoff: Ratio= .75,
                           epsilon: Positive= .5,
                           maxiter: int=1000,
                           anchors: AddrAnchors=None) -> AddressMapping:
        self.mapping = self.diff(self.primary, self.secondary, self.primary.callgraph, self.secondary.callgraph, self._visitor, distance, anchors, sparsity_ratio, tradeoff, epsilon, maxiter)
        return self.mapping

    def diff_function(self, primary: Function,
                            secondary: Function,
                            distance: str='canberra',
                            sparsity_ratio: Ratio=.75,
                            tradeoff: Ratio= .75,
                            epsilon: Positive= .5,
                            maxiter: int=1000,
                            anchors: AddrAnchors=None) -> AddressMapping:
        self.mapping = self.diff(primary, secondary, primary.flowgraph, secondary.flowgraph, self._visitor, distance, anchors, sparsity_ratio, tradeoff, epsilon, maxiter)
        return self.mapping


    def save_sqlite(self, filename: PathLike=''):
        self.mapping.save_sqlite(filename)

    def register_feature(self, feature: Feature, weight: Positive=1.0):
        self._visitor.register_feature(feature, weight)


    # @staticmethod
    # def _compute_constant_similarity(primary_constants: List[Tuple[Str]], secondary_constants: List[Tuple[Str]],
    #                                  weight: Positive = .5, dtype: Dtype) -> SimMatrix:
    #     matrix = np.zeros((len(primary_constants), len(secondary_constants)), dtype)
    #     for constant in set(primary_constants).intersection(secondary_constants):
    #         idx, idy = zip(*product(primary_constants[constant], secondary_constants[constant]))
    #         np.add.at(matrix, (idx, idy), weight / len(idx))
    #     return matrix
















