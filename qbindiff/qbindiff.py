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

    @staticmethod
    def load(primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix, visitor: Visitor, distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Idx, Idx]=[]) -> Tuple[SimMatrix, AffinityMatrix, AffinityMatrix]:
        """
        Initialize the diffing instance by computing the pairwise similarity between the nodes
        of the two graphs to diff.
        :param visitor: list of features extractors to apply
        :param distance: distance metric to use (will be later converted into a similarity metric)
        :param dtype: datatype of the similarity measures
        :param anchors: user defined mapping correspondance
        :return: similarity matrix and the respective affinity matrix of both graphs
        """
        primary_features = visitor.visit(primary)
        secondary_features = visitor.visit(secondary)

        feature_index, feature_weights = Differ._build_feature_index(primary_features, secondary_features, visitor)
        primary_matrix = Differ._vectorize_features(primary_features, feature_index, dtype)
        secondary_matrix = Differ._vectorize_features(secondary_features, feature_index, dtype)

        sim_matrix = Differ._compute_similarity(primary_matrix, secondary_matrix, distance, dtype, feature_weights)
        Differ._apply_anchors(sim_matrix, anchors)
        return sim_matrix, primary_affinity, secondary_affinity
    
    @staticmethod
    def load_from_file(file: PathLike) -> Tuple[SimMatrix, AffinityMatrix, AffinityMatrix]:

        return sim_matrix, primary_affinity, secondary_affinity

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
    def _vectorize_features(program_features: List[Environment], feature_index: Dict, dtype: np.dtype) -> FeatureVector:
        feature_matrix = np.zerors((len(program_features), len(feature_index)), dtype=dtype)
        for idx, function_features in enumerate(program_features.values()):
            if function_features:  # if the function have features (otherwise array cells are already zeros)
                idy, value = zip(*((feature_index[key], value) for key, value in function_features.items()))
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
    def set_anchors(anchors:Tuple[Idx, Idx]=[]) -> Tuple[Idx, Idx]:
        return anchors

    @staticmethod
    def _apply_anchors(matrix:SimMatrix, anchors:Tuple[Idx, Idx]) -> SimMatrix:
        if anchors:
            idx, idy = anchors
            matrix[idx] = 0
            matrix[:, idy] = 0
            matrix[idx, idy] = 1

    @staticmethod
    def compute(sim_matrix: SimMatrix, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix, sparsity_ratio, epsilon, maxiter, tradeoff) -> Mapping:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then converted into a binary-based format
        :return: iterable
        """
        matcher = Matcher(sim_matrix, primary_affinity, secondary_affinity)
        matcher.process(sparsity_ratio)
        for _ in tqdm(matcher.compute(tradeoff, epsilon, maxiter), total=maxiter):
            pass
        logging.info(matcher.display_statistics())

        return matcher.format_mapping()


    @staticmethod
    def _diff(primary: Iterable, secondary: Iterable, primary_affinity: AffinityMatrix, secondary_affinity: AffinityMatrix, visitor: Visitor,
             distance: Str='canberra', dtype: np.dtype=np.float32, anchors:Tuple[Idx, Idx]=[],
             sparsity_ratio: Ratio=.0, epsilon: Float= .5, maxiter: Int=1000, tradeoff: Ratio: .75) -> Mapping:
        sim_matrix, primary_affinity, secondary_affinity = Differ.load(primary, secondary, primary_affinity, secondary_affinity, visitor, distance, dtype, anchors)
        mapping = Differ.compute(sim_matrix, primary_affinity, secondary_affinity, sparsity_ratio, epsilon, maxiter, tradeoff)
        return Mapping(mapping)



class QBinDiff(Differ):
    """
    QBinDiff class that provides a high-level interface to trigger a diff between two binaries.
    """

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        super(QBinDiff, self).__init__(ProgramVisitor())
        self.primary = primary
        self.secondary = secondary

    def register_feature(self, feature: FeatureExtractor, weight: Float=1):
        self.visitor.register_feature(feature, weight)

    def diff_program(self) -> FunctionMapping:
        anchors = self.set_anchors()
        mapping = self._diff(self.primary, self.secondary, self.primary.callgraph, self.secondary.callgraph, anchors)
        return FunctionMapping(self.primary, self.secondary, mapping)

    def diff_function(self, primary_function: Function, secondary_function: Function) -> BasicBlockMapping:
        anchors = self.set_anchors()
        mapping = self._diff(primary_function, secondary_function, primary_function.flowgraph, secondary_function.flowgraph, anchors)
        return BBlockMapping(self.primary, self.secondary, mapping)

    def _compute_similarity(self, primary_matrix: FeatureVector, secondary_matrix: FeatureVector, distance: Str='canberra', dtype: np.dtype=np.float32) -> SimMatrix:
        matrix = Differ._compute_feature_similarity(primary_matrix, secondary_matrix, distance, dtype)
        matrix += .01 * Differ._compute_address_similarity(len(self.primary), len(self.secondary), dtype)
        matrix /= matrix.max()
        return matrix

    def set_anchors(self, anchors:Tuple[Idx, Idx]=[]) -> Tuple[Idx, Idx]:
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

