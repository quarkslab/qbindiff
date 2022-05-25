import numpy as np
import scipy.spatial
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Iterable

from qbindiff.loader import Program
from qbindiff.features.extractor import FeatureExtractor, FeatureCollector
from qbindiff.visitor import ProgramVisitor
from qbindiff.types import SimMatrix


class GenericPass(metaclass=ABCMeta):
    """Class to define a interface to Passes"""

    @abstractmethod
    def __call__(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
    ) -> None:
        """Execute the pass that operates on the similarity matrix inplace"""
        raise NotImplementedError()


class FeaturePass(GenericPass):
    """
    Run all the feature extractors previously registered and compute the similarity
    matrix
    """

    def __init__(self, distance: str):
        self._default_distance = distance
        self._distances = {}
        self._visitor = ProgramVisitor()

    def distance(self, key: str) -> str:
        """Returns the correct distance for the given feature key"""
        return self._distances.get(key, self._default_distance)

    def register_extractor(
        self, extractor: FeatureExtractor, distance: Optional[str] = None
    ) -> None:
        """
        Register a feature extractor optionally specifying a distance to use.
        The class will be called when the visitor will traverse the graph.
        """
        self._visitor.register_feature_extractor(extractor)
        if distance:
            self._distances[extractor.key] = distance

    def _create_feature_matrix(
        self,
        features: dict[Any, FeatureCollector],
        features_keys: dict[str, Iterable[str]],
        node_to_index: dict[Any, int],
        dim: int,
        dtype: type,
    ):
        """
        Utility function to generate the feature matrix.
        It returns a tuple with (feature_matrix, mapping, nonempty_set) where
          feature_matrix: is the actual feature matrix, each row corresponds to a
                          node and each column to a feature
          mapping: a dict representing the mapping between the nodes index in the
                   adjacency matrix and in the similarity matrix.
                   {adj_index : sim_index}
          nonempty_set: set with all the node index (index in the adjacency matrix)
                        that have been added to the feature matrix (aka nodes with
                        non empty feature vector)

        :param features: Dict of features {node : feature_collector}
        :param features_keys: List of all the features keys
        :param node_to_index: Dict representing the mapping between nodes to indexes in
                              the similarity matrix. {node : sim_index}
        :param dim: Size of the feature matrix
        :param dtype: dtype of the feature matrix
        """

        feature_matrix = np.zeros((0, dim), dtype=dtype)
        mapping = {}
        nonempty_set = set()
        i = 0
        for node_label, feature in features.items():
            node_index = node_to_index[node_label]
            vec = feature.to_vector(features_keys, False)
            if vec:
                mapping[node_index] = i
                feature_matrix = np.vstack((feature_matrix, vec))
                nonempty_set.add(node_index)
                i += 1
        return (feature_matrix, mapping, nonempty_set)

    def _compute_sim_matrix(
        self,
        shape: tuple[int, int],
        primary_features: dict[Any, FeatureCollector],
        secondary_features: dict[Any, FeatureCollector],
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
        features_keys: dict[str, list[str]],
        distance: str,
        dtype: type,
        weights: Optional[Iterable[float]] = None,
    ):
        """
        Utility function that generate a similarity matrix given two collections of
        features only considering a subset of the features (specified by feature_keys).

        It returns the whole similarity matrix of the given shape.

        :param shape: The shape of the similarity matrix. It is zero initialized
        :param primary_features: Dict of features {node : feature_collector}
        :param secondary_features: Dict of features {node : feature_collector}
        :param primary_mapping: A mapping between function labels and indexes in the
                                similarity matrix
        :param secondary_mapping: A mapping between function labels and indexes in the
                                  similarity matrix
        :param features_keys: All the features keys to consider
        :param distance: The distance to use
        :param dtype: dtype of the similarity matrix
        :param weights: Optional weights
        """

        # Find the dimension of the feature matrix
        dim = 0
        for main_key, subkeys in features_keys.items():
            if subkeys:
                dim += len(subkeys)
            else:
                dim += 1

        # Build the feature matrices
        (
            primary_feature_matrix,  # the feature matrix
            temp_map_primary,  # temporary mappings between the nodes index in the adjacency matrix and in the similarity matrix
            nonempty_rows,  # non empty rows that will be kept after the distance is calculated
        ) = self._create_feature_matrix(
            primary_features, features_keys, primary_mapping, dim, dtype
        )
        (
            secondary_feature_matrix,
            temp_map_secondary,
            nonempty_cols,
        ) = self._create_feature_matrix(
            secondary_features, features_keys, secondary_mapping, dim, dtype
        )

        # Generate the partial similarity matrix (only non empty rows and cols)
        if weights:
            tmp_sim_matrix = scipy.spatial.distance.cdist(
                primary_feature_matrix,
                secondary_feature_matrix,
                distance,
                w=weights,
            ).astype(dtype)
        else:
            tmp_sim_matrix = scipy.spatial.distance.cdist(
                primary_feature_matrix, secondary_feature_matrix, distance
            ).astype(dtype)

        # Normalize
        if len(tmp_sim_matrix) > 0 and tmp_sim_matrix.max() != 0:
            tmp_sim_matrix /= tmp_sim_matrix.max()
        tmp_sim_matrix[:] = 1 - tmp_sim_matrix

        # Fill the entire similarity matrix
        sim_matrix = np.zeros(shape, dtype=dtype)
        for idx in nonempty_rows:  # Rows insertion
            sim_matrix[idx, : tmp_sim_matrix.shape[1]] = tmp_sim_matrix[
                temp_map_primary[idx]
            ]
        # Cols permutation
        cols_dim = sim_matrix.shape[1]
        mapping = np.full(cols_dim, cols_dim - 1, dtype=int)
        for idx in nonempty_cols:
            mapping[idx] = temp_map_secondary[idx]
        sim_matrix[:] = sim_matrix[:, mapping]

        return sim_matrix

    def __call__(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
        fill: Optional[bool] = False,
    ) -> None:
        """
        Generate the similarity matrix by calculating the distance between the feature
        vectors.

        :param fill: if True the whole matrix will be erased before writing in it
        """

        # fill the matrix with -1 (not-set value)
        if fill:
            sim_matrix[:] = -1

        reverse_primary_mapping = {idx: l for l, idx in primary_mapping.items()}
        reverse_secondary_mapping = {idx: l for l, idx in secondary_mapping.items()}

        # Do not extract features on functions that already have a similarity score
        ignore_primary = set()
        ignore_secondary = set()
        for i in range(sim_matrix.shape[0]):
            if -1 not in sim_matrix[i]:
                ignore_primary.add(reverse_primary_mapping[i])
        for j in range(sim_matrix.shape[1]):
            if -1 not in sim_matrix[:, j]:
                ignore_secondary.add(reverse_secondary_mapping[j])

        # Extract the features
        primary.set_function_filter(lambda label: label not in ignore_primary)
        secondary.set_function_filter(lambda label: label not in ignore_secondary)
        key_fun = lambda *args: args[0][0]  # ((label, node), iteration)
        primary_features = self._visitor.visit(primary, key_fun=key_fun)
        secondary_features = self._visitor.visit(secondary, key_fun=key_fun)
        primary.set_function_filter(lambda _: True)
        secondary.set_function_filter(lambda _: True)

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

        # Build the similarity matrices separately for each main feature
        all_sim_matrix = []
        simple_feature_keys = defaultdict(dict)  # {distance: {main_key: (), ...}, ...}
        norm_coeff = 0
        for main_key, subkeys in features_keys.items():
            if subkeys:
                # Compute the similarity matrix for the current feature
                tmp_sim_matrix = self._compute_sim_matrix(
                    sim_matrix.shape,
                    primary_features,
                    secondary_features,
                    primary_mapping,
                    secondary_mapping,
                    {main_key: subkeys},
                    self.distance(main_key),
                    sim_matrix.dtype,
                )
                all_sim_matrix.append(f_weights[main_key] * tmp_sim_matrix)
                norm_coeff += f_weights[main_key]
            else:
                # It is a simple feature (no subkeys)
                simple_feature_keys[self.distance(main_key)][main_key] = ()
        # Add the simple features similarity
        for distance, simple_feature_keys in simple_feature_keys.items():
            weights = tuple(f_weights[key] for key in simple_feature_keys)
            tmp_sim_matrix = self._compute_sim_matrix(
                sim_matrix.shape,
                primary_features,
                secondary_features,
                primary_mapping,
                secondary_mapping,
                simple_feature_keys,
                distance,
                sim_matrix.dtype,
                weights=weights,
            )
            norm_weight = sum(weights)
            all_sim_matrix.append(norm_weight * tmp_sim_matrix)
            norm_coeff += norm_weight

        # Build the whole similarity matrix by combining the previous ones
        res = sum(all_sim_matrix) / norm_coeff

        sim_matrix[res.nonzero()] = res[res.nonzero()]
