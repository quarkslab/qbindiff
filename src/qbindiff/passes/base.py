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

"""Standard passes

"""

import logging
import tqdm
import numpy as np
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from qbindiff.loader import Program
from qbindiff.visitor import ProgramVisitor
from qbindiff.features.manager import FeatureKeyManager
from qbindiff.features.extractor import FeatureExtractor, FeatureCollector
from qbindiff.passes.metrics import pairwise_distances
from qbindiff.utils import is_debug
from qbindiff.types import SimMatrix, Distance


class FeaturePass:
    """
    Run all the feature extractors previously registered and compute the similarity
    matrix
    """

    def __init__(self, distance: Distance):
        """

        :param distance: distance to compute the similarity of type py:class:`qbindiff.types.Distance`
        """

        self._default_distance = distance
        self._distances = {}
        self._visitor = ProgramVisitor()

    def distance(self, key: str) -> Distance:
        """Returns the correct distance for the given feature key"""
        return self._distances.get(key, self._default_distance)

    def register_extractor(
        self, extractor: FeatureExtractor, distance: Distance | None = None
    ) -> None:
        """
        Register a feature extractor optionally specifying a distance to use.
        The class will be called when the visitor will traverse the graph.
        """
        self._visitor.register_feature_extractor(extractor)
        if distance is not None:
            self._distances[extractor.key] = distance

    def _create_feature_matrix(
        self,
        features: dict[Any, FeatureCollector],
        features_main_keys: list[str],
        node_to_index: dict[Any, int],
        shape: tuple[int, int],
        dtype: type,
    ):
        """
        Utility function to generate the sparse feature matrix.
        It returns the sparse feature matrix where each row corresponds to a node and
        each column to a feature.
        If a node has no features it is converted to a zero vector.
        The mapping node_to_index will be used to map each node label to an index in the
        feature matrix.

        :param features: Dict of features {node : feature_collector}
        :param features_main_keys: List of all the features main keys
        :param node_to_index: Dict representing the mapping between nodes to indexes in
                              the similarity matrix. {node : sim_index}
        :param shape: The shape of the sparse feature matrix
        :param dtype: dtype of the feature matrix
        """

        feature_matrix = lil_matrix(shape, dtype=dtype)
        for node_label, feature in tqdm.tqdm(
            features.items(), total=len(features), disable=not is_debug()
        ):
            vec = feature.to_sparse_vector(dtype, features_main_keys)
            feature_matrix[node_to_index[node_label]] = vec
        return feature_matrix.tocsr()

    def _compute_sim_matrix(
        self,
        shape: tuple[int, int],
        primary_features: dict[Any, FeatureCollector],
        secondary_features: dict[Any, FeatureCollector],
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
        features_main_keys: list[str],
        distance: Distance,
        dtype: type,
        weights: Iterable[float] | None = None,
    ):
        """
        Utility function that generate a similarity matrix given two collections of
        features only considering a subset of the features (specified by feature_keys).
        WARNING: Please note that it will calculate the distance even between nodes
        that have no features. The results should be subsequently discarded as they are
        not representative of the similarity between the two nodes.

        It returns the whole similarity matrix of the given shape.

        :param shape: The shape of the similarity matrix. It is zero initialized
        :param primary_features: Dict of features {node : feature_collector}
        :param secondary_features: Dict of features {node : feature_collector}
        :param primary_mapping: A mapping between function labels and indexes in the
                                similarity matrix
        :param secondary_mapping: A mapping between function labels and indexes in the
                                  similarity matrix
        :param features_main_keys: List of all the features main keys to consider
        :param distance: The distance to use
        :param dtype: dtype of the similarity matrix
        :param weights: Optional weights
        """

        # Find the dimension of the feature matrix
        dim = FeatureKeyManager.get_cumulative_size(features_main_keys)

        # Build the sparse feature matrices
        logging.debug(f"Building primary feature matrix of size {(shape[0], dim)}")
        primary_feature_matrix = self._create_feature_matrix(
            primary_features,
            features_main_keys,
            primary_mapping,
            (shape[0], dim),
            dtype,
        )
        logging.debug(
            f"Sparse primary feature matrix computed, nnz element: {primary_feature_matrix.nnz}"
        )
        logging.debug(f"Building secondary feature matrix of size {(shape[1], dim)}")
        secondary_feature_matrix = self._create_feature_matrix(
            secondary_features,
            features_main_keys,
            secondary_mapping,
            (shape[1], dim),
            dtype,
        )
        logging.debug(
            f"Sparse secondary feature matrix computed, nnz element: {secondary_feature_matrix.nnz}"
        )
        logging.debug(f"Calculating distance {distance}")

        # Generate the partial similarity matrix (only non empty rows and cols)
        if weights:
            sim_matrix = pairwise_distances(
                primary_feature_matrix,
                secondary_feature_matrix,
                metric=distance,
                w=weights,
                n_jobs=-1,
            ).astype(dtype)
        else:
            sim_matrix = pairwise_distances(
                primary_feature_matrix,
                secondary_feature_matrix,
                metric=distance,
                n_jobs=-1,
            ).astype(dtype)

        logging.debug("Distance calculated")
        # Normalize
        if len(sim_matrix) > 0 and sim_matrix.max() != 0:
            sim_matrix /= sim_matrix.max()
        sim_matrix[:] = 1 - sim_matrix

        return sim_matrix

    def __call__(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
        fill: bool = False,
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

        p_features = primary_features
        s_features = secondary_features

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

        # Build the similarity matrices separately for each main feature.
        # The linear combination is perfomed online to save precious memory.
        result_matrix = np.zeros_like(sim_matrix)
        simple_feature_keys = defaultdict(list)  # {distance: [main_key, ...], ...}
        norm_coeff = 0
        for main_key, subkeys in features_keys.items():
            if subkeys:
                # Compute the similarity matrix for the current feature
                logging.debug(f"Computing similarity matrix for the feature {main_key}")
                tmp_sim_matrix = self._compute_sim_matrix(
                    sim_matrix.shape,
                    primary_features,
                    secondary_features,
                    primary_mapping,
                    secondary_mapping,
                    [main_key],
                    self.distance(main_key),
                    sim_matrix.dtype,
                )
                result_matrix += f_weights[main_key] * tmp_sim_matrix

                del tmp_sim_matrix  # Free the memory
                norm_coeff += f_weights[main_key]
            else:
                # It is a simple feature (no subkeys)
                simple_feature_keys[self.distance(main_key)].append(main_key)
        # Add the simple features similarity
        for distance, features in simple_feature_keys.items():
            logging.debug(
                "Computing similarity matrix for the simple features "
                f"{features} using the distance {distance}"
            )
            weights = tuple(f_weights[key] for key in features)
            tmp_sim_matrix = self._compute_sim_matrix(
                sim_matrix.shape,
                primary_features,
                secondary_features,
                primary_mapping,
                secondary_mapping,
                features,
                distance,
                sim_matrix.dtype,
                weights=weights,
            )
            norm_weight = sum(weights)
            result_matrix += norm_weight * tmp_sim_matrix
            del tmp_sim_matrix  # Free the memory
            norm_coeff += norm_weight
        # Normalize
        result_matrix /= norm_coeff

        # Overwrite the real similarity matrix
        for idx in map(lambda l: primary_mapping[l], ignore_primary):
            result_matrix[idx] = sim_matrix[idx]
        for idx in map(lambda l: secondary_mapping[l], ignore_secondary):
            result_matrix[:, idx] = sim_matrix[:, idx]
        sim_matrix[:] = result_matrix
        return p_features, s_features
