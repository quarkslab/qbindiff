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

"""Interface of the feature extractor
"""

from scipy.sparse import lil_array
from collections import defaultdict

from qbindiff.features.manager import FeatureKeyManager
from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand
from qbindiff.types import Positive, SparseVector, FeatureValue


class FeatureCollector:
    """
    The FeatureCollector is in charge of aggregate all the features
    values so that it can compute the vector embedding.
    FeatureExtractor objects receive the collector as argument of the visit
    functions, and have to register theirs own result to it. The feature
    score can either be a number or a dict.
    """

    def __init__(self):
        self._features: dict[str, FeatureValue] = {}

    def add_feature(self, key: str, value: float) -> None:
        """
        Add a feature value in the collector. Features are responsible to call
        this function to register a value with its own name.

        :param key: name of the feature adding the value
        :param value: float value to be added in the collector
        """
        FeatureKeyManager.add(key)
        self._features.setdefault(key, 0)
        self._features[key] += value

    def add_dict_feature(self, key: str, value: dict[str, float]) -> None:
        """
        Add a feature value in the collector if the value is a dictionary of string to float.

        :param key: name of the feature adding the value
        :param value: Feature value to add
        """
        self._features.setdefault(key, defaultdict(float))

        if value == {}:
            FeatureKeyManager.add(key)
            self._features[key] = 0
        else:
            for k, v in value.items():
                FeatureKeyManager.add(key, k)
                self._features[key][k] += v

    def feature_vector(self) -> None:
        """Show the feature vector associated to the node"""
        feature_vector = {}
        for main_key, feature in self._features.items():
            feature_vector[main_key] = feature
        return feature_vector

    def full_keys(self) -> dict[str, set[str]]:
        """
        Returns a dict in which keys are the keys of the features and values are the subkeys.
        If a Feature directly maps to a float value, the set will be empty.

        e.g: {feature_key1: [], feature_key2: [], ..., feature_keyN: [subkey1, ...], ...}

        :return: dictionary mapping feature keys to their subkeys if any
        """
        keys = {}
        for main_key, feature in self._features.items():
            keys.setdefault(main_key, set())
            if isinstance(feature, dict):
                keys[main_key].update(feature.keys())

        return keys

    def to_sparse_vector(self, dtype: type, main_key_list: list[str]) -> SparseVector:
        """
        Transform the collection to a sparse feature vector.

        :param dtype: dtype of the sparse vector
        :param main_key_list: A list of main keys that act like a filter: only those
                              keys are considered when building the vector.
        :return:
        """

        manager = FeatureKeyManager
        size = manager.get_cumulative_size(main_key_list)
        vector = lil_array((1, size), dtype=dtype)
        offset = 0
        for main_key in sorted(main_key_list):  # Sort them to keep consistency
            if main_key not in self._features:
                offset += manager.get_size(main_key)
                continue

            if isinstance(self._features[main_key], dict):  # with subkeys
                for subkey, value in self._features[main_key].items():
                    vector[0, offset + manager.get(main_key, subkey)] = value
            else:  # without subkeys
                vector[0, offset] = self._features[main_key]
            offset += manager.get_size(main_key)
        return vector.tocsr()


class FeatureExtractor:
    """
    Base class that represent a feature extractor which sole contraints are to
    define a unique key and a function call that is to be called by the visitor.
    """

    key: str = ""  #: feature name (short)

    def __init__(self, weight: Positive = 1.0):
        """
        :param weight: weight to apply to this feature
        """
        self._weight = weight

    @property
    def weight(self) -> Positive:
        """
        Weight applied to the feature
        """
        return self._weight

    @weight.setter
    def weight(self, value: Positive) -> None:
        """
        Set the weight to the feature.

        :param value: weight value to set
        """
        self._weight = value


class FunctionFeatureExtractor(FeatureExtractor):
    """
    Function extractor feature. It inherits FeatureExtractor
    and defines the method :py:meth:`visit_function` that has
    to be implemented by all its inheriting classes.
    """

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Function being called by the visitor when encountering a function in the program.
        Inheriting classes of the implement the feature extraction in this method.

        :param program: Program being visited
        :param function: Function object being visited
        :param collector: collector in which to save the feature value.
        """
        raise NotImplementedError()


class BasicBlockFeatureExtractor(FeatureExtractor):
    """
    Basic Block extractor feature. It inherits from FeatureExtractor
    and defines the method :py:meth:`visit_basic_block` that has
    to be implemented by all its inheriting classes.
    """

    def visit_basic_block(
        self, program: Program, basicblock: BasicBlock, collector: FeatureCollector
    ) -> None:
        """
        Function being called by the visitor when encountering a basic block in the program.
        Classes inheriting have to implement this method.

        :param program: program being visited
        :param basicblock: basic block being visited
        :param collector: collector in which to save the feature value
        """
        raise NotImplementedError()


class InstructionFeatureExtractor(FeatureExtractor):
    """
    Instruction extractor feature. It inherits from FeatureExtractor
    and defines the method :py:meth:`visit_instruction` that has to
    be implemented by all its inheriting classes.
    """

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Function being called by the visitor when encountering an instruction in the program.
        Classes inheriting have to implement this method.

        :param program: program being visited
        :param instruction: instruction being visited
        :param collector: collector in which to save the feature value
        """
        raise NotImplementedError()


class OperandFeatureExtractor(FeatureExtractor):
    """
    Operand extractor feature. It inherits from FeatureExtractor
    and defines the method :py:meth:`visit_operand` that has to be
    implemented by all its inheriting classes.
    """

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        """
        Function being called by the visitor when encountering an operand in the program.
        Classes inheriting have to implement this method.

        :param program: program being visited
        :param operand: operand being visited
        :param collector: collector in which to save the feature value
        """
        raise NotImplementedError()
