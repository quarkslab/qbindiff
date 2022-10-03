from dataclasses import dataclass
from scipy.sparse import lil_array
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

from qbindiff.features.manager import FeatureKeyManager
from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand
from qbindiff.types import Positive, SparseVector


class FeatureCollector:
    """
    Dict wrapper, representing a collection of features where the key is the feature
    name and the value is the feature score which can be either a number or a dict.
    """

    def __init__(self):
        self._features: dict[str, float | dict[str, float]] = {}

    def add_feature(self, key: str, value: float) -> None:
        FeatureKeyManager.add(key)
        self._features.setdefault(key, 0)
        self._features[key] += value

    def add_dict_feature(self, key: str, value: dict[str, float]) -> None:
        self._features.setdefault(key, defaultdict(float))
        for k, v in value.items():
            FeatureKeyManager.add(key, k)
            self._features[key][k] += v

    def full_keys(self) -> dict[str, set[str]]:
        """
        Returns a dict in which keys are the keys of the features and values are the
        subkeys.
        Ex: {feature_key1: [], feature_key2: [], ..., feature_keyN: [subkey1, ...], ...}
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
        """

        manager = FeatureKeyManager
        size = manager.get_cum_size(main_key_list)
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


T = TypeVar("T")


@dataclass
class FeatureOption:
    name: str
    description: str
    parser: Callable[[str], T]


class FeatureExtractor:
    """
    Abstract class that represent a feature extractor which sole contraints are to
    define a unique key and a function call that is to be called by the visitor.
    """

    key = ""
    options: dict[str, FeatureOption] = {}  # Dict {name : option}, each option is a instance of FeatureOption

    def __init__(self, weight: Positive = 1.0):
        self._weight = weight

    @property
    def weight(self) -> Positive:
        return self._weight

    @weight.setter
    def weight(self, value: Positive) -> None:
        self._weight = value


class FunctionFeatureExtractor(FeatureExtractor):
    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class BasicBlockFeatureExtractor(FeatureExtractor):
    def visit_basic_block(
        self, program: Program, basicblock: BasicBlock, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class InstructionFeatureExtractor(FeatureExtractor):
    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class OperandFeatureExtractor(FeatureExtractor):
    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()
