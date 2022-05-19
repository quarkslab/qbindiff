from collections import defaultdict
from collections.abc import Iterable
from typing import Union, Any, Optional

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand
from qbindiff.types import Positive


class FeatureCollector:
    """
    Dict wrapper, representing a collection of features where the key is the feature
    name and the value is the feature score which can be either a number or a dict.
    """

    def __init__(self):
        self._features: dict[str, Union[float, dict[str, float]]] = {}

    def add_feature(self, key: str, value: float) -> None:
        self._features.setdefault(key, 0)
        self._features[key] += value

    def add_dict_feature(self, key: str, value: dict[str, float]) -> None:
        self._features.setdefault(key, defaultdict(float))
        for k, v in value.items():
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

    def to_vector(
        self, key_order: dict[str, Iterable[str]], empty_default: Optional[Any] = None
    ) -> list[float]:
        """
        Transform the collection to a feature vector. If the parameter `empty_default`
        is specified then if the feature vector is the zero vector the `empty_default`
        is returned

        :param key_order: The order in which the keys are accessed
        :param empty_default: Default value to be returned in case the feature vector
                              is a zero vector, if None then the zero vector is returned
        """

        vector = []
        is_zero = True
        for main_key, subkey_list in key_order.items():
            if subkey_list:
                feature = self._features.get(main_key, {})
                for subkey in subkey_list:
                    val = feature.get(subkey, 0)
                    if val != 0:
                        is_zero = False
                    vector.append(val)
            else:
                value = self._features.get(main_key, 0)
                if not value:  # It might be a empty dict or a list, ...
                    vector.append(0)
                else:
                    vector.append(value)
                    is_zero = False

        if is_zero and empty_default is not None:
            return empty_default
        return vector


class FeatureExtractor:
    """
    Abstract class that represent a feature extractor which sole contraints are to
    define a unique key and a function call that is to be called by the visitor.
    """

    key = ""

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
