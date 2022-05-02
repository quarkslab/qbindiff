from collections import defaultdict
from collections.abc import Iterable
from typing import Union

from qbindiff.loader import Function, BasicBlock, Instruction, Operand, Expr
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

    def to_vector(self, key_order: dict[str, Iterable[str]]) -> list[float]:
        """
        Transform the collection to a feature vector

        :param key_order: The order in which the keys are accessed
        """
        vector = []
        for main_key, subkey_list in key_order.items():
            if subkey_list:
                feature = self._features.get(main_key, {})
                for subkey in subkey_list:
                    vector.append(feature.get(subkey, 0))
            else:
                vector.append(self._features.get(main_key, 0))

        return vector


class FeatureExtractor:
    """
    Abstract class that represent a feature extractor which sole contraints are to
    define name, key and a function call that is to be called by the visitor.
    """

    name = ""
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
    def visit_function(self, function: Function, collector: FeatureCollector) -> None:
        raise NotImplementedError()


class BasicBlockFeatureExtractor(FeatureExtractor):
    def visit_basic_block(
        self, basicblock: BasicBlock, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class InstructionFeatureExtractor(FeatureExtractor):
    def visit_instruction(
        self, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class OperandFeatureExtractor(FeatureExtractor):
    def visit_operand(self, operand: Operand, collector: FeatureCollector) -> None:
        raise NotImplementedError()


class ExpressionFeatureExtractor(FeatureExtractor):
    def visit_expression(self, expr: Expr, collector: FeatureCollector) -> None:
        raise NotImplementedError()
