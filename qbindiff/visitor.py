from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable

from qbindiff.loader import Function, BasicBlock, Instruction, Operand
from qbindiff.features.extractor import (
    FeatureCollector,
    FeatureExtractor,
    FunctionFeatureExtractor,
    BasicBlockFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
)
from qbindiff.types import Graph


class Visitor(metaclass=ABCMeta):
    """
    Abstract class representing interface that a visitor
    must implements to work with a Differ object.
    """

    @property
    @abstractmethod
    def feature_extractors(self) -> list[FeatureExtractor]:
        """Returns the list of registered features extractor"""
        raise NotImplementedError()

    def visit(
        self, graph: Graph, key_fun: Callable = lambda _, i: i
    ) -> dict[Any, FeatureCollector]:
        """
        Function performing the visit on a Graph object by calling visit_item with a
        FeatureCollector meant to be filled.

        :param graph: the Graph to be visited.
        :param key_fun: a function that takes 2 input arguments, namely the current item and
                        the current iteration number, and returns a unique key for that item.
                        By default the iteration number is used.
        :return: A dict in which keys are key_fun(item, i) and values are the FeatureCollector
        """
        obj_features = {}
        for i, item in enumerate(graph.items()):
            _, node = item
            collector = FeatureCollector()
            self.visit_item(node, collector)
            obj_features[key_fun(item, i)] = collector
        return obj_features

    def visit_item(self, item: Any, collector: FeatureCollector) -> None:
        """
        Abstract method meant to perform the visit of the item.
        It receives an environment in parameter that is meant to be filled.

        :param item: item to be visited
        :param collector: FeatureCollector to fill during the visit
        """
        raise NotImplementedError()

    @abstractmethod
    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.

        :param ft: Feature extractor instance
        """
        raise NotImplementedError()


class NoVisitor(Visitor):
    """
    Trivial visitor that doesn't traverse the items
    """

    @property
    def feature_extractors(self) -> list[FeatureExtractor]:
        return []

    def visit(
        self, it: Iterable, key_fun: Callable = lambda _, i: i
    ) -> dict[Any, FeatureCollector]:
        return {
            key_fun(item, i): FeatureCollector() for i, item in enumerate(it.items())
        }

    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        logging.warning(
            f"NoVisitor is being used. The feature {fte.key} will be ignored"
        )


class ProgramVisitor(Visitor):
    """
    Class aiming at providing a generic program visitor which calls
    the different feature extractor on the appropriate items.
    """

    def __init__(self):
        self._feature_extractors = {}
        self.function_callbacks = []
        self.basic_block_callbacks = []
        self.instruction_callbacks = []
        self.operand_callbacks = []

    def visit_item(self, item: Any, collector: FeatureCollector) -> None:
        """
        Visit a program item according to its type.

        :param item: Can be a Function, Instruction etc..
        :param collector: FeatureCollector to be filled
        """
        if isinstance(item, Function):
            self.visit_function(item, collector)
        elif isinstance(item, BasicBlock):
            self.visit_basic_block(item, collector)
        elif isinstance(item, Instruction):
            self.visit_instruction(item, collector)
        elif isinstance(item, Operand):
            self.visit_operand(item, collector)

    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.

        :param ft: Feature extractor instance
        :return: None
        """
        assert isinstance(fte, FeatureExtractor)
        if isinstance(fte, FunctionFeatureExtractor):
            self.register_function_feature_callback(fte.visit_function)
        if isinstance(fte, BasicBlockFeatureExtractor):
            self.register_basic_block_feature_callback(fte.visit_basic_block)
        if isinstance(fte, InstructionFeatureExtractor):
            self.register_instruction_feature_callback(fte.visit_instruction)
        if isinstance(fte, OperandFeatureExtractor):
            self.register_operand_feature_callback(fte.visit_operand)
        self._feature_extractors[fte.key] = fte

    def register_function_feature_callback(self, callback: Callable) -> None:
        self.function_callbacks.append(callback)

    def register_basic_block_feature_callback(self, callback: Callable) -> None:
        self.basic_block_callbacks.append(callback)

    def register_instruction_feature_callback(self, callback: Callable) -> None:
        self.instruction_callbacks.append(callback)

    def register_operand_feature_callback(self, callback: Callable) -> None:
        self.operand_callbacks.append(callback)

    def visit_function(self, func: Function, collector: FeatureCollector) -> None:
        """
        Visit the given function with the feature extractors registered beforehand.

        :param func: Function to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attacked to a function
        for callback in self.function_callbacks:
            if not func.is_import():
                callback(func, collector)

        # Recursively call visit for all basic blocks
        for bb in func:
            self.visit_basic_block(bb, collector)

    def visit_basic_block(
        self, basic_block: BasicBlock, collector: FeatureCollector
    ) -> None:
        """
        Visit the given basic block with the feature extractors registered beforehand.

        :param basic_block: Basic Block to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attacked to a basic block
        for callback in self.basic_block_callbacks:
            callback(basic_block, collector)

        # Recursively call visit for all instructions
        for inst in basic_block:
            self.visit_instruction(inst, collector)

    def visit_instruction(
        self, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Visit the instruction with the feature extractor registered beforehand.

        :param instruction: Instruction to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an instruction
        for callback in self.instruction_callbacks:
            callback(instruction, collector)

        # Recursively call visit for all operands
        for op in instruction.operands:
            self.visit_operand(op, collector)

    def visit_operand(self, operand: Operand, collector: FeatureCollector) -> None:
        """
        Visit the given operand with the feature extractor registered beforehand.

        :param operand: Operand
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an operand
        for callback in self.operand_callbacks:
            callback(operand, collector)

    @property
    def feature_extractors(self):
        return self._feature_extractors.values()
