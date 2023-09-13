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

"""Visitor pattern module

This module contains the base abstract class that defines the visitor access
pattern to a GenericGraph as well as its standard implementations.
"""

import tqdm
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand
from qbindiff.features.extractor import (
    FeatureCollector,
    FeatureExtractor,
    FunctionFeatureExtractor,
    BasicBlockFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
)
from qbindiff.utils import is_debug
from qbindiff.types import Graph


class Visitor(metaclass=ABCMeta):
    """
    Abstract class representing interface that a visitor
    must implements to work with a Differ object.
    """

    @property
    @abstractmethod
    def feature_extractors(self) -> list[FeatureExtractor]:
        """
        Returns the list of registered features extractor
        """

        raise NotImplementedError()

    def visit(
        self, graph: Graph, key_fun: Callable[[Any, int], Any] = None
    ) -> dict[Any, FeatureCollector]:
        """
        Function performing the visit on a Graph object by calling visit_item with a
        FeatureCollector meant to be filled.

        :param graph: the Graph to be visited.
        :param key_fun: a function that takes 2 input arguments, namely the current item and
                        the current iteration number, and returns a unique key for that item.
                        If not specified, the iteration number is used.
        :return: A dict in which keys are key_fun(item, i) and values are the FeatureCollector
        """

        # By default use the iteration counter as a unique key
        if not key_fun:
            key_fun = lambda _, i: i

        obj_features = {}
        for i, item in tqdm.tqdm(
            enumerate(graph.items()), total=len(graph), disable=not is_debug()
        ):
            _, node = item
            collector = FeatureCollector()
            self.visit_item(graph, node, collector)
            obj_features[key_fun(item, i)] = collector
        return obj_features

    def visit_item(self, graph: Graph, item: Any, collector: FeatureCollector) -> None:
        """
        Abstract method meant to perform the visit of the item.
        It receives an environment in parameter that is meant to be filled.

        :param graph: the graph that is being visited
        :param item: item to be visited
        :param collector: FeatureCollector to fill during the visit
        """
        raise NotImplementedError()

    @abstractmethod
    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.

        :param fte: Feature extractor instance
        """

        raise NotImplementedError()


class NoVisitor(Visitor):
    """
    Trivial visitor that doesn't traverse the graph
    """

    @property
    def feature_extractors(self) -> list[FeatureExtractor]:
        return []

    def visit(
        self, graph: Graph, key_fun: Callable[[Any, int], Any] = None
    ) -> dict[Any, FeatureCollector]:
        # By default use the iteration counter as a unique key
        if not key_fun:
            key_fun = lambda _, i: i

        return {key_fun(item, i): FeatureCollector() for i, item in enumerate(graph.items())}

    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        logging.warning(f"NoVisitor is being used. The feature {fte.key} will be ignored")


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

    def visit_item(self, program: Program, item: Any, collector: FeatureCollector) -> None:
        """
        Visit a program item according to its type.

        :param graph: The program that is being visited
        :param item: Can be a Function, Instruction etc..
        :param collector: FeatureCollector to be filled
        :return: None
        """
        if isinstance(item, Function):
            self.visit_function(program, item, collector)
        elif isinstance(item, BasicBlock):
            self.visit_basic_block(program, item, collector)
        elif isinstance(item, Instruction):
            self.visit_instruction(program, item, collector)
        elif isinstance(item, Operand):
            self.visit_operand(program, item, collector)

    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.

        :param fte: Feature extractor instance
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
        """
        Feature callback at function granularity

        :param callback: feature callback
        """
        self.function_callbacks.append(callback)

    def register_basic_block_feature_callback(self, callback: Callable) -> None:
        """
        Feature callback at basic block granularity

        :param callback: feature callback
        """
        self.basic_block_callbacks.append(callback)

    def register_instruction_feature_callback(self, callback: Callable) -> None:
        """
        Feature callback at function granularity

        :param callback: feature callback
        """

        self.instruction_callbacks.append(callback)

    def register_operand_feature_callback(self, callback: Callable) -> None:
        """
        Feature callback at function granularity

        :param callback: feature callback
        """

        self.operand_callbacks.append(callback)

    def visit_function(self, program: Program, func: Function, collector: FeatureCollector) -> None:
        """
        Visit the given function with the feature extractors registered beforehand.

        :param program: Program that is being visited
        :param func: Function to visit
        :param collector: FeatureCollector to fill
        """

        # Call all callbacks attacked to a function
        for callback in self.function_callbacks:
            if not func.is_import():
                callback(program, func, collector)

        # Recursively call visit for all basic blocks
        for bb in func:
            self.visit_basic_block(program, bb, collector)

    def visit_basic_block(
        self, program: Program, basic_block: BasicBlock, collector: FeatureCollector
    ) -> None:
        """
        Visit the given basic block with the feature extractors registered beforehand.

        :param program: Program that is being visited
        :param basic_block: Basic Block to visit
        :param collector: FeatureCollector to fill
        """

        # Call all callbacks attacked to a basic block
        for callback in self.basic_block_callbacks:
            callback(program, basic_block, collector)

        # Recursively call visit for all instructions
        for inst in basic_block:
            self.visit_instruction(program, inst, collector)

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Visit the instruction with the feature extractor registered beforehand.

        :param program: Program that is being visited
        :param instruction: Instruction to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an instruction
        for callback in self.instruction_callbacks:
            callback(program, instruction, collector)

        # Recursively call visit for all operands
        for op in instruction.operands:
            self.visit_operand(program, op, collector)

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        """
        Visit the given operand with the feature extractor registered beforehand.

        :param program: Program that is being visited
        :param operand: Operand
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an operand
        for callback in self.operand_callbacks:
            callback(program, operand, collector)

    @property
    def feature_extractors(self):
        return self._feature_extractors.values()
