# coding: utf-8

from collections import OrderedDict
import time
import logging

# typing imports
from qbindiff.loader.program import Program
from qbindiff.loader.function import Function
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.operand import Operand
from qbindiff.types import ProgramFeatures, FunctionFeatures


class Environment(object):
    """
    Dict wrapper, representing features where the feature name is the key
    and the value is an integer summing the number of occurrence of that
    feature.
    """
    def __init__(self):
        self.features = {}

    def add_feature(self, key: str, value: int):
        self.features[key] = value

    def inc_feature(self, key: str):
        try:
            self.features[key] += 1
        except KeyError:
            self.features[key] = 1


class FeatureExtractor(object):
    """
    Abstract class that represent a feature extractor which sole contraints
    are to define name, key and a function call that is to be called by the
    visitor.
    """
    name = ""
    key = ""

    def call(self, env: Environment, item: object):
        pass


class InstructionFeatureExtractor(FeatureExtractor):
    pass


class FunctionFeatureExtractor(FeatureExtractor):
    pass


class BasicBlockFeatureExtractor(FeatureExtractor):
    pass


class OperandFeatureExtractor(FeatureExtractor):
    pass


class ProgramVisitor(object):
    """
    Class aiming at providing a generic program visitor which calls
    the different feature extractor on the appropriate items.
    """
    def __init__(self):
        self.function_callbacks = []
        self.basic_block_callbacks = []
        self.instruction_callbacks = []
        self.operand_callbacks = []
        self.stats = {}

    def register_feature(self, ft: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.
        :param ft: Feature extractor instance
        :return: None
        """
        if isinstance(ft, InstructionFeatureExtractor):
            self._register_instruction_feature(ft)
        elif isinstance(ft, BasicBlockFeatureExtractor):
            self._register_basic_block_feature(ft)
        elif isinstance(ft, FunctionFeatureExtractor):
            self._register_function_feature(ft)
        elif isinstance(ft, OperandFeatureExtractor):
            self._register_operand_feature(ft)
        else:
            logging.error("[-] invalid feature extractor")

    def _register_function_feature(self, feature_obj: FunctionFeatureExtractor) -> None:
        self.function_callbacks.append(feature_obj)
        self.stats[feature_obj.name] = 0

    def _register_basic_block_feature(self, feature_obj: BasicBlockFeatureExtractor) -> None:
        self.basic_block_callbacks.append(feature_obj)
        self.stats[feature_obj.name] = 0

    def _register_instruction_feature(self, feature_obj: InstructionFeatureExtractor) -> None:
        self.instruction_callbacks.append(feature_obj)
        self.stats[feature_obj.name] = 0

    def _register_operand_feature(self, feature_obj: OperandFeatureExtractor) -> None:
        self.operand_callbacks.append(feature_obj)
        self.stats[feature_obj.name] = 0

    def visit_program(self, program: Program) -> ProgramFeatures:
        """
        Visit the given program with the feature extractor registered beforehand
        with register_feature.
        :param program: program to visit
        :return: ProgramFeatures (dict: Addr-> FunctionFeatures)
        """
        function_features = OrderedDict()
        for fun_addr, fun in program.items():
            function_features[fun_addr] = self.visit_function(fun)
        return function_features

    def visit_function(self, func: Function) -> FunctionFeatures:
        """
        Visit the given function with the feature extractors registered beforehand.
        :param func: Function to visit
        :return: FunctionFeatures, the features of the function
        """
        env = Environment()
        for callback in self.function_callbacks:
            x0 = time.time()
            callback.call(env, func)
            self.stats[callback.name] += time.time() - x0
        for bb_addr, bb in func.items():
            for callback in self.basic_block_callbacks:
                x0 = time.time()
                callback.call(env, bb)
                self.stats[callback.name] += time.time() - x0
            for inst in bb:
                self.visit_instruction(inst, env)
        return env.features

    def visit_instruction(self, instruction: Instruction, env: Environment) -> None:
        """
        Visit the instruction with the feature extractor registered beforehand. The visit
        does not yield new features but update the given environment
        :param instruction: Instruction to visit
        :param env: Environment
        :return: None (perform side effects on the Environment
        """
        for callback in self.instruction_callbacks:
            x0 = time.time()
            callback.call(env, instruction)
            self.stats[callback.name] += time.time() - x0
        if self.operand_callbacks:
            for op in instruction.operands:
                self.visit_operand(op, env)

    def visit_operand(self, operand: Operand, env: Environment) -> None:
        """
        Visit the given operand and update the environment accordingly.
        :param operand: Operand
        :param env: Environment
        :return: None
        """
        for exp in operand.expressions:
            for callback in self.operand_callbacks:
                x0 = time.time()
                callback.call(env, exp, full_op=operand)
                self.stats[callback.name] += time.time() - x0

    def get_stats(self):
        return self.stats
