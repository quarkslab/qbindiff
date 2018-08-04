# coding: utf-8

from collections import OrderedDict
import time
import logging


class Environment(object):
    def __init__(self):
        self.features = {}

    def add_feature(self, key, value):
        self.features[key] = value

    def inc_feature(self, key):
        try:
            self.features[key] += 1
        except KeyError:
            self.features[key] = 1


class FeatureExtractor(object):
    name = ""
    key = ""

    def call(self, env, item):
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

    def __init__(self):
        self.function_callbacks = []
        self.basic_block_callbacks = []
        self.instruction_callbacks = []
        self.operand_callbacks = []
        self.stats = {}

    def register_feature(self, ft: FeatureExtractor):
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

    def _register_function_feature(self, feature_obj):
        self.function_callbacks.append(feature_obj)
        self.stats[feature_obj] = 0

    def _register_basic_block_feature(self, feature_obj):
        self.basic_block_callbacks.append(feature_obj)
        self.stats[feature_obj] = 0

    def _register_instruction_feature(self, feature_obj):
        self.instruction_callbacks.append(feature_obj)
        self.stats[feature_obj] = 0

    def _register_operand_feature(self, feature_obj):
        self.operand_callbacks.append(feature_obj)
        self.stats[feature_obj] = 0

    def visit_program(self, program):
        function_features = OrderedDict()
        for fun_addr, fun in program.items():
            function_features[fun_addr] = self.visit_function(fun)
        return function_features

    def visit_function(self, func):
        env = Environment()
        for callback in self.function_callbacks:
            x0 = time.time()
            callback.call(env, func)
            self.stats[callback] += time.time() - x0
        for bb_addr, bb in func.items():
            for callback in self.basic_block_callbacks:
                x0 = time.time()
                callback.call(env, bb)
                self.stats[callback] += time.time() - x0
            for inst in bb:
                self.visit_instruction(inst, env)
        return env.features

    def visit_instruction(self, instruction, env):
        for callback in self.instruction_callbacks:
            x0 = time.time()
            callback.call(env, instruction)
            self.stats[callback] += time.time() - x0
        if self.operand_callbacks:
            for op in instruction.operands:
                self.visit_operand(op, env)

    def visit_operand(self, operand, env):
        for exp in operand.expressions:
            for callback in self.operand_callbacks:
                callback.call(env, exp, full_op=operand)

    def get_stats(self):
        return self.stats
