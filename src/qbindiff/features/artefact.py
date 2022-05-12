from collections import defaultdict

from qbindiff.features.extractor import (
    FeatureCollector,
    FunctionFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
)
from qbindiff.loader import Program, Function, Instruction, Operand


class Address(FunctionFeatureExtractor):
    """Address of the function as a feature"""

    key = "addr"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = function.addr
        collector.add_feature(self.key, value)


class DatName(InstructionFeatureExtractor):
    """References to data in the instruction"""

    key = "dat"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        value = defaultdict(int)
        for addr in instruction.data_references:
            value[addr] += 1
        collector.add_dict_feature(self.key, value)


class Constant(OperandFeatureExtractor):
    """Constant (32/64bits) in the instruction (not addresses)"""

    key = "cst"

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        if operand.type == 2:  # capstone.x86.X86_OP_IMM
            collector.add_dict_feature(self.key, {operand.value.imm: 1})


class FuncName(FunctionFeatureExtractor):
    """Match the function names"""

    key = "fname"

    def __init__(self, *args, excluded_prefix: tuple[str] = None, **kwargs):
        """Optionally specify a set of excluded prefix when matching the names"""
        super(FuncName, self).__init__(*args, **kwargs)

        if excluded_prefix is None:
            self._excluded_prefix = ("sub_", "SUB_")
        else:
            self._excluded_prefix = excluded_prefix

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        name_len = len(function.name)
        for prefix in self._excluded_prefix:
            l = min(len(prefix), name_len)
            if prefix[:l] == function.name[:l]:
                return
        collector.add_dict_feature(self.key, {function.name: 1})
