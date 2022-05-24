import re
from collections import defaultdict
from capstone import CS_GRP_JUMP
from typing import Optional, Any, Pattern

from qbindiff.features.extractor import (
    FeatureCollector,
    FunctionFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
)
from qbindiff.loader import Program, Function, Instruction, Operand
from qbindiff.types import DataType


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
        for data in instruction.data_references:
            if data.type != DataType.UNKNOWN and data.value is not None:
                collector.add_dict_feature(self.key, {data.value: 1})


class Constant(InstructionFeatureExtractor):
    """Numeric constant (32/64bits) in the instruction (not addresses)"""

    key = "cst"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        # Ignore jumps since the target is an immutable
        if instruction.capstone.group(CS_GRP_JUMP):
            return
        for operand in instruction.operands:
            if operand.is_immutable():
                collector.add_dict_feature(self.key, {operand.capstone.value.imm: 1})


class FuncName(FunctionFeatureExtractor):
    """Match the function names"""

    key = "fname"

    def __init__(
        self, *args: Any, excluded_regex: Optional[Pattern[str]] = None, **kwargs: Any
    ):
        """Optionally specify a regular expression pattern to exclude function names"""
        super(FuncName, self).__init__(*args, **kwargs)

        if excluded_regex is None:
            self._excluded_regex = re.compile(
                r"^(sub|fun)_[0-9a-f]{1,8}$", re.IGNORECASE
            )
        else:
            self._excluded_regex = excluded_regex

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        name_len = len(function.name)
        if self._excluded_regex.match(function.name):
            return
        collector.add_dict_feature(self.key, {function.name: 1})
