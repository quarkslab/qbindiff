import re
from collections import defaultdict
from typing import Optional, Any, Pattern

from qbindiff.features.extractor import (
    FeatureCollector,
    FunctionFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
)
from qbindiff.loader.types import DataType, ReferenceType
from qbindiff.loader import (
    Program,
    Function,
    Instruction,
    Operand,
    Data,
    Structure,
    StructureMember,
)


class Address(FunctionFeatureExtractor):
    """Address of the function as a feature"""

    key = "addr"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = function.addr
        collector.add_feature(self.key, value)


class DatName(InstructionFeatureExtractor):
    """References to data in the instruction. It's a superset of strref"""

    key = "dat"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        for ref_type, references in instruction.references.items():
            for reference in references:
                if (
                    ref_type == ReferenceType.DATA
                    and reference.type != DataType.UNKNOWN
                    and reference.value is not None
                ):
                    assert isinstance(
                        reference, Data
                    ), "DATA reference not referencing Data"
                    collector.add_dict_feature(self.key, {reference.value: 1})

                elif ref_type == ReferenceType.STRUC:
                    assert isinstance(
                        reference, Structure | StructureMember
                    ), "STRUC reference not referencing Structure nor StructureMember"
                    if isinstance(reference, Structure):
                        collector.add_dict_feature(self.key, {reference.name: 1})
                    elif isinstance(reference, StructureMember):
                        collector.add_dict_feature(
                            self.key,
                            {reference.structure.name + "." + reference.name: 1},
                        )

                else:  # Enum, calls
                    pass


class StrRef(InstructionFeatureExtractor):
    """References to strings in the instruction"""

    key = "strref"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        for data in instruction.data_references:
            if data.type == DataType.ASCII:
                collector.add_dict_feature(self.key, {data.value: 1})


class Constant(OperandFeatureExtractor):
    """Numeric constant (32/64bits) in the instruction (not addresses)"""

    key = "cst"

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        if operand.is_immutable():
            collector.add_dict_feature(self.key, {operand.immutable_value: 1})


class FuncName(FunctionFeatureExtractor):
    """Match the function names"""

    key = "fname"

    def __init__(
        self, *args: Any, excluded_regex: Optional[Pattern[str]] = None, **kwargs: Any
    ):
        """Optionally specify a regular expression pattern to exclude function names"""
        super(FuncName, self).__init__(*args, **kwargs)

        self._excluded_regex = excluded_regex

    def is_excluded(self, function: Function) -> bool:
        if self._excluded_regex is None:
            return bool(
                re.match(
                    rf"^(sub|fun)_0*{function.addr:x}$", function.name, re.IGNORECASE
                )
            )
        else:
            return bool(self._excluded_regex.match(function.name))

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        if self.is_excluded(function):
            return
        collector.add_dict_feature(self.key, {function.name: 1})
