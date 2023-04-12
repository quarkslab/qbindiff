import re
import random
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
    """
    Address of the function as feature
    """

    key = "addr"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = function.addr
        collector.add_feature(self.key, value)


class DatName(InstructionFeatureExtractor):
    """
    References to data in the instruction. It's a superset of strref
    """

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
    """
    References to strings in the instruction
    """

    key = "strref"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:

        for data in instruction.data_references:
            if data.type == DataType.ASCII:
                collector.add_dict_feature(self.key, {data.value: 1})


class Constant(OperandFeatureExtractor):
    """
    Numeric constant (32/64bits) in the instruction (not addresses)
    """

    key = "cst"

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:

        if operand.is_immutable():
            collector.add_dict_feature(self.key, {str(operand.immutable_value): 1})  # This should be a string


class FuncName(FunctionFeatureExtractor):
    """
    Match the function names. Optionally specify a regular expression pattern to exclude function names
    """

    key = "fname"

    def __init__(
        self, *args: Any, excluded_regex: Optional[Pattern[str]] = None, **kwargs: Any
    ):

        super(FuncName, self).__init__(*args, **kwargs)

        self._excluded_regex = excluded_regex

    def is_excluded(self, function: Function) -> bool:
        """
        Returns if the function should be excluded (and not considered) based on an optional regex

        :param function: function to consider
        :return: bool
        """

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
            # We cannot properly exclude the name since a zero feature vector will
            # have a distance of zero (hence similarity of 1) with any other zero
            # feature vector. Hence, add a good enough random number to reduce the
            # chance of a collision
            collector.add_dict_feature(
                self.key, {function.name + str(random.randrange(1000000000)): 1}
            )
        else:
            collector.add_dict_feature(self.key, {function.name: 1})
