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

"""Generic features
"""

import re
import random
from typing import Any
from re import Pattern

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
    Address of the function as a feature
    """

    key = "addr"

    def visit_function(self, _: Program, function: Function, collector: FeatureCollector) -> None:
        value = function.addr
        collector.add_feature(self.key, value)


class DatName(InstructionFeatureExtractor):
    """
    References to data in the instruction (as retrieved by the backend loader).
    This feature maps the data value to the number of reference occurences to it.
    It's a superset of :py:obj:`StrRef` feature.
    """

    key = "dat"

    def visit_instruction(
        self, _: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        for ref_type, references in instruction.references.items():
            for reference in references:
                if (
                    ref_type == ReferenceType.DATA
                    and reference.type != DataType.UNKNOWN
                    and reference.value is not None
                ):
                    assert isinstance(reference, Data), "DATA reference not referencing Data"
                    collector.add_dict_feature(self.key, {reference.value: 1})

                elif ref_type == ReferenceType.STRUC:
                    assert isinstance(
                        reference, Structure | StructureMember
                    ), "STRUC reference not referencing Structure nor StructureMember"
                    if isinstance(reference, Structure):
                        collector.add_dict_feature(self.key, {reference.name: 1})
                    elif isinstance(reference, StructureMember):
                        collector.add_dict_feature(
                            self.key, {reference.structure.name + "." + reference.name: 1}
                        )

                else:  # Enum, calls
                    pass


class StrRef(InstructionFeatureExtractor):
    """
    References to strings in the instruction.
    This feature maps the string to the number of occurences to it.
    """

    key = "strref"

    def visit_instruction(
        self, _: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        for data in instruction.data_references:
            if data.type == DataType.ASCII:
                collector.add_dict_feature(self.key, {data.value: 1})


class Constant(OperandFeatureExtractor):
    """
    Numeric constant (32/64bits) in the instruction (not addresses).
    This maps numerical values to the number of occurences to it.
    It excludes the addresses (relies on IDA to discriminate them).
    """

    key = "cst"

    def visit_operand(self, _: Program, operand: Operand, collector: FeatureCollector) -> None:
        if operand.is_immediate():
            collector.add_dict_feature(self.key, {str(operand.value): 1})  # This should be a string


class FuncName(FunctionFeatureExtractor):
    """
    Match the function names.
    Optionally the constructor takes a regular expression pattern to exclude function names
    """

    key = "fname"

    def __init__(self, *args: Any, excluded_regex: Pattern[str] | None = None, **kwargs: Any):
        """
        :param args: parameters of a feature extractor
        :param excluded_regex: regex to apply in order to exclude names
        :param kwargs: keyworded arguments
        """
        super(FuncName, self).__init__(*args, **kwargs)
        self._excluded_regex = excluded_regex

    def is_excluded(self, function: Function) -> bool:
        """
        Returns if the function should be excluded (and not considered) based on an optional regex

        :param function: function to consider
        :return: bool
        """
        if self._excluded_regex is None:
            return bool(re.match(rf"^(sub|fun)_0*{function.addr:x}$", function.name, re.IGNORECASE))
        else:
            return bool(self._excluded_regex.match(function.name))

    def visit_function(self, _: Program, function: Function, collector: FeatureCollector) -> None:
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
