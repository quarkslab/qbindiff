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

"""Features releated to the mnemonic of the instruction
"""

from collections import defaultdict

from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Instruction
from qbindiff.loader.types import ProgramCapability


class PcodeMnemonicSimple(InstructionFeatureExtractor):
    """
    Pcode mnemonic feature.
    It extracts a dictionary with mnemonic as key and 1 as value.
    """

    key = "PM"
    required_capabilities = ProgramCapability.PCODE

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        pcode_count = defaultdict(int)
        pypcode_context = program.pypcode
        try:
            for pcode_ins in instruction.pcode_ops:
                pcode_count[str(pcode_ins.opcode).split(".")[-1]] += 1
        except pypcode.pypcode_native.BadDataError:
            pcode_count = {}

        collector.add_dict_feature(self.key, pcode_count)


class MnemonicSimple(InstructionFeatureExtractor):
    """
    Mnemonic feature.
    It extracts a dictionary with mnemonic as key and 1 as value.
    """

    key = "M"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):
    """
    Typed mnemonic feature.
    It extracts a dictionary where key is a combination of the mnemonic
    and the type of the operands.
    e.g I: immediate, R: Register, thus mov rax, 10, becomes MOVRI.
    Values of the dictionary is 1 if the typed mnemonic is present.
    """

    key = "Mt"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        mnemonic = instruction.mnemonic
        # keep the first letter of the type name as types
        # (ex : mov rsp, 8 will give movri (for register, immediate))
        operands_types = "".join([op.type.name[0] for op in instruction.operands])
        key = mnemonic + operands_types
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):
    """
    Categorization of instructions feature.
    It can correspond to instructions subset (XMM, AES etc..),
    or more generic grouping like (arithmetic, comparisons etc..).
    Requires INSTR_GROUP capability.
    It relies on :py:class:`InstructionGroups` for the different categories.

    .. warning:: As of now there are not many categories. This might change in the future.
    """

    key = "Gp"
    help_msg = """
    Categorization of instructions feature.
    It can correspond to instructions subset (XMM, AES etc..),
    or more generic grouping like (arithmetic, comparisons etc..).
    Requires INSTR_GROUP capability.
    """.strip()
    required_capabilities = ProgramCapability.INSTR_GROUP

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        for key in instruction.groups:
            # Key should be a str, not an int returned by hash
            collector.add_dict_feature(self.key, {key.name: 1})
