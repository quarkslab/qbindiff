from collections import defaultdict

from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Instruction


class MnemonicSimple(InstructionFeatureExtractor):
    """
    Mnemonic feature.
    It extracts a dictionary with mnemonic as key and 0 or 1 as value.
    """

    key = "M"

    def visit_instruction(self, program: Program, instruction: Instruction, collector: FeatureCollector) -> None:
        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):
    """
    Typed mnemonic feature.
    It extracts a dictionary where key is the mnemonic and the type of operands
    e.g I: immediate, R: Register, thus mov rax, 10, becomes MOVRI.
    Values of the dictionary is 0 or 1 if the typed mnemonic is present.
    """
    
    key = "Mt"

    def visit_instruction(self, program: Program, instruction: Instruction, collector: FeatureCollector) -> None:
        # Use keys as string so it can later be sorted
        # FIXME: Make it way simpler
        op_types = defaultdict(int)
        for op in instruction.operands:
            op_types[str(op.type)] += 1
        op_types[instruction.mnemonic] = 1
        key = str(hash(frozenset(sorted(op_types.items()))))  # Key should be a str, not an int returned by hash
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):
    """
    Categorization of instructions feature.
    It can correspond to instructions subset (XMM, AES etc..),
    or more generic grouping like (arithmetic, comparisons etc..).
    As of now, rely on capstone groups.

    .. warning:: Feature in maintenance. Do nothing at the moment.
    """

    key = "Gp"

    def visit_instruction(self, program: Program, instruction: Instruction, collector: FeatureCollector) -> None:
        for key in instruction.groups:
            if key not in ["UNDEFINED", "NOTINCS", "NOTINIDA", "DEPRECATED"]:
                collector.add_dict_feature(self.key, {str(key): 1})  # Key should be a str, not an int returned by hash
