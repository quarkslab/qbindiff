from collections import defaultdict

from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Instruction


class MnemonicSimple(InstructionFeatureExtractor):
    """
    This feature extracts a dictionary with the instruction mnemonic as key and 1 as value
    """
    key = "M"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:

        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):
    """
    This features extracts a dictionary with hash of the mnemonic and operands of the instruction as key, 1 as value
    """
    key = "Mt"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:

        # Use keys as string so it can later be sorted
        op_types = defaultdict(int)
        for op in instruction.operands:
            op_types[str(op.type)] += 1
        op_types[instruction.mnemonic] = 1
        key = str(hash(frozenset(sorted(op_types.items()))))  # Key should be a str, not an int returned by hash
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):
    """
    This feature extracts a dictionary with groups as key and 1 as value
    """

    key = "Gp"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
    
        for key in instruction.groups:
            if key not in ["UNDEFINED", "NOTINCS", "NOTINIDA", "DEPRECATED"]:
                collector.add_dict_feature(self.key, {str(key): 1})  # Key should be a str, not an int returned by hash
