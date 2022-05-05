from collections import defaultdict

from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Instruction


class MnemonicSimple(InstructionFeatureExtractor):
    """Mnemonic of instructions feature"""

    key = "M"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ):
        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):
    """Mnemonic and type of operand feature"""

    key = "Mt"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ):
        # Use keys as string so it can later be sorted
        op_types = defaultdict(int)
        for op in instruction.operands:
            op_types[str(op.type)] += 1
        op_types[instruction.mnemonic] = 1
        key = hash(frozenset(sorted(op_types.items())))
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):
    """Group of the instruction (FPU, SSE, stack..)"""

    key = "Gp"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ):
        for key in instruction.groups:
            if key not in ["UNDEFINED", "NOTINCS", "NOTINIDA", "DEPRECATED"]:
                collector.add_dict_feature(self.key, {key: 1})
