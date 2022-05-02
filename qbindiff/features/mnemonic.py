from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader.instruction import Instruction


class MnemonicSimple(InstructionFeatureExtractor):
    """Mnemonic of instructions feature"""

    name = "mnemonic"
    key = "M"

    def visit_instruction(self, instruction: Instruction, collector: FeatureCollector):
        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):
    """Mnemonic and type of operand feature"""

    name = "mnemonic_typed"
    key = "Mt"

    def visit_instruction(self, instruction: Instruction, collector: FeatureCollector):
        keycode = "".join(str(x.type.value) for x in instruction.operands)
        key = instruction.mnemonic + keycode
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):
    """Group of the instruction (FPU, SSE, stack..)"""

    name = "groups_category"
    key = "Gp"

    def visit_instruction(self, instruction: Instruction, collector: FeatureCollector):
        for key in instruction.groups:
            if key not in ["UNDEFINED", "NOTINCS", "NOTINIDA", "DEPRECATED"]:
                collector.add_dict_feature(self.key, {key: 1})
