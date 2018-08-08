from qbindiff.features.visitor import InstructionFeatureExtractor, Environment
from qbindiff.loader.instruction import Instruction


class MnemonicSimple(InstructionFeatureExtractor):
    """Mnemonic of instructions feature"""
    name = "mnemonic"
    key = "M"

    def call(self, env: Environment, instruction: Instruction):
        env.inc_feature(instruction.mnemonic)


class MnemonicTyped(InstructionFeatureExtractor):
    """Mnemonic and type of operand feature"""
    name = "mnemonic_typed"
    key = "Mt"

    def call(self, env: Environment, instruction: Instruction):
        keycode = ''.join(str(x.type.value) for x in instruction.operands)
        env.inc_feature(instruction.mnemonic+keycode)


class GroupsCategory(InstructionFeatureExtractor):
    """Group of the instruction (FPU, SSE, stack..)"""
    name = "groups_category"
    key = "Gp"

    def call(self, env: Environment, instruction: Instruction):
        for g in instruction.groups:
            if g not in ['UNDEFINED', 'NOTINCS', 'NOTINIDA', 'DEPRECATED']:
                env.inc_feature(g)
