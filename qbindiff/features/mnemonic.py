from qbindiff.features.visitor import InstructionFeature, Environment
from qbindiff.loader.instruction import Instruction


class MnemonicSimple(InstructionFeature):
    """Mnemonic of instructions feature"""
    name = "mnemonic"
    key = "M"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        key = instruction.mnemonic
        env.inc_feature(key)


class MnemonicTyped(InstructionFeature):
    """Mnemonic and type of operand feature"""
    name = "mnemonic_typed"
    key = "Mt"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        keycode = ''.join(str(x.type.value) for x in instruction.operands)
        key = instruction.mnemonic+keycode
        env.inc_feature(key)


class GroupsCategory(InstructionFeature):
    """Group of the instruction (FPU, SSE, stack..)"""
    name = "groups_category"
    key = "Gp"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        for key in instruction.groups:
            if key not in ['UNDEFINED', 'NOTINCS', 'NOTINIDA', 'DEPRECATED']:
                env.inc_feature(key)

