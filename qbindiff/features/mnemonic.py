from qbindiff.features.visitor import InstructionFeature, Environment
from qbindiff.loader.instruction import Instruction

import os
DIRNAME = os.path.dirname(__file__)

with open('{}/feature_index.json'.format(DIRNAME)) as file:
    MENMONIC_INDEX = json.load(file)
with open('{}/feature_index.json'.format(DIRNAME)) as file:
    OPERAND_INDEX = json.load(file)
with open('{}/feature_index.json'.format(DIRNAME)) as file:
    INSTRUCTION_INDEX = json.load(file)


class MnemonicSimple(InstructionFeature):
    """Mnemonic of instructions feature"""
    name = "mnemonic"
    key = "M"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        env.inc_feature(instruction.mnemonic)


class MnemonicTyped(InstructionFeature):
    """Mnemonic and type of operand feature"""
    name = "mnemonic_typed"
    key = "Mt"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        keycode = ''.join(str(x.type.value) for x in instruction.operands)
        env.inc_feature(instruction.mnemonic+keycode)


class GroupsCategory(InstructionFeature):
    """Group of the instruction (FPU, SSE, stack..)"""
    name = "groups_category"
    key = "Gp"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        for g in instruction.groups:
            if g not in ['UNDEFINED', 'NOTINCS', 'NOTINIDA', 'DEPRECATED']:
                env.inc_feature(g)


class MnemonicTyped(InstructionFeature):
    """Group of the instruction (FPU, SSE, stack..)"""
    name = "mnemonic_typed"
    key = "mt"

    def visit_instruction(self, instruction: Instruction, env: Environment):
        mnemonic = MNEMONIC_INDEX[instruction.mnemonic]
        operands = ', '.join(OPERAND_INDEX[instruction.operand])
        value = INSTRUCTION_INDEX[' '.join((mnemonic, operands))]
        env.inc_feature(value)
