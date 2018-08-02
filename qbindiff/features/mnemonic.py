from qbindiff.features.visitor import InstructionFeatureExtractor


class MnemonicSimple(InstructionFeatureExtractor):
    name = "mnemonic"
    key = "M"

    def call(self, env, instruction):
        env.inc_feature(instruction.mnemonic)


class MnemonicTyped(InstructionFeatureExtractor):
    name = "mnemonic_typed"
    key = "Mt"

    def call(self, env, instruction):
        keycode = ''.join(str(x.type.value) for x in instruction.operands)
        env.inc_feature(instruction.mnemonic+keycode)


class GroupsCategory(InstructionFeatureExtractor):
    name = "groups_category"
    key = "Gp"

    def call(self, env, instruction):
        for g in instruction.groups:
            if g not in ['UNDEFINED', 'NOTINCS', 'NOTINIDA', 'DEPRECATED']:
                env.inc_feature(g)
