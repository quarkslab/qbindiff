from qbindiff.features.visitor import InstructionFeatureExtractor


class MnemonicSimple(InstructionFeatureExtractor):
    name = "mnemonic"
    key = "M"

    def call(self, env, instruction):
        env.inc_feature(instruction.mnemonic)

class MnemonicTyped(InstructionFeatureExtractor):
    name = "mnemonic_typed"
    key = "Mt"

    map_table = {'void':'', 'reg':'b', 'mem': 'c', 'phrase': 'd',
                 'displ': 'e', 'imm': 'f', 'far': 'g', 'near': 'h',
                 'idpspec0': 'i', 'idpspec1': 'j', 'idpspec2': 'k',
                 'idpspec3': 'l', 'idpspec4': 'm', 'idpspec5': 'n'}

    def call(self, env, instruction):
        keycode = ''.join(self.map_table[x['type']] for x in instruction.operands)
        env.inc_feature(instruction.mnemonic+keycode)

class GroupsCategory(InstructionFeatureExtractor):
    name = "groups_category"
    key = "Gp"

    def call(self, env, instruction):
        for g in instruction.groups:
            if g not in ['UNDEFINED', 'NOTINCS', 'NOTINIDA', 'DEPRECATED']:
                env.inc_feature(g)
