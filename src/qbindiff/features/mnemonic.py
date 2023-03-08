from collections import defaultdict

from qbindiff.features.extractor import InstructionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Instruction


class MnemonicSimple(InstructionFeatureExtractor):

    key = "M"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Mnemonic of instructions feature

        :param program: program to consider
        :param instruction: instruction of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        collector.add_dict_feature(self.key, {instruction.mnemonic: 1})


class MnemonicTyped(InstructionFeatureExtractor):

    key = "Mt"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Mnemonic and type of operand feature

        :param program: program to consider
        :param instruction: instruction of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        # Use keys as string so it can later be sorted
        op_types = defaultdict(int)
        for op in instruction.operands:
            op_types[str(op.type)] += 1
        op_types[instruction.mnemonic] = 1
        key = str(hash(frozenset(sorted(op_types.items()))))  # Key should be a str, not an int returned by hash
        collector.add_dict_feature(self.key, {key: 1})


class GroupsCategory(InstructionFeatureExtractor):

    key = "Gp"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Group of the instruction (FPU, SSE, stack..)

        :param program: program to consider
        :param instruction: instruction of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        for key in instruction.groups:
            if key not in ["UNDEFINED", "NOTINCS", "NOTINIDA", "DEPRECATED"]:
                collector.add_dict_feature(self.key, {str(key): 1})  # Key should be a str, not an int returned by hash
