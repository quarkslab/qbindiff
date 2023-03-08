from collections import defaultdict

from qbindiff.features.extractor import FunctionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Function


class ChildNb(FunctionFeatureExtractor):

    key = "cnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Number of children of the function

        :param program: program to consider
        :param function: function of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        value = len(function.children)
        collector.add_feature(self.key, value)


class ParentNb(FunctionFeatureExtractor):

    key = "pnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Number of parents of the function

        :param program: program to consider
        :param function: function of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        value = len(function.parents)
        collector.add_feature(self.key, value)


class RelativeNb(FunctionFeatureExtractor):

    key = "rnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Number of relatives of the function

        :param program: program to consider
        :param function: function of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        value = len(function.parents) + len(function.children)
        collector.add_feature(self.key, value)


class LibName(FunctionFeatureExtractor):

    key = "lib"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Call to library functions (local function)

        :param program: program to consider
        :param function: function of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_library():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)


class ImpName(FunctionFeatureExtractor):

    key = "imp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        """
        Call to imported functions

        :param program: program to consider
        :param function: function of the program from which we want to extract the feature
        :param collector: collector to register features
        :return: None
        """

        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_import():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)
