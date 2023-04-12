from collections import defaultdict

from qbindiff.features.extractor import FunctionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Function


class ChildNb(FunctionFeatureExtractor):
    """
    This feature extracts the number of function children of the considered function, inside a program. 
    """

    key = "cnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = len(function.children)
        collector.add_feature(self.key, value)


class ParentNb(FunctionFeatureExtractor):
    """
    This features extracts the number of function parents of the considered function, inside a program
    """

    key = "pnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = len(function.parents)
        collector.add_feature(self.key, value)


class RelativeNb(FunctionFeatureExtractor):
    """
    This features extracts the number of function relatives of the considered function, inside a program
    """

    key = "rnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = len(function.parents) + len(function.children)
        collector.add_feature(self.key, value)


class LibName(FunctionFeatureExtractor):
    """
    This features extracts a dictionary with the addresses of children function as key and the number of time these children are called if they are library functions
    """

    key = "lib"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_library():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)


class ImpName(FunctionFeatureExtractor):
    """
    This features extracts a dictionary with the addresses of children function as key and the number of time these children are called if they are imported functions
    """

    key = "imp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:

        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_import():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)
