from collections import defaultdict

from qbindiff.features.extractor import FunctionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Function


class ChildNb(FunctionFeatureExtractor):
    """Number of children of the function"""

    key = "cnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(function.children)
        collector.add_feature(self.key, value)


class ParentNb(FunctionFeatureExtractor):
    """Number of parents of the function"""

    key = "pnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(function.parents)
        collector.add_feature(self.key, value)


class RelativeNb(FunctionFeatureExtractor):
    """Number of relatives of the function"""

    key = "rnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = len(function.parents) + len(function.children)
        collector.add_feature(self.key, value)


class LibName(FunctionFeatureExtractor):
    """Call to library functions (local function)"""

    key = "lib"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_library():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)


class ImpName(FunctionFeatureExtractor):
    """Call to imported functions"""

    key = "imp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ):
        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_import():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)
