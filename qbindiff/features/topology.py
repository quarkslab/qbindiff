from qbindiff.features.visitor import FunctionFeature, FeatureCollector
from qbindiff.loader.function import Function


class ChildNb(FunctionFeature):
    """Number of children of the function"""

    name = "child_nb"
    key = "cnb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(function.children)
        collector.add_feature(self.key, value)


class ParentNb(FunctionFeature):
    """Number of parents of the function"""

    name = "parent_nb"
    key = "pnb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(function.parents)
        collector.add_feature(self.key, value)


class RelativeNb(FunctionFeature):
    """Number of relatives of the function"""

    name = "relative_nb"
    key = "rnb"

    def visit_function(self, function: Function, collector: FeatureCollector):
        value = len(function.parents) + len(function.children)
        collector.add_feature(self.key, value)
