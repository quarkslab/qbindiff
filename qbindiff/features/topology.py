from qbindiff.features.visitor import FunctionFeature, Environment
from qbindiff.loader.function import Function


class NbChildren(FunctionFeature):
    """Number of children of the function"""
    name = "nb_children"
    key = "Nbc"

    def visit_function(self, fun: Function, env: Environment):
        n_children = len(fun.children)
        env.add_feature("N_CHILDREN", n_children)


class NbParents(FunctionFeature):
    """Number of parents of the function"""
    name = "nb_parents"
    key = "Nbp"

    def visit_function(self, fun: Function, env: Environment):
        n_parents = len(fun.parents)
        env.add_feature('N_PARENTS', n_parents)


class NbFamily(FunctionFeature):
    """Number of familiy members of the function"""
    name = "nb_family"
    key = "Nbf"

    def visit_function(self, fun: Function, env: Environment):
        n_family = len(fun.parents) + len(fun.children)
        env.add_feature('N_FAMILY', n_family)
