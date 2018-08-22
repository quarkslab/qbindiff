from qbindiff.features.visitor import FunctionFeatureExtractor, Environment
from qbindiff.loader.function import Function


class NbChildren(FunctionFeatureExtractor):
    """Number of children of the function"""
    name = "nb_children"
    key = "Nbc"

    def call(self, env: Environment, fun: Function):
        n_children = len(fun.children)
        env.add_feature("N_CHILDREN", n_children)


class NbParents(FunctionFeatureExtractor):
    """Number of parents of the function"""
    name = "nb_parents"
    key = "Nbp"

    def call(self, env: Environment, fun: Function):
        n_parents = len(fun.parents)
        env.add_feature('N_PARENTS', n_parents)


class NbFamily(FunctionFeatureExtractor):
    """Number of familiy members of the function"""
    name = "nb_family"
    key = "Nbf"

    def call(self, env: Environment, fun: Function):
        n_family = len(fun.parents) + len(fun.children)
        env.add_feature('N_FAMILY', n_family)
