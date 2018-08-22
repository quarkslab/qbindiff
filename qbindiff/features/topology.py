import networkx
import community
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

class NbGChildren(FunctionFeatureExtractor):
    """Number of small children of the function"""
    name = "nb_grand_children"
    key = "Nbgc"

    def call(self, env: Environment, fun: Function):
        n_g_children = 0
        for child in fun.children:
            n_g_children += len(child.children)
        env.add_feature('N_G_CHILDREN', n_g_children)


class NbGParents(FunctionFeatureExtractor):
    """Number of grand parents of the function"""
    name = "nb_grand_parents"
    key = "Nbgp"

    def call(self, env: Environment, fun: Function):
        n_g_parents = 0
        for parent in fun.parents:
            n_g_parents += len(parent.parents)
        env.add_feature('N_G_PARENTS', n_g_parents)

