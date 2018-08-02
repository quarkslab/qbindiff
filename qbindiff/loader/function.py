import networkx
from qbindiff.loader.backend.qbindiff import FunctionBackendQBinDiff
from qbindiff.loader.backend.binexport import FunctionBackendBinExport
from qbindiff.loader.types import LoaderType, FunctionType
from typing import Set


class Function(dict):
    def __init__(self, loader, *args, **kwargs):
        super(dict, self).__init__()
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data):
        self._backend = FunctionBackendQBinDiff(self, data)

    def load_binexport(self, *args, **kwargs):
        self._backend = FunctionBackendBinExport(self, *args, **kwargs)

    @property
    def addr(self) -> int:
        return self._backend.addr

    @property
    def graph(self) -> networkx.DiGraph:
        return self._backend.graph

    @property
    def parents(self) -> Set[int]:
        return self._backend.parents

    @property
    def children(self) -> Set[int]:
        return self._backend.children

    @property
    def type(self):
        return self._backend.type

    @type.setter
    def type(self, value) -> FunctionType:
        self._backend.type = value

    def is_import(self) -> bool:
        return self._backend.is_import()

    def is_alone(self):
        if self.children:
            return False
        if self.parents:
            return False
        return True

    def __repr__(self):
        return '<Function: 0x%x>' % self.addr
