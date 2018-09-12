import networkx
from qbindiff.loader.types import LoaderType, FunctionType

from typing import Set
from qbindiff.types import Addr


class Function(dict):
    """
    Function representation of a binary function. This class is a dict
    of basic block addreses to the basic block (list of instruction).
    """
    def __init__(self, loader, *args, **kwargs):
        super(dict, self).__init__()
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        elif loader == LoaderType.ida:
            self.load_ida(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data):
        from qbindiff.loader.backend.qbindiff import FunctionBackendQBinDiff
        self._backend = FunctionBackendQBinDiff(self, data)

    def load_binexport(self, *args, **kwargs):
        from qbindiff.loader.backend.binexport import FunctionBackendBinExport
        self._backend = FunctionBackendBinExport(self, *args, **kwargs)

    def load_ida(self, addr):
        from qbindiff.loader.backend.ida import FunctionBackendIDA
        self._backend = FunctionBackendIDA(self, addr)

    @property
    def edges(self):
        return list(self.graph.edges)

    @property
    def addr(self) -> Addr:
        """
        Address of the function
        :return: addr
        """
        return self._backend.addr

    @property
    def graph(self) -> networkx.DiGraph:
        """
        Gives the networkx DiGraph of the function. This is used to perform networkx
        based algorithm.
        :return: directed graph of the function
        """
        return self._backend.graph

    @property
    def parents(self) -> Set[Addr]:
        """
        Set of function parents in the call graph. Thus
        functions that calls this function
        :return: caller functions
        """
        return self._backend.parents

    @property
    def children(self) -> Set[Addr]:
        """
        Set of functions called by this function in the
        call graph.
        :return: callee functions
        """
        return self._backend.children

    @property
    def type(self) -> FunctionType:
        """
        Returns the type of the instruction (as defined by IDA)
        :return: function type
        """
        return self._backend.type

    @type.setter
    def type(self, value) -> None:
        """ Set the type value """
        self._backend.type = value

    def is_import(self) -> bool:
        """
        Returns whether or not this function is an import function.
        (Thus not having content)
        :return: bool
        """
        return self._backend.is_import()

    def is_alone(self):
        """
        Returns whether or not the function have neither
        caller nor callee.
        :return: bool
        """
        if self.children:
            return False
        if self.parents:
            return False
        return True

    def __repr__(self):
        return '<Function: 0x%x>' % self.addr
