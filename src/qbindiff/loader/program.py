import networkx
from typing import Callable, Union, Any
from collections.abc import Iterator

from qbindiff.abstract import GenericGraph
from qbindiff.loader import Function
from qbindiff.loader.types import LoaderType
from qbindiff.types import Addr


class Program(dict, GenericGraph):
    """
    Program class that shadows the underlying program backend used.
    It inherits from dict which keys are function addresses and
    values are Function object.
    """

    def __init__(
        self,
        file_path: str = None,
        loader: LoaderType = LoaderType.binexport,
        exec_path: str = None,
    ):
        super(Program, self).__init__()
        self._backend = None
        self._file_path = file_path
        self._exec_path = exec_path

        if file_path is None:  # Inside IDA just call Program()
            from qbindiff.loader.backend.ida import ProgramBackendIDA

            self._backend = ProgramBackendIDA(self)

        elif loader == LoaderType.binexport:
            from qbindiff.loader.backend.binexport import ProgramBackendBinExport

            self._backend = ProgramBackendBinExport(self, file_path)

        elif loader == LoaderType.qbinexport:
            from qbindiff.loader.backend.qbinexport import ProgramBackendQBinExport

            self._backend = ProgramBackendQBinExport(self, file_path, exec_path)

        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)
        self._filter = lambda x: True

    @staticmethod
    def from_binexport(file_path: str) -> "Program":
        """
        Load the Program using the binexport backend. This function
        is meant to be used with an empty instanciation: Program()
        :param file_path: File path to the binexport file
        :return: None
        """
        return Program(file_path, LoaderType.binexport)

    @staticmethod
    def from_ida() -> "Program":
        """
        Load the program using the idapython API
        :return: None
        """
        return Program()

    def __repr__(self):
        return "<Program:%s>" % self.name

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Return an iterator over the items. Each item is {node_label: node}"""
        for addr in self.keys():
            f = self[addr]
            if self._filter(f):  # yield function only if filter agree to keep it
                yield (addr, f)

    def get_node(self, node_label: Any):
        """Returns the node identified by the `node_label`"""
        return self[node_label]

    @property
    def node_labels(self) -> Iterator[Any]:
        """Return an iterator over the node labels"""
        for addr in self.keys():
            if self._filter(self[addr]):
                yield addr

    @property
    def nodes(self) -> Iterator[Any]:
        """Return an iterator over the nodes"""
        yield from self.__iter__()

    @property
    def edges(self) -> Iterator[tuple[Any, Any]]:
        """
        Return an iterator over the edges.
        An edge is a pair (node_label_a, node_label_b)
        """
        return self.callgraph.edges

    @property
    def name(self) -> str:
        """
        Returns the name of the program as defined by the backend
        :return: program name
        """
        return self._backend.name

    @property
    def file_path(self):
        """Returns the file path"""
        return self._file_path

    @property
    def exec_path(self):
        """Returns the executable path"""
        return self._exec_path

    def set_function_filter(self, func: Callable[[Function], bool]) -> None:
        """
        Filter out some functions, to ignore them in later processing.

        .. warning: The filter only apply for __iter__ function and callgraph property.
                    Accessing functions through the dictionary does not apply the filter

        :param func: function take an Function object and returns whether or not to keep it
        :return: None
        """
        self._filter = func

    def __iter__(self):
        """
        Override the built-in __iter__ to iterate all functions
        located in the program.

        :return: Iterator of all functions (sorted by address)
        """
        for addr in sorted(self.keys()):
            f = self[addr]
            if self._filter(f):  # yield function only if filter agree to keep it
                yield f

    @property
    def callgraph(self) -> networkx.DiGraph:
        cg = self._backend.callgraph
        funcs = list(self)  # functions already filtered
        return cg.subgraph([x.addr for x in funcs])

    def remove_function(self, addr: Addr) -> None:
        """Remove a function from the callgraph"""
        func = self[addr]
        self.pop(addr)
        for p_addr in func.parents:
            self[p_addr].children.remove(addr)
            self._backend.callgraph.remove_edge(p_addr, addr)
        for c_addr in func.children:
            self[c_addr].parents.remove(addr)
            self._backend.callgraph.remove_edge(addr, c_addr)
        self._backend.callgraph.remove_node(addr)
