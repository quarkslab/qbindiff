import networkx
from typing import Callable, Union

from qbindiff.loader import Function
from qbindiff.loader.types import LoaderType
from qbindiff.loader.backend.binexport import ProgramBackendBinExport


class Program(dict):
    """
    Program class that shoadows the underlying program backend used.
    It inherits from dict which keys are function addresses and
    values are Function object.
    """
    def __init__(self, loader: Union[str, LoaderType] = LoaderType.binexport, *args):
        dict.__init__(self)
        self._backend = None
        loader = LoaderType[loader] if isinstance(loader, str) else loader
        if loader == LoaderType.binexport:
            self.load_binexport(*args)
        elif loader == LoaderType.ida:
            self.load_ida(*args)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)
        self._filter = lambda x: True

    def load_binexport(self,  file_path: str) -> None:
        """
        Load the Program using the binexport backend. This function
        is meant to be used with an empty instanciation: Program()
        :param file_path: File path to the binexport file
        :return: None
        """
        self._backend = ProgramBackendBinExport(self, file_path)

    def load_ida(self) -> None:
        """
        Load the program using the idapython API
        :return: None
        """
        from qbindiff.loader.backend.ida import ProgramBackendIDA
        self._backend = ProgramBackendIDA(self)

    def __repr__(self):
        return '<Program:%s>' % self.name

    @property
    def name(self) -> str:
        """
        Returns the name of the program as defined by the backend
        :return: program name
        """
        return self._backend.name

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
