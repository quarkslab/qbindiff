from collections import OrderedDict

from qbindiff.loader.types import LoaderType
from qbindiff.loader.backend.binexport import ProgramBackendBinExport
from qbindiff.loader.backend.qbindiff import ProgramBackendQBinDiff


class Program(OrderedDict):
    """
    Program class that shoadows the underlying program backend used.
    It inherits from OrderedDict which keys are function addresses and
    values are Function object.
    """
    def __init__(self, loader: str=None, *args):
        OrderedDict.__init__(self)
        self._backend = None
        if loader is not None:
            loader = LoaderType[loader]
            if loader == LoaderType.qbindiff:
                self.load_qbindiff(*args)
            elif loader == LoaderType.binexport:
                self.load_binexport(*args)
            elif loader == LoaderType.ida:
                self.load_ida(*args)
            else:
                raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self,  file_path: str) -> None:
        """
        Load the Program using the binexport backend. This function
        is meant to be used with an empty instanciation: Program()
        :param file_path: File path to the binexport file
        :return: None
        """
        self._backend = ProgramBackendBinExport(self, file_path)

    def load_qbindiff(self, directory, call_graph) -> None:
        """
        Load the Program using the qbindiff backend. This functions is
        meant to be used with a parameter-less instanciation: Program()
        :param directory: directory path which contains all the function JSON files
        :param call_graph: file path to the call graph json file
        :return: None
        """
        self._backend = ProgramBackendQBinDiff(self, directory, call_graph)

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
