from collections import OrderedDict

from qbindiff.loader.types import LoaderType
from qbindiff.loader.backend.binexport import ProgramBackendBinExport
from qbindiff.loader.backend.qbindiff import ProgramBackendQBinDiff


class Program(OrderedDict):
    def __init__(self, loader: LoaderType=None, *args):
        super(dict, self).__init__()
        self._backend = None
        if loader is not None:
            if loader == LoaderType.qbindiff:
                self.load_qbindiff(*args)
            elif loader == LoaderType.binexport:
                self.load_binexport(*args)
            else:
                raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self,  file_path) -> None:
        self._backend = ProgramBackendBinExport(self, file_path)

    def load_qbindiff(self, directory, call_graph) -> None:
        self._backend = ProgramBackendQBinDiff(self, directory, call_graph)

    def __repr__(self):
        return '<Program:%s>' % self.name

    @property
    def name(self):
        return self._backend.name
