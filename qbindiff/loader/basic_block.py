from qbindiff.loader.types import LoaderType
from qbindiff.types import Addr


class BasicBlock(list):
    """
    Function representation of a binary basic block. This class is a list of instructions.
    """

    def __init__(self, loader, *args, **kwargs):
        super(BasicBlock, self).__init__()

        self._backend = None
        if loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        elif loader == LoaderType.ida:
            self.load_ida(*args, **kwargs)
        elif loader == LoaderType.qbinexport:
            self.load_qbinexport(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self, *args, **kwargs):
        from qbindiff.loader.backend.binexport import BasicBlockBackendBinExport

        self._backend = BasicBlockBackendBinExport(self, *args, **kwargs)

    def load_ida(self, addr):
        raise NotImplementedError("Ida backend loader is not yet fully implemented")

    def load_qbinexport(self, *args, **kwargs):
        from qbindiff.loader.backend.qbinexport import BasicBlockBackendQBinExport

        self._backend = BasicBlockBackendQBinExport(self, *args, **kwargs)

    @property
    def addr(self) -> Addr:
        """Address of the basic block"""
        return self._backend.addr
