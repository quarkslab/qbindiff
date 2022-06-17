from __future__ import annotations
from collections.abc import Iterable
from functools import cached_property

from qbindiff.loader.backend import AbstractBasicBlockBackend
from qbindiff.loader import Instruction
from qbindiff.loader.types import LoaderType
from qbindiff.types import Addr


class BasicBlock(Iterable[Instruction]):
    """
    Representation of a binary basic block. This class is an Iterable of Instruction.
    """

    def __init__(self, loader, *args, **kwargs):
        super(BasicBlock, self).__init__()

        self._backend = None
        if loader is None and (backend := kwargs.get("backend")) is not None:
            self._backend = backend  # Load directly from instanciated backend
        elif loader == LoaderType.binexport:
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

        self._backend = BasicBlockBackendQBinExport(*args, **kwargs)

    @staticmethod
    def from_backend(backend: AbstractBasicBlockBackend) -> BasicBlock:
        """Load the BasicBlock from an instanciated basic block backend object"""
        return BasicBlock(None, backend=backend)

    def __iter__(self):
        return self.instructions.__iter__()

    @property
    def addr(self) -> Addr:
        """Address of the basic block"""
        return self._backend.addr

    @cached_property
    def instructions(self) -> list[Instruction]:
        return [Instruction.from_backend(i) for i in self._backend.instructions]
