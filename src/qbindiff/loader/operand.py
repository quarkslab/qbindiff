from __future__ import annotations

from qbindiff.loader.backend import AbstractOperandBackend
from qbindiff.loader.types import LoaderType


class Operand:
    """
    Represent an operand object which hide the underlying backend implementation
    """

    def __init__(self, loader: LoaderType | None, /, *args, **kwargs):
        self._backend = None
        if loader is None and (backend := kwargs.get("backend")) is not None:
            self._backend = backend  # Load directly from instanciated backend
        elif loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        elif loader == LoaderType.ida:
            self.load_ida(*args, **kwargs)
        elif loader == LoaderType.quokka:
            self.load_quokka(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self, *args, **kwargs) -> None:
        """
        Load the operand using the data of the binexport file
        :param args: program, function, and operand index
        :return: None
        """
        from qbindiff.loader.backend.binexport import OperandBackendBinexport

        self._backend = OperandBackendBinexport(*args, **kwargs)

    def load_ida(self, op_t, ea) -> None:
        """
        Instanciate the operand using IDA API
        :param op_t: op_t* as defined in the IDA SDK
        :param ea: address of the instruction
        :return: None
        """
        from qbindiff.loader.backend.ida import OperandBackendIDA

        self._backend = OperandBackendIDA(op_t, ea)

    def load_quokka(self, *args, **kwargs) -> None:
        """Load the operand using the quokka backend"""
        from qbindiff.loader.backend.quokka import OperandBackendQuokka

        self._backend = OperandBackendQuokka(*args, **kwargs)

    @staticmethod
    def from_backend(backend: AbstractOperandBackend) -> Operand:
        """Load the Operand from an instanciated operand backend object"""
        return Operand(None, backend=backend)

    @property
    def type(self) -> int:
        """Returns the operand type as int : 1 corresponds to a register (ex: rax), 2 to an immediate (ex: 8) and 3 to a memory access (ex : [...])"""
        return self._backend.type

    @property
    def immutable_value(self) -> int | None:
        """
        Returns the immutable value (not addresses) used by the operand.
        If there is no immutable value then returns None.
        """
        if self.is_immutable():
            return self._backend.immutable_value
        return None

    def is_immutable(self) -> bool:
        """Returns whether the operand is an immutable (not considering addresses)"""
        return self._backend.is_immutable()

    def __str__(self):
        return str(self._backend)

    def __repr__(self):
        return "<Op:%s>" % str(self)
