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
        """
        Load the operand using the quokka backend
        
        :return: None
        """

        from qbindiff.loader.backend.quokka import OperandBackendQuokka

        self._backend = OperandBackendQuokka(*args, **kwargs)

    @staticmethod
    def from_backend(backend: AbstractOperandBackend) -> Operand:
        """
        Load the Operand from an instanciated operand backend object
        """

        return Operand(None, backend=backend)

    @property
    def type(self) -> int:
        """
        The operand type as int as defined in the IDA API.
        Example : 1 corresponds to a register (ex: rax)
        """

        return self._backend.type

    @property
    def value(self) -> int | None:
        """
        The immediate value (not addresses) used by the operand.
        If not returns None.
        """

        if self.is_immediate():
            return self._backend.value
        return None

    def is_immediate(self) -> bool:
        """
        Whether the operand is an immediate (not considering addresses)
        """

        return self._backend.is_immediate()

    def __str__(self) -> str:
        return str(self._backend)

    def __repr__(self) -> str:
        return "<Op:%s>" % str(self)
