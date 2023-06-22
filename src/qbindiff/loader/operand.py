from __future__ import annotations

from qbindiff.loader.backend import AbstractOperandBackend


class Operand:
    """
    Represent an operand object which hide the underlying backend implementation
    """

    def __init__(self, backend: AbstractOperandBackend):
        self._backend = backend  # Load directly from instanciated backend

    @staticmethod
    def from_backend(backend: AbstractOperandBackend) -> Operand:
        """
        Load the Operand from an instanciated operand backend object
        """

        return Operand(backend)

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
