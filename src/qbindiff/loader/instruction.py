from __future__ import annotations
from functools import cached_property

from qbindiff.loader.backend import AbstractInstructionBackend
from qbindiff.loader import Data, Operand
from qbindiff.loader.types import LoaderType, ReferenceType, ReferenceTarget
from qbindiff.types import Addr


class Instruction:
    """
    Defines an Instruction object that wrap the backend using under the scene.
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
        Load the Instruction using the protobuf data
        :param args: program, function, addr, protobuf index
        :return:
        """
        from qbindiff.loader.backend.binexport import InstructionBackendBinExport

        self._backend = InstructionBackendBinExport(*args, **kwargs)

    def load_ida(self, addr) -> None:
        """
        Load the Instruction using the IDA backend, (only applies when running in IDA)
        :param addr: Address of the instruction
        :return: None
        """
        from qbindiff.loader.backend.ida import InstructionBackendIDA

        self._backend = InstructionBackendIDA(addr)

    def load_quokka(self, *args, **kwargs) -> None:
        """Load the Instruction using the Quokka backend"""
        from qbindiff.loader.backend.quokka import InstructionBackendQuokka

        self._backend = InstructionBackendQuokka(*args, **kwargs)

    @staticmethod
    def from_backend(backend: AbstractInstructionBackend) -> Instruction:
        """Load the Instruction from an instanciated instruction backend object"""
        return Instruction(None, backend=backend)

    @property
    def addr(self) -> int:
        """
        Returns the address of the instruction
        :return: addr
        """
        return self._backend.addr

    @property
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        :return: mnemonic as a string
        """
        return self._backend.mnemonic

    @cached_property
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """Returns all the references towards the instruction"""
        return self._backend.references

    @property
    def data_references(self) -> list[Data]:
        """Returns the list of data that are referenced by the instruction"""
        if self.references == {}:
            return {}
        else :
            return self.references[ReferenceType.DATA]

    @cached_property
    def operands(self) -> list[Operand]:
        """
        Returns the list of operands as Operand object.
        :return: list of operands
        """
        return [Operand.from_backend(o) for o in self._backend.operands]

    @property
    def groups(self) -> list[int]:
        """
        Returns a list of groups of this instruction.
        :return: list of groups for the instruction
        """
        return self._backend.groups

    @property
    def id(self) -> int:
        """Return the instruction ID as int"""
        return self._backend.id

    @property
    def comment(self) -> str:
        """
        Comment as set in IDA on the instruction
        :return: comment associated with the instruction in IDA
        """
        return self._backend.comment

    @property
    def bytes(self) -> bytes:
        """Returns the bytes representation of the instruction"""
        return self._backend.bytes

    def __str__(self):
        return "%s %s" % (
            self.mnemonic,
            ", ".join(str(op) for op in self._backend.operands),
        )

    def __repr__(self):
        return "<Inst:%s>" % str(self)
