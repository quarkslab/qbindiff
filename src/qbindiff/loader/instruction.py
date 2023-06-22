from __future__ import annotations
from functools import cached_property

from qbindiff.loader.backend import AbstractInstructionBackend
from qbindiff.loader import Data, Operand
from qbindiff.loader.types import ReferenceType, ReferenceTarget
from qbindiff.types import Addr
from typing import List, Dict


class Instruction:
    """
    Defines an Instruction object that wrap the backend using under the scene.
    """

    def __init__(self, backend: AbstractInstructionBackend):
        self._backend = backend  # Load directly from instanciated backend

    @staticmethod
    def from_backend(backend: AbstractInstructionBackend) -> Instruction:
        """
        Load the Instruction from an instanciated instruction backend object
        """

        return Instruction(backend)

    @property
    def addr(self) -> int:
        """
        Returns the address of the instruction
        """

        return self._backend.addr

    @property
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        """

        return self._backend.mnemonic

    @cached_property
    def references(self) -> Dict[ReferenceType, List[ReferenceTarget]]:
        """
        Returns all the references towards the instruction
        """

        return self._backend.references

    @property
    def data_references(self) -> List[Data]:
        """
        Returns the list of data that are referenced by the instruction

        .. warning::
           The BinExport backend tends to return empty references and so are data references

        """
        if ReferenceType.DATA in self.references:
            return self.references[ReferenceType.DATA]
        else:
            return []

    @cached_property
    def operands(self) -> List[Operand]:
        """
        Returns the list of operands as Operand object.
        """

        return [Operand.from_backend(o) for o in self._backend.operands]

    @property
    def groups(self) -> List[int]:
        """
        Returns a list of groups of this instruction.
        """

        return self._backend.groups

    @property
    def id(self) -> int:
        """
        Return the instruction ID as int
        """

        return self._backend.id

    @property
    def comment(self) -> str:
        """
        Comment as set in IDA on the instruction
        """

        return self._backend.comment

    @property
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the instruction
        """

        return self._backend.bytes

    def __str__(self) -> str:
            
        return "%s %s" % (
            self.mnemonic,
            ", ".join(str(op) for op in self._backend.operands),
        )

    def __repr__(self) -> str:
        return "<Inst:%s>" % str(self)
