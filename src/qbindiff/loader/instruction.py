# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instruction
"""

from __future__ import annotations
from functools import cached_property

from qbindiff.loader.backend import AbstractInstructionBackend
from qbindiff.loader import Data, Operand
from qbindiff.loader.types import ReferenceType, ReferenceTarget
from qbindiff.types import Addr


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
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """
        Returns all the references towards the instruction
        """

        return self._backend.references

    @property
    def data_references(self) -> list[Data]:
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
    def operands(self) -> list[Operand]:
        """
        Returns the list of operands as Operand object.
        """

        return [Operand.from_backend(o) for o in self._backend.operands]

    @property
    def groups(self) -> list[int]:
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
