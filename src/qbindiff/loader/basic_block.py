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

"""Basic Block
"""

from __future__ import annotations
from collections.abc import Iterable
from functools import cached_property

from qbindiff.loader.backend import AbstractBasicBlockBackend
from qbindiff.loader import Instruction
from qbindiff.types import Addr


class BasicBlock(Iterable[Instruction]):
    """
    Representation of a binary basic block.
    This class is an Iterable of Instruction.
    """

    def __init__(self, backend: AbstractBasicBlockBackend):
        super(BasicBlock, self).__init__()

        self._backend = backend  # Load directly from instanciated backend

    @staticmethod
    def from_backend(backend: AbstractBasicBlockBackend) -> BasicBlock:
        """
        Load the BasicBlock from an instanciated basic block backend object

        :param backend: backend to use
        :return: the loaded basic block
        """
        return BasicBlock(backend)

    def __iter__(self):
        return self.instructions.__iter__()

    def __len__(self):
        return len(self._backend)

    @property
    def addr(self) -> Addr:
        """
        Address of the basic block
        """
        return self._backend.addr

    @cached_property
    def instructions(self) -> list[Instruction]:
        """
        List of Instruction objects of the basic block
        """
        return [Instruction.from_backend(i) for i in self._backend.instructions]

    @property
    def bytes(self) -> bytes:
        """
        Raw bytes of basic block instructions.
        """
        return self._backend.bytes
