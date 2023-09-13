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

"""Common interface to a binary

This module contains the common interface used by qbindiff to access and interact
with binaries.
The data is being loaded by the backend loaders.
"""

from qbindiff.loader.data import Data
from qbindiff.loader.structure import Structure, StructureMember
from qbindiff.loader.operand import Operand
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.basic_block import BasicBlock
from qbindiff.loader.function import Function
from qbindiff.loader.program import Program
from qbindiff.loader.types import LoaderType

LOADERS = {
    "binexport": LoaderType.binexport,
    "quokka": LoaderType.quokka,
    "ida": LoaderType.ida,
}
