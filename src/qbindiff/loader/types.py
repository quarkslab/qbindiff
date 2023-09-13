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

"""Types
"""

from __future__ import annotations
from enum import IntEnum
from typing import TypeAlias, TYPE_CHECKING
import enum_tools.documentation

if TYPE_CHECKING:
    from qbindiff.loader.data import Data
    from qbindiff.loader.structure import Structure, StructureMember


@enum_tools.documentation.document_enum
class LoaderType(IntEnum):
    """
    Enum of different loaders (supported or not)
    """

    unknown = 0  # doc: unknown loader
    binexport = 1  # doc: binexport loader
    diaphora = 2  # doc: diaphora loader (not supported)
    ida = 3  # doc: IDA loader
    quokka = 4  # doc: Quokka loader


@enum_tools.documentation.document_enum
class OperandType(IntEnum):
    """
    All the operand types as defined by IDA
    """

    unknown = 0  # doc: type is unknown
    register = 1  # doc: register (GPR)
    memory = 2  # doc: Direct memory reference
    phrase = 3  # doc: Memory access with base+reg  or  base + offset * factor
    displacement = 4  # doc: Memory access with base+offset
    immediate = 5  # doc: Immediate value
    far = 6  # doc: Asbolute address
    near = 7  # doc: Relative address
    # TODO: To improve with architectures type specific (reglist,


@enum_tools.documentation.document_enum
class FunctionType(IntEnum):
    """
    Function types as defined by IDA.
    """

    normal = 0  # doc: Normal function
    library = 1  # doc: Function identified as a library one
    imported = 2  # doc: Imported function e.g: function in PLT
    thunk = 3  # doc: Function identified as thunk (trampoline to another one)
    invalid = 4  # doc: Invalid function (not properly disassembled)
    extern = 5  # doc: External symbol (function without content)


@enum_tools.documentation.document_enum
class DataType(IntEnum):
    """
    Types of data
    """

    UNKNOWN = 0  # doc: Data type is unknown
    BYTE = 1  # doc: 1 byte
    WORD = 2  # doc: 2 bytes
    DOUBLE_WORD = 3  # doc: 4 bytes
    QUAD_WORD = 4  # doc: 8 bytes
    OCTO_WORD = 5  # doc: 16 bytes
    FLOAT = 6  # doc: float value
    DOUBLE = 7  # doc: double value
    ASCII = 8  # doc: ASCII string


@enum_tools.documentation.document_enum
class StructureType(IntEnum):
    """
    Different structure types.
    """

    UNKNOWN = 0  # doc: Type unknown
    STRUCT = 1  # doc: Type is structure
    ENUM = 2  # doc: Type is enum
    UNION = 3  # doc: Type is union


@enum_tools.documentation.document_enum
class ReferenceType(IntEnum):
    """
    Reference types.
    """

    DATA = 0  # doc: Reference is data
    ENUM = 1  # doc: Reference is an enum
    STRUC = 2  # doc: Reference is a structure
    UNKNOWN = 3  # doc: Reference type is unknown


ReferenceTarget: TypeAlias = "Data | Structure | StructureMember"
"""Data reference target"""
