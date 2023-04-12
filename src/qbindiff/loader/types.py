from enum import IntEnum
from typing import TypeAlias


class LoaderType(IntEnum):
    """
    The different kind of loaders (diaphora not supported)
    """

    unknown = 0
    binexport = 1
    diaphora = 2
    ida = 3
    quokka = 4


class OperandType(IntEnum):
    """
    All the operand types as defined by IDA
    """

    void = 0
    register = 1
    memory = 2
    phrase = 3  # base+reg  or  base + offset * factor
    displacement = 4  # base+offset
    immediate = 5
    far = 6
    near = 7
    specific0 = 8
    specific1 = 9
    specific2 = 10
    specific3 = 11
    specific4 = 12
    specific5 = 13


class FunctionType(IntEnum):
    """
    Function types as defined by IDA
    """

    normal = 0
    library = 1
    imported = 2
    thunk = 3
    invalid = 4
    extern = 5


class DataType(IntEnum):
    """
    Types of data
    """

    UNKNOWN = 0
    BYTE = 1
    WORD = 2
    DOUBLE_WORD = 3
    QUAD_WORD = 4
    OCTO_WORD = 5
    FLOAT = 6
    DOUBLE = 7
    ASCII = 8


class StructureType(IntEnum):
    """
    Different structures
    """

    UNKNOWN = 0
    STRUCT = 1
    ENUM = 2
    UNION = 3


class ReferenceType(IntEnum):
    """
    Reference types
    """

    DATA = 0
    ENUM = 1
    STRUC = 2
    UNKNOWN = 3


ReferenceTarget: TypeAlias = "Data | Structure | StructureMember"
