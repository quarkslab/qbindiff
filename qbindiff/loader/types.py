from enum import IntEnum


class LoaderType(IntEnum):
    qbindiff = 0
    binexport = 1
    diaphora = 2


class OperandType(IntEnum):
    void = 0
    register = 1
    memory = 3
    phrase = 4         # base+reg  or  base + offset * factor
    displacement = 5   # base+offset
    immediate = 6
    far = 7
    near = 8
    specific0 = 9
    specific1 = 10
    specific2 = 11
    specific3 = 12
    specific4 = 13
    specific5 = 14


class FunctionType(IntEnum):
    normal = 0
    library = 1
    imported = 2
    thunk = 3
    invalid = 4
