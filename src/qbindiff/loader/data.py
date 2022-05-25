from typing import Any

from qbindiff.types import Addr, DataType


class Data:
    """Class that represents a data reference"""

    def __init__(self, data_type, addr, value):
        self._type = data_type
        self._addr = addr
        self._value = value

    @property
    def type(self) -> DataType:
        """Returns the type of the data"""
        return self._type

    @property
    def value(self) -> Any:
        """Returns the data value"""
        return self._value

    @property
    def addr(self) -> Addr:
        """Returns the address of the referenced data"""
        return self._addr
