from typing import Any

from qbindiff.loader.types import DataType
from qbindiff.types import Addr


class Data:
    """
    Class that represents a data reference
    """

    def __init__(self, data_type: DataType, addr: Addr, value: Any):
        self.type = data_type
        self.addr = addr
        self.value = value
