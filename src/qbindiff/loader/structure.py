from __future__ import annotations
from typing import Any, Dict

from qbindiff.loader.types import DataType, StructureType


class StructureMember:
    """
    Class that represents a struct member reference
    """

    def __init__(
        self,
        data_type: DataType,
        name: str,
        size: int,
        value: Any,
        structure: Structure,
    ):
        self.type = data_type
        self.name = name
        self.size = size
        self.value = value
        self.structure = structure


class Structure:
    """
    Class that represents a struct reference
    """

    def __init__(self, struct_type: StructureType, name: str, size: int):
        self.type = struct_type
        self.name = name
        self.size = size
        self.members: Dict[int, StructureMember] = {}  # { offset: StructureMember }

    def add_member(
        self, offset: int, data_type: DataType, name: str, size: int, value: Any
    ) -> None:
        """
        Add a new member of the struct at offset `offset`

        :param offset: offset where to add the member
        :param data_type: type of the member
        :param name : its name
        :param size: its size
        :param value: its value
        :return: None
        """

        self.members[offset] = StructureMember(data_type, name, size, value, self)

    def member_by_name(self, name: str) -> StructureMember | None:
        """
        Get member by name. WARNING: time complexity O(n)

        :param name: name from which we want to recover the structure member
        :return: member of the structure denoted by its name or None
        """

        try:
            return next(filter(lambda m: m.name == name, self.members.values()))
        except StopIteration:
            return None
