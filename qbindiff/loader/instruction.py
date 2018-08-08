from qbindiff.loader.types import LoaderType

from typing import List
from qbindiff.loader.operand import Operand


class Instruction(object):
    """
    Defines an Instruction object that wrap the backend using under the scene.
    """
    def __init__(self, loader, *args):
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data: dict) -> None:
        """
        Load the isntruction using the raw dict data
        :param data: raw data
        :return: None
        """
        from qbindiff.loader.backend.qbindiff import InstructionBackendQBinDiff
        self._backend = InstructionBackendQBinDiff(data)

    def load_binexport(self, *args) -> None:
        """
        Load the Instruction using the protobuf data
        :param args: program, function, addr, protobuf index
        :return:
        """
        from qbindiff.loader.backend.binexport import InstructionBackendBinExport
        self._backend = InstructionBackendBinExport(*args)

    @property
    def addr(self) -> int:
        """
        Returns the address of the instruction
        :return: addr
        """
        return self._backend.addr

    @property
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        :return: mnemonic as a string
        """
        return self._backend.mnemonic

    @property
    def operands(self) -> List[Operand]:
        """
        Returns the list of operands as Operand object.
        Note: The objects are recreated each time this function is called.
        :return: list of operands
        """
        return self._backend.operands

    @property
    def groups(self) -> List[str]:
        """
        Returns a list of groups of this instruction. Groups are capstone based
        but enriched.
        Note: Binexport does not support groups and thus the list is empty
        :return: list of groups for the instruction
        """
        return self._backend.groups

    @property
    def comment(self) -> str:
        """
        Comment as set in IDA on the instruction
        :return: comment associated with the instruction in IDA
        """
        return self._backend.comment

    def __str__(self):
        return "%s %s" % (self.mnemonic, ', '.join(str(op) for op in self._backend.operands))

    def __repr__(self):
        return "<Inst:%s>" % str(self)
