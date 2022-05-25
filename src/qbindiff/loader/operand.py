from capstone import CS_OP_IMM
from typing import Any

from qbindiff.loader.types import LoaderType

# Capstone typing
capstoneOperand = Any
capstoneValue = Any


class Operand:
    """
    Represent an operand object which hide the underlying backend implementation
    """

    def __init__(self, loader, *args, **kwargs):
        self._backend = None
        if loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        elif loader == LoaderType.ida:
            self.load_ida(*args, **kwargs)
        elif loader == LoaderType.qbinexport:
            self.load_qbinexport(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self, *args, **kwargs) -> None:
        """
        Load the operand using the data of the binexport file
        :param args: program, function, and operand index
        :return: None
        """
        from qbindiff.loader.backend.binexport import OperandBackendBinexport

        self._backend = OperandBackendBinexport(*args, **kwargs)

    def load_ida(self, op_t, ea) -> None:
        """
        Instanciate the operand using IDA API
        :param op_t: op_t* as defined in the IDA SDK
        :param ea: address of the instruction
        :return: None
        """
        from qbindiff.loader.backend.ida import OperandBackendIDA

        self._backend = OperandBackendIDA(op_t, ea)

    def load_qbinexport(self, *args, **kwargs) -> None:
        """Load the operand using the qbinexport backend"""
        from qbindiff.loader.backend.qbinexport import OperandBackendQBinExport

        self._backend = OperandBackendQBinExport(*args, **kwargs)

    @property
    def capstone(self) -> capstoneOperand:
        """Returns the latent capstone operand object"""
        return self._backend.capstone

    @property
    def type(self) -> int:
        """
        Returns the capstone operand type
        :return: int
        """
        return self._backend.type

    @property
    def value(self) -> capstoneValue:
        """Returns the capstone operand value"""
        return self._backend.value

    def is_immutable(self):
        return self.capstone.type == CS_OP_IMM

    def __str__(self):
        return str(self._backend)

    def __repr__(self):
        return "<Op:%s>" % str(self)
