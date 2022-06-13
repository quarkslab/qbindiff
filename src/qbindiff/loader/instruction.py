from qbindiff.loader.types import LoaderType, ReferenceType, ReferenceTarget
from qbindiff.loader import Data, Operand
from qbindiff.types import Addr


class Instruction:
    """
    Defines an Instruction object that wrap the backend using under the scene.
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
        Load the Instruction using the protobuf data
        :param args: program, function, addr, protobuf index
        :return:
        """
        from qbindiff.loader.backend.binexport import InstructionBackendBinExport

        self._backend = InstructionBackendBinExport(*args, **kwargs)

    def load_ida(self, addr) -> None:
        """
        Load the Instruction using the IDA backend, (only applies when running in IDA)
        :param addr: Address of the instruction
        :return: None
        """
        from qbindiff.loader.backend.ida import InstructionBackendIDA

        self._backend = InstructionBackendIDA(addr)

    def load_qbinexport(self, *args, **kwargs) -> None:
        """Load the Instruction using the QBinExport backend"""
        from qbindiff.loader.backend.qbinexport import InstructionBackendQBinExport

        self._backend = InstructionBackendQBinExport(*args, **kwargs)

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
    @cache
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """Returns all the references towards the instruction"""
        return self._backend.references

    @property
    def data_references(self) -> list[Data]:
        """Returns the list of data that are referenced by the instruction"""
        return self.references[ReferenceType.DATA]

    @property
    def operands(self) -> list[Operand]:
        """
        Returns the list of operands as Operand object.
        Note: The objects are recreated each time this function is called.
        :return: list of operands
        """
        return self._backend.operands

    @property
    def groups(self) -> list[int]:
        """
        Returns a list of groups of this instruction.
        :return: list of groups for the instruction
        """
        return self._backend.groups

    @property
    def id(self) -> int:
        """Return the instruction ID as int"""
        return self._backend.id

    @property
    def comment(self) -> str:
        """
        Comment as set in IDA on the instruction
        :return: comment associated with the instruction in IDA
        """
        return self._backend.comment

    def __str__(self):
        return "%s %s" % (
            self.mnemonic,
            ", ".join(str(op) for op in self._backend.operands),
        )

    def __repr__(self):
        return "<Inst:%s>" % str(self)
