
from qbindiff.loader.types import LoaderType
# from qbindiff.loader.backend.qbindiff import InstructionBackendQBinDiff
# from qbindiff.loader.backend.binexport import InstructionBackendBinExport
from qbindiff.loader.operand import Operand
from typing import List, Generator


class Instruction(object):
    def __init__(self, loader, *args):
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data):
        from qbindiff.loader.backend.qbindiff import InstructionBackendQBinDiff
        self._backend = InstructionBackendQBinDiff(data)

    def load_binexport(self, *args):
        from qbindiff.loader.backend.binexport import InstructionBackendBinExport
        self._backend = InstructionBackendBinExport(*args)

    @property
    def addr(self) -> int:
        return self._backend.addr

    @property
    def mnemonic(self) -> str:
        return self._backend.mnemonic

    @property
    def operands(self):
        return self._backend.operands

    @property
    def groups(self) -> List[str]:
        return self._backend.groups

    @property
    def comment(self):
        return self._backend.comment

    def __str__(self):
        return "%s %s" % (self.mnemonic, ', '.join(str(op) for op in self._backend.operands))

    def __repr__(self):
        return "<Inst:%s>" % str(self)
