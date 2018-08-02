
from qbindiff.loader.types import LoaderType
from qbindiff.loader.backend.qbindiff import OperandBackendQBinDiff
from qbindiff.loader.backend.binexport import OperandBackendBinexport


class Operand(object):
    def __init__(self, loader, *args):
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data):
        self._backend = OperandBackendQBinDiff(data)

    def load_binexport(self, *args):
        self._backend = OperandBackendBinexport(*args)

    @property
    def type(self):
        return self._backend.type

    @property
    def expressions(self):
        return self._backend.expressions

    def __str__(self):
        return str(self._backend)

    def __repr__(self):
        return "<Op:%s>" % str(self)
