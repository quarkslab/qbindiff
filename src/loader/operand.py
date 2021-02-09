from qbindiff.loader.types import LoaderType

# typing imports
from typing import Dict, Iterator
from qbindiff.loader.types import OperandType

Expr = Dict[str, str]  # each dict contains two keys 'types' and 'value' with their associated value


class Operand(object):
    """
    Represent an operand object which hide the underlying
    backend implementation
    """
    def __init__(self, loader, *args):
        self._backend = None
        if loader == LoaderType.qbindiff:
            self.load_qbindiff(*args)
        elif loader == LoaderType.binexport:
            self.load_binexport(*args)
        elif loader == LoaderType.ida:
            self.load_ida(*args)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_qbindiff(self, data: dict) -> None:
        """
        Instanciate the operand using the dict values retrieved.
        :param data: raw dict of operand data
        :return: None
        """
        from qbindiff.loader.backend.qbindiff import OperandBackendQBinDiff
        self._backend = OperandBackendQBinDiff(data)

    def load_binexport(self, *args) -> None:
        """
        Load the operand using the data of the binexport file
        :param args: program, function, and operand index
        :return: None
        """
        from qbindiff.loader.backend.binexport import OperandBackendBinexport
        self._backend = OperandBackendBinexport(*args)

    def load_ida(self, op_t, ea) -> None:
        '''
        Instanciate the operand using IDA API
        :param op_t: op_t* as defined in the IDA SDK
        :param ea: address of the instruction
        :return: None
        '''
        from qbindiff.loader.backend.ida import OperandBackendIDA
        self._backend = OperandBackendIDA(op_t, ea)

    @property
    def type(self) -> OperandType:
        """
        Returns the operand type as defined in the types.py
        :return: OperandType
        """
        return self._backend.type

    @property
    def expressions(self) -> Iterator[Expr]:
        """
        Returns an iterator of expressions. Each expression
        is a dictionnary containting two keys "type" and "value"
        :return:
        """
        return self._backend.expressions

    def __str__(self):
        return str(self._backend)

    def __repr__(self):
        return "<Op:%s>" % str(self)
