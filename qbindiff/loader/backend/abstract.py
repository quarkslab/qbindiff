import networkx
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator

from qbindiff.loader import Operand, Expr
from qbindiff.loader.types import FunctionType, OperandType
from qbindiff.types import Addr


class AbstractOperandBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a Operand
    """

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> OperandType:
        """Returns the operand type as defined in the types.py"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def expressions(self) -> Iterator[Expr]:
        """Returns an iterator of expressions"""
        raise NotImplementedError()


class AbstractInstructionBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a Instruction
    """

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """The address of the instruction"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def mnemonic(self) -> str:
        """Returns the instruction mnemonic as a string"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def operands(self) -> list[Operand]:
        """Returns the list of operands as Operand object"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def groups(self) -> list[str]:
        """
        Returns a list of groups of this instruction. Groups are capstone based
        but enriched.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def comment(self) -> str:
        """Comment associated with the instruction"""
        raise NotImplementedError()


class AbstractBasicBlockBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a BasicBlock
    """

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """The address of the basic block"""
        raise NotImplementedError()


class AbstractFunctionBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a Function
    """

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """The address of the function"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def graph(self) -> networkx.DiGraph:
        """The Control Flow Graph of the function"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def parents(self) -> set[Addr]:
        """Set of function parents in the call graph"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def children(self) -> set[Addr]:
        """Set of function children in the call graph"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> FunctionType:
        """The type of the function (as defined by IDA)"""
        raise NotImplementedError()

    @type.setter
    @abstractmethod
    def type(self, value: FunctionType) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the function"""
        raise NotImplementedError()

    @name.setter
    @abstractmethod
    def name(self, value: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def is_import(self) -> bool:
        """True if the function is imported"""
        raise NotImplementedError()


class AbstractProgramBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a Program
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the program"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def callgraph(self) -> networkx.DiGraph:
        """The callgraph of the program"""
        raise NotImplementedError()
