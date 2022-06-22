import networkx
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator

from qbindiff.loader import Structure
from qbindiff.loader.types import FunctionType, ReferenceType, ReferenceTarget
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
    def immutable_value(self) -> int | None:
        """
        Returns the immutable value (not addresses) used by the operand.
        If there is no immutable value then returns None.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> int:
        """Returns the operand type as int"""
        raise NotImplementedError()

    @abstractmethod
    def is_immutable(self) -> bool:
        """Returns whether the operand is an immutable (not considering addresses)"""
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
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """Returns all the references towards the instruction"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def operands(self) -> Iterator[AbstractOperandBackend]:
        """Returns an iterator over backend operand objects"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def groups(self) -> list[int]:
        """Returns a list of groups of this instruction"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def id(self) -> int:
        """
        Return the instruction ID as int.
        WARNING: the instruction ID must be in range [0, 2000[
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def comment(self) -> str:
        """Comment associated with the instruction"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def bytes(self) -> bytes:
        """Returns the bytes representation of the instruction"""
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

    @property
    @abstractmethod
    def instructions(self) -> Iterator[AbstractInstructionBackend]:
        """Returns an iterator over backend instruction objects"""
        raise NotImplementedError()


class AbstractFunctionBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represent a generic backend loader for a Function
    """

    @property
    @abstractmethod
    def basic_blocks(self) -> Iterator[AbstractBasicBlockBackend]:
        """Returns an iterator over backend basic blocks objects"""
        raise NotImplementedError()

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

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the function"""
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
    def structures(self) -> list[Structure]:
        """Returns the list of structures defined in program"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def callgraph(self) -> networkx.DiGraph:
        """The callgraph of the program"""
        raise NotImplementedError()
