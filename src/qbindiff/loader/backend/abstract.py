import networkx
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator

from qbindiff.loader import Structure
from qbindiff.loader.types import FunctionType, ReferenceType, ReferenceTarget, OperandType
from qbindiff.types import Addr
from typing import Set, Dict, List


class AbstractOperandBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represents a generic backend loader for a Operand
    """

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def value(self) -> int | None:
        """
        Returns the immediate value (not addresses) if the operand is constant.
        If not, None is returned.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> OperandType:
        """
        Returns the operand type.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_immediate(self) -> bool:
        """
        Returns whether the operand is an immediate (not addresses)
        """
        raise NotImplementedError()


class AbstractInstructionBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represents a generic backend loader for a Instruction
    """

    """
    Max instruction ID. All the instruction IDs will be in the range [0, MAX_ID]
    """
    MAX_ID = 2500

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """
        The address of the instruction
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def references(self) -> Dict[ReferenceType, List[ReferenceTarget]]:
        """
        Returns all the references towards the instruction
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def operands(self) -> Iterator[AbstractOperandBackend]:
        """
        Returns an iterator over backend operand objects
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def groups(self) -> List[str]:
        """
        Returns a list of groups of this instruction
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def id(self) -> int:
        """
        Returns the capstone instruction ID as a non negative int. The ID is in the range [0, MAX_ID].
        The id is MAX_ID if there is no capstone instruction available.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def comment(self) -> str:
        """
        Comment associated with the instruction
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the instruction
        """
        raise NotImplementedError()


class AbstractBasicBlockBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represents a generic backend loader for a BasicBlock
    """

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """
        The address of the basic block
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def instructions(self) -> Iterator[AbstractInstructionBackend]:
        """
        Returns an iterator over backend instruction objects
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the basic block
        """
        raise NotImplementedError()


class AbstractFunctionBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represents a generic backend loader for a Function.
    """

    @property
    @abstractmethod
    def basic_blocks(self) -> Iterator[AbstractBasicBlockBackend]:
        """
        Returns an iterator over backend basic blocks objects.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def addr(self) -> Addr:
        """
        The address of the function.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def graph(self) -> networkx.DiGraph:
        """
        The Control Flow Graph of the function.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def parents(self) -> Set[Addr]:
        """
        Set of function parents in the call graph.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def children(self) -> Set[Addr]:
        """
        Set of function children in the call graph.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> FunctionType:
        """
        The type of the function.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the function.
        """
        raise NotImplementedError()

    def unload_blocks(self) -> None:
        """
        Unload basic blocks from memory
        """
        pass


class AbstractProgramBackend(metaclass=ABCMeta):
    """
    This is an abstract class and should not be used as is.
    It represents a generic backend loader for a Program
    """

    @property
    @abstractmethod
    def functions(self) -> Iterator[AbstractFunctionBackend]:
        """
        Returns an iterator over backend function objects.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the program.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def structures(self) -> List[Structure]:
        """
        Returns the list of structures defined in program.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def callgraph(self) -> networkx.DiGraph:
        """
        The callgraph of the program.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def fun_names(self) -> Dict[str, Addr]:
        """
        Returns a dictionary with function name as key and the function address as value.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def exec_path(self) -> str:
        """
        Returns the executable path
        """
        raise NotImplementedError()