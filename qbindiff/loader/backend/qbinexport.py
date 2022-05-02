import qbinexport
from collections.abc import Iterator

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand, Expr
from qbindiff.loader.backend import AbstractProgramBackend, AbstractFunctionBackend
from qbindiff.loader.types import FunctionType
from qbindiff.types import Addr


# Aliases
qbProgram = qbinexport.Program
qbFunction = qbinexport.function.Function
qbBlock = qbinexport.block.Block
qbInstruction = qbinexport.instruction.Instruction
qbOperand = qbinexport.instruction.Operand


class OperandBackendQbinExport(AbstractOperandBackend):
    """Backend loader of a Operand using QBinExport"""

    def __init__(self, qb_operand: qbOperand):
        super(OperandBackendQbinExport, self).__init__()

        self.qb_operand = qb_operand

    def __str__(self) -> str:
        return ""  # Not supported

    @property
    def type(self) -> OperandType:
        """Returns the operand type as defined in the types.py"""
        return -1  # Not supported

    @property
    def expressions(self) -> Iterator[Expr]:
        """Returns an iterator of expressions"""
        return iter([])  # Not supported


class InstructionBackendQbinExport(AbstractInstructionBackend):
    """Backend loader of a Instruction using QBinExport"""

    def __init__(self, qb_instruction: qbInstruction):
        super(InstructionBackendQbinExport, self).__init__()

        self.qb_instr = qb_instruction

    @property
    def addr(self) -> Addr:
        """The address of the instruction"""
        return self.qb_instr.address

    @property
    def mnemonic(self) -> str:
        """Returns the instruction mnemonic as a string"""
        return self.qb_instr.cs_inst.mnemonic

    @property
    def operands(self) -> list[Operand]:
        """Returns the list of operands as Operand object"""
        operand_list = []
        for o in self.qb_instr.operands:
            operand_list.append(Operand(LoaderType.qbinexport, o))
        return operand_list

    @property
    def groups(self) -> list[str]:
        """
        Returns a list of groups of this instruction. Groups are capstone based
        but enriched.
        """
        return []  # Not supported

    @property
    def comment(self) -> str:
        """Comment associated with the instruction"""
        return []  # Not supported


class BasicBlockBackendQBinExport(AbstractBasicBlockBackend):
    """Backend loader of a BasicBlock using QBinExport"""

    def __init__(self, basic_block: BasicBlock, qb_block: qbBlock):
        super(BasicBlockBackendQBinExport, self).__init__()

        # Private attributes
        self._addr = qb_block.start

        for instr in qb_block.instructions:
            basic_block.append(Instruction(LoaderType.qbinexport, instr))

    @property
    def addr(self) -> Addr:
        """The address of the basic block"""
        return self._addr


class FunctionBackendQBinExport(AbstractFunctionBackend):
    """Backend loader of a Function using QBinExport"""

    def __init__(self, function: Function, qb_func: qbFunction):
        super(FunctionBackendQBinExport, self).__init__()

        self.qb_prog = qb_func.program
        self.qb_func = qb_func

        # private attributes
        self._type = None
        self._name = None

        # [TODO] Init all the properties and free the memory of qb_prog/qb_func

        bblocks = {
            addr: self.qb_func.get_block(addr) for addr in self.qb_func.graph.nodes
        }
        for addr, block in bblocks.items():
            b = BasicBlock(LoaderType.qbinexport, block)
            if addr in function:
                logging.error("Address collision for 0x%x" % addr)
            function[addr] = b

    @property
    def addr(self) -> Addr:
        """The address of the function"""
        return self.qb_func.start

    @property
    def graph(self) -> networkx.DiGraph:
        """The Control Flow Graph of the function"""
        raise NotImplementedError()

    @property
    def parents(self) -> set[Addr]:
        """Set of function parents in the call graph"""
        return {
            self.qb_prog.get_function_by_chunk(c).start for c in self.qb_func.callers
        }

    @property
    def children(self) -> set[Addr]:
        """Set of function children in the call graph"""
        return {self.qb_prog.get_function_by_chunk(c).start for c in self.qb_func.calls}

    @property
    def type(self) -> FunctionType:
        """The type of the function (as defined by IDA)"""
        if self._type is not None:
            return self._type

        f_type = self.qb_func.type
        if f_type == qbinexport.types.FunctionType.NORMAL:
            self._type = FunctionType.normal
        elif f_type == qbinexport.types.FunctionType.IMPORTED:
            self._type = FunctionType.imported
        elif f_type == qbinexport.types.FunctionType.LIBRARY:
            self._type = FunctionType.library
        elif f_type == qbinexport.types.FunctionType.THUNK:
            self._type = FunctionType.thunk
        elif f_type == qbinexport.types.FunctionType.EXTERN:
            self._type = FunctionType.extern
        elif f_type == qbinexport.types.FunctionType.INVALID:
            self._type = FunctionType.invalid
        else:
            raise NotImplementedError(f"Function type {f_type} not implemented")

        return self._type

    @type.setter
    def type(self, value: FunctionType) -> None:
        self._type = value

    @property
    def is_import(self) -> bool:
        """True if the function is imported"""
        # Should we consider also FunctionType.library?
        if self.type in (FunctionType.imported, FunctionType.thunk):
            return True
        return False

    @property
    def name(self) -> str:
        """The name of the function"""
        if self._name is None:
            self._name = self.qb_func.name
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value


class ProgramBackendQBinExport(AbstractProgramBackend):
    """Backend loader of a Program using QBinExport"""

    def __init__(self, program: Program, export_path: str, exec_path: str):
        super(ProgramBackendQBinExport, self).__init__()

        self.qb_prog = qbinexport.Program(export_path, exec_path)

        for addr, func in self.qb_prog.items():
            f = Function(LoaderType.qbinexport, func)
            if addr in program:
                logging.error("Address collision for 0x%x" % addr)
            program[addr] = f

    @property
    def name(self):
        return self.qb_prog.executable.exec_file.name

    @property
    def callgraph(self):
        pass
