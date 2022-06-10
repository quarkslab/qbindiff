import qbinexport, networkx
from functools import cache
from typing import Any

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand, Data
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader.types import (
    FunctionType,
    LoaderType,
    DataType,
    StructureType,
    ReferenceType,
    ReferenceTarget,
)
from qbindiff.types import Addr


# Aliases
qbProgram = qbinexport.Program
qbFunction = qbinexport.function.Function
qbBlock = qbinexport.block.Block
qbInstruction = qbinexport.instruction.Instruction
qbOperand = qbinexport.instruction.Operand
capstoneOperand = Any  # Don't import the whole capstone module just for the typing
capstoneValue = Any  # Don't import the whole capstone module just for the typing


# ===== General purpose utils functions =====


def convert_data_type(qbe_data_type: qbinexport.types.DataType) -> DataType:
    """Convert a qbinexport DataType to qbindiff DataType"""

    if qbe_data_type == qbinexport.types.DataType.ASCII:
        return DataType.ASCII
    elif qbe_data_type == qbinexport.types.DataType.BYTE:
        return DataType.BYTE
    elif qbe_data_type == qbinexport.types.DataType.WORD:
        return DataType.WORD
    elif qbe_data_type == qbinexport.types.DataType.DOUBLE_WORD:
        return DataType.DOUBLE_WORD
    elif qbe_data_type == qbinexport.types.DataType.QUAD_WORD:
        return DataType.QUAD_WORD
    elif qbe_data_type == qbinexport.types.DataType.OCTO_WORD:
        return DataType.OCTO_WORD
    elif qbe_data_type == qbinexport.types.DataType.FLOAT:
        return DataType.FLOAT
    elif qbe_data_type == qbinexport.types.DataType.DOUBLE:
        return DataType.DOUBLE
    else:
        return DataType.UNKNOWN


def convert_struct_type(
    qbe_struct_type: qbinexport.types.StructureType,
) -> StructureType:
    """Convert a qbinexport StructureType to qbindiff StructureType"""

    if qbe_struct_type == qbinexport.types.StructureType.ENUM:
        return StructureType.ENUM
    elif qbe_struct_type == qbinexport.types.StructureType.STRUCT:
        return StructureType.STRUCT
    elif qbe_struct_type == qbinexport.types.StructureType.UNION:
        return StructureType.UNION
    else:
        return StructureType.UNKNOWN


def convert_ref_type(qbe_ref_type: qbinexport.types.ReferenceType) -> ReferenceType:
    """Convert a qbinexport ReferenceType to qbindiff ReferenceType"""

    if qbe_ref_type == qbinexport.types.ReferenceType.CALL:
        return ReferenceType.CALL
    elif qbe_ref_type == qbinexport.types.ReferenceType.DATA:
        return ReferenceType.DATA
    elif qbe_ref_type == qbinexport.types.ReferenceType.ENUM:
        return ReferenceType.ENUM
    elif qbe_ref_type == qbinexport.types.ReferenceType.STRUC:
        return ReferenceType.STRUC
    else:
        return StructureType.UNKNOWN


# ===========================================


class OperandBackendQBinExport(AbstractOperandBackend):
    """Backend loader of a Operand using QBinExport"""

    def __init__(self, operand_str: str, cs_operand: capstoneOperand):
        super(OperandBackendQBinExport, self).__init__()

        self.cs_operand = cs_operand
        self._str = operand_str

    def __str__(self) -> str:
        return self._str

    @property
    def capstone(self) -> capstoneOperand:
        """Returns the capstone operand object"""
        return self.cs_operand

    @property
    def type(self) -> int:
        """Returns the capstone operand type"""
        return self.cs_operand.type

    @property
    def value(self) -> capstoneValue:
        """Returns the capstone operand value"""
        return self.cs_operand.value


class InstructionBackendQBinExport(AbstractInstructionBackend):
    """Backend loader of a Instruction using QBinExport"""

    def __init__(self, qb_instruction: qbInstruction):
        super(InstructionBackendQBinExport, self).__init__()

        self.qb_instr = qb_instruction
        self.cs_instr = qb_instruction.cs_inst
        self._operands = None

    @property
    def addr(self) -> Addr:
        """The address of the instruction"""
        return self.cs_instr.address

    @property
    def mnemonic(self) -> str:
        """Returns the instruction mnemonic as a string"""
        return self.cs_instr.mnemonic

    @property
    @cache
    def data_references(self) -> list[Data]:
        """Returns the list of data that are referenced by the instruction"""

        ref = []
        for r in self.qb_instr.data_references:
            if isinstance(r, qbinexport.data.Data):
                if r.type == qbinexport.types.DataType.ASCII:
                    data_type = DataType.ASCII
                elif r.type == qbinexport.types.DataType.BYTE:
                    data_type = DataType.BYTE
                elif r.type == qbinexport.types.DataType.WORD:
                    data_type = DataType.WORD
                elif r.type == qbinexport.types.DataType.DOUBLE_WORD:
                    data_type = DataType.DOUBLE_WORD
                elif r.type == qbinexport.types.DataType.QUAD_WORD:
                    data_type = DataType.QUAD_WORD
                elif r.type == qbinexport.types.DataType.OCTO_WORD:
                    data_type = DataType.OCTO_WORD
                elif r.type == qbinexport.types.DataType.FLOAT:
                    data_type = DataType.FLOAT
                elif r.type == qbinexport.types.DataType.DOUBLE:
                    data_type = DataType.DOUBLE
                else:
                    data_type = DataType.UNKNOWN
                ref.append(Data(data_type, r.address, r.value))
            else:
                pass  # TODO understand what it is
        return ref

    @property
    def operands(self) -> list[Operand]:
        """Returns the list of operands as Operand object"""
        if not self._operands:
            self._operands = []
            for o in self.cs_instr.operands:
                self._operands.append(
                    Operand(LoaderType.qbinexport, self.cs_instr.op_str, o)
                )

        return self._operands

    @property
    def groups(self) -> list[str]:
        """
        Returns a list of groups of this instruction. Groups are capstone based
        but enriched.
        """
        return []  # Not supported

    @property
    def capstone(self) -> "capstone.CsInsn":
        """Return the capstone instruction"""
        return self.cs_instr

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

        # Stop the exploration if it's an imported function
        if self.is_import():
            return

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
        return self.qb_func.graph

    @property
    @cache
    def parents(self) -> set[Addr]:
        """Set of function parents in the call graph"""

        parents = set()
        for chunk in self.qb_func.callers:
            try:
                for func in self.qb_prog.get_function_by_chunk(chunk):
                    parents.add(func.start)
            except IndexError:
                pass  # Sometimes there can be a chunk that is not part of any function
        return parents

    @property
    @cache
    def children(self) -> set[Addr]:
        """Set of function children in the call graph"""

        children = set()
        for chunk in self.qb_func.calls:
            try:
                for func in self.qb_prog.get_function_by_chunk(chunk):
                    children.add(func.start)
            except IndexError:
                pass  # Sometimes there can be a chunk that is not part of any function
        return children

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

    def is_import(self) -> bool:
        """True if the function is imported"""
        # Should we consider also FunctionType.thunk?
        if self.type in (FunctionType.imported, FunctionType.extern):
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

        self._callgraph = networkx.DiGraph()

        for addr, func in self.qb_prog.items():
            f = Function(LoaderType.qbinexport, func)
            if addr in program:
                logging.error("Address collision for 0x%x" % addr)
            program[addr] = f

            self._callgraph.add_node(addr)
            for c_addr in f.children:
                self._callgraph.add_edge(addr, c_addr)
            for p_addr in f.parents:
                self._callgraph.add_edge(p_addr, addr)

    @property
    def name(self):
        return self.qb_prog.executable.exec_file.name

    @property
    def callgraph(self) -> networkx.DiGraph:
        """The callgraph of the program"""
        return self._callgraph
