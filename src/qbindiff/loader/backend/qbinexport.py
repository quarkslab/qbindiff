import qbinexport, networkx
from functools import cache
from capstone import CS_OP_IMM, CS_GRP_JUMP
from typing import Any

from qbindiff.loader import (
    Program,
    Function,
    BasicBlock,
    Instruction,
    Operand,
    Data,
    Structure,
)
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

    if qbe_ref_type == qbinexport.types.ReferenceType.DATA:
        return ReferenceType.DATA
    elif qbe_ref_type == qbinexport.types.ReferenceType.ENUM:
        return ReferenceType.ENUM
    elif qbe_ref_type == qbinexport.types.ReferenceType.STRUC:
        return ReferenceType.STRUC
    else:
        return StructureType.UNKNOWN


def to_hex2(s):
    r = "".join("{0:02x}".format(c) for c in s)
    while r[0] == "0":
        r = r[1:]
    return r


def to_x(s):
    from struct import pack

    if not s:
        return "0"
    x = pack(">q", s)
    while x[0] in ("\0", 0):
        x = x[1:]
    return to_hex2(x)


# ===========================================


class OperandBackendQBinExport(AbstractOperandBackend):
    """Backend loader of a Operand using QBinExport"""

    def __init__(self, cs_instruction, cs_operand: capstoneOperand):
        super(OperandBackendQBinExport, self).__init__()

        self.cs_instr = cs_instruction
        self.cs_operand = cs_operand

    def __str__(self) -> str:
        op = self.cs_operand
        if self.type == capstone.CS_OP_REG:
            return self.cs_instr.reg_name(op.reg)
        elif self.type == capstone.CS_OP_IMM:
            return to_x(op.imm)
        elif self.type == capstone.CS_OP_MEM:
            op_str = ""
            if op.mem.segment != 0:
                op_str += f"[{self.cs_instr.reg_name(op.mem.segment)}]:"
            if op.mem.base != 0:
                op_str += f"[{self.cs_instr.reg_name(op.mem.base)}"
            if op.mem.index != 0:
                op_str += f"+{self.cs_instr.reg_name(op.mem.index)}"
            if op.mem.disp != 0:
                op_str += f"+0x{op.mem.disp:x}"
            op_str += "]"
            return op_str
        else:
            raise NotImplementedError(f"Unrecognized capstone type {self.type}")

    @property
    def immutable_value(self) -> int | None:
        """
        Returns the immutable value (not addresses) used by the operand.
        If there is no immutable value then returns None.
        """

        if self.is_immutable():
            return self.cs_operand.value.imm
        return None

    @property
    def type(self) -> int:
        """Returns the capstone operand type"""
        return self.cs_operand.type

    def is_immutable(self) -> bool:
        """Returns whether the operand is an immutable (not considering addresses)"""

        # Ignore jumps since the target is an immutable
        return self.cs_operand.type == CS_OP_IMM and not self.cs_instr.group(
            CS_GRP_JUMP
        )


class InstructionBackendQBinExport(AbstractInstructionBackend):
    """Backend loader of a Instruction using QBinExport"""

    def __init__(self, qb_instruction: qbInstruction, structures: list[Structure]):
        super(InstructionBackendQBinExport, self).__init__()

        self.qb_instr = qb_instruction
        self.cs_instr = qb_instruction.cs_inst
        # Hoping that there won't be two struct with the same name
        self.structures = {struct.name: struct for struct in structures}
        self._operands = None

    def _cast_references(
        self, references: list[qbinexport.types.ReferenceTarget]
    ) -> list[ReferenceTarget]:
        """Cast the qbinexport references to qbindiff reference types"""

        ret_ref = []
        for ref in references:
            match ref:
                case qbinexport.data.Data():
                    data_type = convert_data_type(ref.type)
                    ret_ref.append(Data(data_type, ref.address, ref.value))
                case qbinexport.structure.Structure(name=name):
                    ret_ref.append(self.structures[name])
                case qbinexport.structure.StructureMember(
                    structure=qbe_struct, name=name
                ):
                    ret_ref.append(
                        self.structures[qbe_struct.name].member_by_name(name)
                    )
                case qbinexport.Instruction():  # Not implemented for now
                    logging.warning("Skipping instruction reference")
        return ret_ref

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
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """Returns all the references towards the instruction"""

        ref = {}
        for ref_type, references in self.qb_instr.references.items():
            ref[convert_ref_type(ref_type)] = self._cast_references(references)
        return ref

    @property
    def operands(self) -> list[Operand]:
        """Returns the list of operands as Operand object"""
        if not self._operands:
            self._operands = []
            for o in self.cs_instr.operands:
                self._operands.append(Operand(LoaderType.qbinexport, self.cs_instr, o))

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

    def __init__(
        self, basic_block: BasicBlock, qb_block: qbBlock, structures: list[Structure]
    ):
        super(BasicBlockBackendQBinExport, self).__init__()

        # Private attributes
        self._addr = qb_block.start

        for instr in qb_block.instructions:
            basic_block.append(Instruction(LoaderType.qbinexport, instr, structures))

    @property
    def addr(self) -> Addr:
        """The address of the basic block"""
        return self._addr


class FunctionBackendQBinExport(AbstractFunctionBackend):
    """Backend loader of a Function using QBinExport"""

    def __init__(
        self, function: Function, qb_func: qbFunction, structures: list[Structure]
    ):
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
            b = BasicBlock(LoaderType.qbinexport, block, structures)
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
            f = Function(LoaderType.qbinexport, func, self.structures)
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
    @cache
    def structures(self) -> list[Structure]:
        """Returns the list of structures defined in program"""

        struct_list = []
        for qbe_struct in self.qb_prog.structures:
            struct = Structure(
                convert_struct_type(qbe_struct.type), qbe_struct.name, qbe_struct.size
            )
            for offset, member in qbe_struct.items():
                struct.add_member(
                    offset,
                    convert_data_type(member.type),
                    member.name,
                    member.size,
                    member.value,
                )
            struct_list.append(struct)
        return struct_list

    @property
    def callgraph(self) -> networkx.DiGraph:
        """The callgraph of the program"""
        return self._callgraph
