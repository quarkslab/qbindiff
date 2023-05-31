# builtin imports
from __future__ import annotations
import logging
import weakref
from typing import Any, TypeAlias, Dict, List
from collections.abc import Iterator
from functools import cached_property

# third-party imports
import capstone
import binexport
import binexport.types
import networkx

# local imports
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader import Structure
from qbindiff.loader.types import FunctionType, ReferenceType, ReferenceTarget, OperandType
from qbindiff.types import Addr

# Type aliases
beFunction: TypeAlias = binexport.function.FunctionBinExport
beBasicBlock: TypeAlias = binexport.basic_block.BasicBlockBinExport
beInstruction: TypeAlias = binexport.instruction.InstructionBinExport
capstoneOperand: TypeAlias = Any  # Relaxed typing


# === General purpose utils functions ===
def _get_capstone_disassembler(binexport_arch: str, mode: int = 0):
    def capstone_context(arch, mode):
        context = capstone.Cs(arch, mode)
        context.detail = True
        return context

    if binexport_arch == "x86":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_32 | mode)
    elif binexport_arch == "x86-64":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_64 | mode)
    elif binexport_arch == "ARM-32":
        return capstone_context(capstone.CS_ARCH_ARM, mode)

    raise NotImplementedError(f"Architecture {binexport_arch} has not be implemented")


def is_same_mnemonic(mnemonic1: str, mnemonic2: str) -> bool:
    """Check whether two mnemonics are the same"""

    def normalize(mnemonic: str) -> str:
        if mnemonic == "ldmia":
            return "ldm"
        return mnemonic.replace("lo", "cc")

    mnemonic1 = normalize(mnemonic1)
    mnemonic2 = normalize(mnemonic2)

    if mnemonic1 == mnemonic2:
        return True

    if len(mnemonic1) > len(mnemonic2):
        mnemonic1, mnemonic2 = mnemonic2, mnemonic1
    if mnemonic1 + ".w" == mnemonic2:
        return True

    return False
# =======================================


class OperandBackendBinExport(AbstractOperandBackend):
    def __init__(self, cs_instruction: capstone.CsInsn, cs_operand: capstoneOperand):
        super(OperandBackendBinExport, self).__init__()

        self.cs_instr = cs_instruction
        self.cs_operand = cs_operand

    def __str__(self) -> str:
        op = self.cs_operand
        if self.type == capstone.CS_OP_REG:
            return self.cs_instr.reg_name(op.reg)
        elif self.type == capstone.CS_OP_IMM:
            return hex(op.imm)
        elif self.type == capstone.CS_OP_MEM:
            op_str = ""
            if op.mem.segment != 0:
                op_str += f"[{self.cs_instr.reg_name(op.mem.segment)}]:"
            if op.mem.base != 0:
                op_str += f"[{self.cs_instr.reg_name(op.mem.base)}"
            if op.mem.index != 0:
                op_str += f"+{self.cs_instr.reg_name(op.mem.index)}"
            if (disp := op.mem.disp) != 0:
                if disp > 0:
                    op_str += "+"
                else:
                    op_str += "-"
                    disp = -disp
                op_str += f"0x{disp:x}"
            op_str += "]"
            return op_str
        else:
            raise NotImplementedError(f"Unrecognized capstone type {self.type}")

    @property
    def value(self) -> int | None:
        """
        Return the immediate value (not addresses) used by the operand.
        """

        if self.is_immediate():
            return self.cs_operand.value.imm
        return None

    @property
    def type(self) -> OperandType:
        """Returns the capstone operand type"""
        op = self.cs_operand
        typ = OperandType.unknown
        if self.type == capstone.CS_OP_REG:
            return OperandType.register
        elif self.type == capstone.CS_OP_IMM:
            return OperandType.immediate
        elif self.type == capstone.CS_OP_MEM:
            if op.mem.base != 0 and op.mem.value != 0:
                typ = OperandType.displacement
            if op.mem.base != 0 and op.mem.index != 0:
                typ = OperandType.phrase
            if op.mem.disp != 0:
                typ = OperandType.displacement
        else:
            raise NotImplementedError(f"Unrecognized capstone type {self.type}")
        return typ

    def is_immediate(self) -> bool:
        """Returns whether the operand is an immediate value (not considering addresses)"""
        # Ignore jumps since the target is an immediate
        return self.type == OperandType.immediate and not self.cs_instr.group(capstone.CS_GRP_JUMP)


class InstructionBackendBinExport(AbstractInstructionBackend):
    def __init__(self, cs_instruction: capstone.CsInsn):
        super(InstructionBackendBinExport, self).__init__()

        self.cs_instr = cs_instruction

    @property
    def addr(self) -> Addr:
        return self.cs_instr.address

    @property
    def mnemonic(self) -> str:
        return self.cs_instr.mnemonic

    @property
    def references(self) -> Dict[ReferenceType, List[ReferenceTarget]]:
        """
        Returns all the references towards the instruction
        BinExport only exports data references' address so no data type nor value.
        """
        return {}  # Not supported

    @property
    def operands(self) -> Iterator[OperandBackendBinExport]:
        """Returns an iterator over backend operand objects"""
        if self.cs_instr is None:
            return iter([])
        return (OperandBackendBinExport(self.cs_instr, o) for o in self.cs_instr.operands)

    @property
    def groups(self) -> List[str]:
        return self.cs_instr.groups

    @property
    def id(self) -> int:
        """Return the capstone instruction ID"""
        return self.cs_instr.id

    @property
    def comment(self) -> str:
        return ""  # Not supported

    @property
    def bytes(self) -> bytes:
        """Returns the bytes representation of the instruction"""
        return bytes(self.cs_instr.bytes)


class BasicBlockBackendBinExport(AbstractBasicBlockBackend):

    def __init__(self, program: weakref.ref[ProgramBackendBinExport], be_block: beBasicBlock):
        super(BasicBlockBackendBinExport, self).__init__()

        self._program = program
        self.be_block = be_block

    def _disassemble(self, bb_asm: bytes, correct_mnemonic: str, correct_size: int) -> list[capstone.CsInsn]:
        """
        Disassemble the basic block using capstone trying to guess the instruction set
        when unable to determine it from binexport.

        :param bb_asm: basic block raw bytes
        :param correct_mnemonic: The correct mnemonic of the first instruction of the
                                 basic block
        :param correct_size: The right size of the first instruction of the basic block
        """

        instructions = []
        mnemonic = None
        size = None
        arch = self.program.architecture_name
        capstone_mode = 0
        arm_mode = 0

        # No need to guess the context for these arch
        if arch in ("x86", "x86-64"):
            md = _get_capstone_disassembler(arch)
            return list(md.disasm(bb_asm, self.addr))

        # Bruteforce-guessing the context
        while size != correct_size or not is_same_mnemonic(mnemonic, correct_mnemonic):
            # change mode
            if arch == "ARM-32":
                capstone_mode = 0
                if arm_mode & 0b1:
                    capstone_mode |= capstone.CS_MODE_ARM
                if arm_mode & 0b10:
                    capstone_mode |= capstone.CS_MODE_THUMB
                #if be_program.cortexm:
                #    capstone_mode |= capstone.CS_MODE_MCLASS
                if arm_mode > 0b11:
                    logging.error(f"Cannot guess the instruction set of the instruction at address 0x{self.addr:x}")
                arm_mode += 1

            md = _get_capstone_disassembler(arch, capstone_mode)
            disasm = md.disasm(bb_asm, self.addr)
            try:
                instr = next(disasm)
                mnemonic = instr.mnemonic
                size = instr.size
            except StopIteration:
                mnemonic = None
                size = None

        instructions.append(instr)
        instructions.extend(disasm)
        return instructions

    @property
    def program(self) -> ProgramBackendBinExport:
        """Wrapper on weak reference on ProgramBackendBinExport"""
        return self._program()

    @property
    def addr(self) -> Addr:
        return self.be_block.addr

    @property
    def instructions(self) -> Iterator[InstructionBackendBinExport]:
        """Returns an iterator over backend instruction objects"""

        # Generates the first instruction and use it to guess the context for capstone
        first_instr = next(iter(self.be_block.values()))
        capstone_instructions = self._disassemble(self.be_block.bytes, first_instr.mnemonic, len(first_instr.bytes))

        # Then iterate over the instructions
        return (InstructionBackendBinExport(instr) for instr in capstone_instructions)

    @property
    def bytes(self) -> bytes:
        return b"".join(x.bytes for x in self.instructions)


class FunctionBackendBinExport(AbstractFunctionBackend):
    def __init__(self, program: weakref.ref[ProgramBackendBinExport], be_func: beFunction):
        super(FunctionBackendBinExport, self).__init__()
        self.be_func = be_func
        self._program = program

    @property
    def basic_blocks(self) -> Iterator[BasicBlockBackendBinExport]:
        """Returns an iterator over backend basic blocks objects"""
        return (BasicBlockBackendBinExport(self._program, bb) for addr, bb in self.be_func.blocks.items())

    @property
    def addr(self) -> Addr:
        """The address of the function"""
        return self.be_func.addr

    @property
    def graph(self) -> "networkx.DiGraph":
        """The Control Flow Graph of the function"""
        return self.be_func.graph

    @property
    def parents(self) -> set[Addr]:
        """Set of function parents in the call graph"""
        return {func.addr for func in self.be_func.parents}

    @property
    def children(self) -> set[Addr]:
        """Set of function children in the call graph"""
        return {func.addr for func in self.be_func.children}

    @cached_property
    def type(self) -> FunctionType:
        """The type of the function (as defined by IDA)"""

        f_type = self.be_func.type
        if f_type == binexport.types.FunctionType.NORMAL:
            return FunctionType.normal
        elif f_type == binexport.types.FunctionType.LIBRARY:
            return FunctionType.library
        elif f_type == binexport.types.FunctionType.IMPORTED:
            return FunctionType.imported
        elif f_type == binexport.types.FunctionType.THUNK:
            return FunctionType.thunk
        elif f_type == binexport.types.FunctionType.INVALID:
            return FunctionType.invalid
        else:
            raise NotImplementedError(f"Function type {f_type} not implemented")

    @property
    def name(self):
        return self.be_func.name

    def is_import(self) -> bool:
        """True if the function is imported"""
        # Should we consider also FunctionType.thunk?
        return self.type in (FunctionType.imported, FunctionType.extern)

    def unload_blocks(self) -> None:
        """Unload basic blocks from memory"""
        # del self.be_func.blocks
        pass  # can't delete it as it is decorated with @cache_property


class ProgramBackendBinExport(AbstractProgramBackend):
    def __init__(self, file: str, *, enable_cortexm: bool = False):
        super(ProgramBackendBinExport, self).__init__()

        self._enable_cortexm = enable_cortexm

        self.exec_path = file
        self.be_prog = binexport.ProgramBinExport(file)
        self.architecture_name = self.be_prog.architecture
        self._fun_names = {}  # {fun_name : fun_address}

    def __repr__(self) -> str:
        return f"<{type(self).__name__}:{self.name}>"

    @property
    def functions(self) -> Iterator[FunctionBackendBinExport]:
        """Returns an iterator over backend function objects"""

        functions = []
        for addr, func in self.be_prog.items():
            f = FunctionBackendBinExport(weakref.ref(self), func)
            functions.append(f)
            self._fun_names[f.name] = f.addr

        return iter(functions)

    @property
    def cortexm(self) -> bool:
        return self._enable_cortexm

    @property
    def name(self):
        return self.be_prog.name

    @property
    def structures(self) -> list[Structure]:
        """
        Returns the list of structures defined in program.
        WARNING: Not supported by BinExport
        """

        return []  # Not supported

    @property
    def callgraph(self) -> "networkx.DiGraph":
        return self.be_prog.callgraph

    @property
    def fun_names(self) -> dict[str, int]:
        """
        Returns a dictionary with function name as key and the function address as value
        """
        return self._fun_names
