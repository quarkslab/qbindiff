# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BinExport backend loader
"""

# builtin imports
from __future__ import annotations
import logging
import weakref
from typing import Any, TypeAlias, TYPE_CHECKING
from collections.abc import Iterator
from functools import cached_property

# third-party imports
import capstone  # type: ignore[import-untyped]
import binexport  # type: ignore[import-untyped]
import binexport.types  # type: ignore[import-untyped]
import networkx

# local imports
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader.backend.utils import convert_operand_type
from qbindiff.loader import Structure
from qbindiff.utils import log_once
from qbindiff.loader.types import (
    FunctionType,
    ReferenceType,
    ReferenceTarget,
    OperandType,
    InstructionGroup,
    ProgramCapability,
)

if TYPE_CHECKING:
    from qbindiff.types import Addr

# Type aliases
beFunction: TypeAlias = binexport.function.FunctionBinExport
beBasicBlock: TypeAlias = binexport.basic_block.BasicBlockBinExport
capstoneOperand: TypeAlias = Any  # Relaxed typing


# === General purpose utils functions ===
def _get_capstone_disassembler(binexport_arch: str, mode: int = 0):
    def capstone_context(arch, mode):
        context = capstone.Cs(arch, mode)
        context.detail = True
        return context

    if binexport_arch == "x86-32" or binexport_arch == "x86":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_32 | mode)
    elif binexport_arch == "x86-64":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_64 | mode)
    elif binexport_arch == "ARM-32":
        return capstone_context(capstone.CS_ARCH_ARM, mode)
    elif binexport_arch == "ARM-64":
        return capstone_context(capstone.CS_ARCH_ARM64, mode)
    elif binexport_arch == "MIPS-32":
        return capstone_context(capstone.CS_ARCH_MIPS, capstone.CS_MODE_32 | mode)
    elif binexport_arch == "MIPS-64":
        return capstone_context(capstone.CS_ARCH_MIPS, capstone.CS_MODE_64 | mode)

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


def parse_architecture_flag(arch_mode_str: str) -> capstone.Cs | None:
    """Return the capstone architecture corresponding to the string passed as parameter.
    The format should be something like 'CS_ARCH_any:[CS_MODE_any, ...]"""

    separator = ":"
    # Replace trailing spaces
    split = arch_mode_str.replace(" ", "").split(separator)
    # Check for only one separator
    if len(split) != 2:
        return None

    # Unpack arch and mode
    arch_str, mode_str = split
    arch = capstone.__dict__.get(arch_str)
    if arch == None:
        return None

    mode = 0
    # Look for one or more modes
    for m in mode_str.split(","):
        mode_attr = capstone.__dict__.get(m)
        if mode_attr == None:
            return None
        # Capstone mode is a bitwise enum
        mode |= mode_attr

    # Arch and Mode are valid capstone attributes, instantiate Cs object
    cs = capstone.Cs(arch, mode)
    if cs:
        # Enable details for operand
        cs.detail = True
    return cs


# =======================================


class OperandBackendBinExport(AbstractOperandBackend):
    def __init__(
        self,
        cs: capstone.Cs,
        cs_instruction: capstone.CsInsn,
        cs_operand: capstoneOperand,
        cs_operand_position: int,
    ):
        super(OperandBackendBinExport, self).__init__()

        self.cs = cs
        self.cs_instr = cs_instruction
        self.cs_operand = cs_operand
        self.cs_operand_position = cs_operand_position

    def __str__(self) -> str:
        return self.cs_instr.op_str.split(",")[self.cs_operand_position]

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
        return convert_operand_type(self.cs.arch, self.cs_operand)

    def is_immediate(self) -> bool:
        """Returns whether the operand is an immediate value (not considering addresses)"""
        # Ignore jumps since the target is an immediate
        return self.type == OperandType.immediate and not self.cs_instr.group(capstone.CS_GRP_JUMP)


class InstructionBackendBinExport(AbstractInstructionBackend):
    def __init__(self, cs: capstone.Cs, cs_instruction: capstone.CsInsn):
        super(InstructionBackendBinExport, self).__init__()

        self.cs = cs
        self.cs_instr = cs_instruction

    @property
    def addr(self) -> Addr:
        return self.cs_instr.address

    @property
    def mnemonic(self) -> str:
        return self.cs_instr.mnemonic

    @property
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
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
        return (
            OperandBackendBinExport(self.cs, self.cs_instr, o, i)
            for i, o in enumerate(self.cs_instr.operands)
        )

    @property
    def groups(self) -> list[InstructionGroup]:
        """
        Returns a list of groups of this instruction.
        """
        return list(map(InstructionGroup.from_capstone, self.cs_instr.groups))

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

    def __len__(self) -> int:
        """
        The numbers of instructions in the basic block
        """
        return len(self.be_block)

    def _guess_thumb_context(self, instr_bytes: bytes, mnemonic: str) -> int:
        """Guess wether the instruction is thumb or not"""

        if len(instr_bytes) < 2:  # Must be an error
            raise ValueError(f"Instruction malformed of size {len(instr_bytes)} bytes.")

        if len(instr_bytes) == 2:  # Must be thumb
            return capstone.CS_MODE_THUMB

        # Might be either thumb or normal arm.
        # There is no easy way of knowing whether a 4/8 bytes instruction is thumb or not
        # and IDA sometimes likes to merge two instructions together (so two 4bytes thumb
        # instructions might become a single 8bytes thumb instruction from IDA perspective).
        # The only way of checking if the context is correct is by comparing the mnemonic
        # of the capstone instruction with the one in BinExport.
        # Of course we have to rely on heuristics to know whether the two mnemonics are the
        # same or not.
        log_once(
            logging.WARNING,
            f"Relying on heuristics to guess the context mode of the binary (thumb or not)",
        )

        # Save the original mode
        arch = self.program.architecture_name

        # Bruteforce-guessing the context
        for capstone_mode in [capstone.CS_MODE_ARM, capstone.CS_MODE_THUMB]:  # try both modes
            # Disassemble the instruction and check the mnemonic
            disassembler = _get_capstone_disassembler(arch, capstone_mode)
            disasm = disassembler.disasm(instr_bytes, self.addr)
            try:
                instr = next(disasm)
                # Check if the mnemonic is the same
                if is_same_mnemonic(instr.mnemonic, mnemonic):
                    return capstone_mode
            except StopIteration:
                pass

        # We have not being lucky
        raise TypeError(
            f"Cannot guess {self.program.name} ISA for the instruction at address {self.addr:#x} "
            f"(consider setting it manually)"
        )

    def _disassemble(
        self, bb_asm: bytes, correct_mnemonic: str, correct_size: int
    ) -> list[capstone.CsInsn]:
        """
        Disassemble the basic block using capstone trying to guess the instruction set
        when unable to determine it from binexport.

        :param bb_asm: basic block raw bytes
        :param correct_mnemonic: The correct mnemonic of the first instruction of the
                                 basic block
        :param correct_size: The right size of the first instruction of the basic block
        """

        # Check if we already have a capstone context, if so use it
        if self.program.cs:
            return list(self.program.cs.disasm(bb_asm, self.addr))

        # Continue with the old method
        arch = self.program.architecture_name
        capstone_mode = 0

        # No need to guess the context for these arch
        if arch in ("x86", "x86-32", "x86-64", "MIPS-32", "MIPS-64", "ARM-64"):
            pass

        # For arm thumb use appropriate context guessing heuristics
        elif arch == "ARM-32":
            capstone_mode = self._guess_thumb_context(bb_asm[:correct_size], correct_mnemonic)

        # Everything else not yet supported
        else:
            raise NotImplementedError(f"The architecture {arch} is not yet supported in QBinDiff")

        # Set the program wide disassembler
        self.program.cs = _get_capstone_disassembler(arch, capstone_mode)
        return list(self.program.cs.disasm(bb_asm, self.addr))

    @property
    def program(self) -> ProgramBackendBinExport:
        """Wrapper on weak reference on ProgramBackendBinExport"""
        if (program := self._program()) is None:
            raise RuntimeError(
                "Trying to access an already expired weak reference on ProgramBackendBinExport"
            )
        return program

    @property
    def addr(self) -> Addr:
        return self.be_block.addr

    @property
    def instructions(self) -> Iterator[InstructionBackendBinExport]:
        """Returns an iterator over backend instruction objects"""

        # Generates the first instruction and use it to guess the context for capstone
        first_instr = next(iter(self.be_block.instructions.values()))
        capstone_instructions = self._disassemble(
            self.be_block.bytes, first_instr.mnemonic, len(first_instr.bytes)
        )

        # Then iterate over the instructions
        return (
            InstructionBackendBinExport(self.program.cs, instr) for instr in capstone_instructions
        )

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
        return (
            BasicBlockBackendBinExport(self._program, bb)
            for addr, bb in self.be_func.blocks.items()
        )

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


class ProgramBackendBinExport(AbstractProgramBackend):
    def __init__(self, file: str, *, arch: str | None = None, exec_path: str | None = None):
        super().__init__()

        self.be_prog = binexport.ProgramBinExport(file)
        self.architecture_name = self.be_prog.architecture
        self._exec_path: str | None = exec_path
        self._fun_names: dict[str, Addr] = {}  # {fun_name : fun_address}
        self.cs = None

        # Check if the architecture is set by the user
        if arch:
            # Parse the architecture
            self.cs = parse_architecture_flag(arch)
            if not self.cs:
                raise Exception("Unable to instantiate capstone context from given arch: %s" % arch)
        else:
            logging.info(
                "No architecture set but BinExport backend is used. If invalid instructions"
                " are found consider setting manually the architecture"
            )
            # self.cs will be set at basic block level
            # self.cs = _get_capstone_disassembler(self.be_prog.architecture)

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

    @property
    def exec_path(self) -> str:
        """
        Returns the executable path if it has been provided, otherwise try to guess it
        by removing the final .BinExport suffix
        """
        if self._exec_path:
            return self._exec_path
        return self.name.replace(".BinExport", "")  # Try to guess it as best effort

    @property
    def export_path(self) -> str:
        """
        Returns the .Binexport path
        """
        return str(self.be_prog.path)

    @property
    def capabilities(self) -> ProgramCapability:
        """
        Returns the supported capabilities
        """
        return ProgramCapability.INSTR_GROUP
