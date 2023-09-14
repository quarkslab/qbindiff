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

"""Quokka backend loader
"""

# local imports
from __future__ import annotations
import logging
import weakref
from functools import cached_property
from collections.abc import Iterator
from typing import Any, TypeAlias

# third party imports
import quokka
import quokka.types
import networkx
import capstone
from capstone import CS_OP_REG, CS_OP_IMM, CS_OP_MEM, CS_GRP_JUMP

# local imports
from qbindiff.loader import Data, Structure
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader.types import (
    FunctionType,
    DataType,
    StructureType,
    ReferenceType,
    ReferenceTarget,
    OperandType,
)
from qbindiff.types import Addr


# Aliases
capstoneOperand: TypeAlias = Any  # Relaxed typing


# ===== General purpose utils functions =====
def convert_data_type(qbe_data_type: quokka.types.DataType) -> DataType:
    """
    Convert a quokka DataType to qbindiff DataType

    :param qbe_data_type: the Quokka datatype to convert
    :return: the corresponding qbindiff datatype
    """

    if qbe_data_type == quokka.types.DataType.ASCII:
        return DataType.ASCII
    elif qbe_data_type == quokka.types.DataType.BYTE:
        return DataType.BYTE
    elif qbe_data_type == quokka.types.DataType.WORD:
        return DataType.WORD
    elif qbe_data_type == quokka.types.DataType.DOUBLE_WORD:
        return DataType.DOUBLE_WORD
    elif qbe_data_type == quokka.types.DataType.QUAD_WORD:
        return DataType.QUAD_WORD
    elif qbe_data_type == quokka.types.DataType.OCTO_WORD:
        return DataType.OCTO_WORD
    elif qbe_data_type == quokka.types.DataType.FLOAT:
        return DataType.FLOAT
    elif qbe_data_type == quokka.types.DataType.DOUBLE:
        return DataType.DOUBLE
    else:
        return DataType.UNKNOWN


def convert_struct_type(qbe_struct_type: quokka.types.StructureType) -> StructureType:
    """
    Convert a quokka StructureType to qbindiff StructureType

    :param qbe_struct_type: the Quokka structure to convert
    :return: the corresponding qbindiff structure
    """

    if qbe_struct_type == quokka.types.StructureType.ENUM:
        return StructureType.ENUM
    elif qbe_struct_type == quokka.types.StructureType.STRUCT:
        return StructureType.STRUCT
    elif qbe_struct_type == quokka.types.StructureType.UNION:
        return StructureType.UNION
    else:
        return StructureType.UNKNOWN


def convert_ref_type(qbe_ref_type: quokka.types.ReferenceType) -> ReferenceType:
    """
    Convert a quokka ReferenceType to qbindiff ReferenceType

    :param qbe_ref_type: the Quokka reference to convert
    :return: the corresponding qbindiff reference
    """

    if qbe_ref_type == quokka.types.ReferenceType.DATA:
        return ReferenceType.DATA
    elif qbe_ref_type == quokka.types.ReferenceType.ENUM:
        return ReferenceType.ENUM
    elif qbe_ref_type == quokka.types.ReferenceType.STRUC:
        return ReferenceType.STRUC
    else:
        return ReferenceType.UNKNOWN


# ===========================================


class OperandBackendQuokka(AbstractOperandBackend):
    """
    Backend loader of an Operand using Quokka
    """

    def __init__(
        self,
        cs_instruction: "capstone.CsInsn",
        cs_operand: capstoneOperand,
        cs_operand_position: int,
    ):
        super(OperandBackendQuokka, self).__init__()

        self.cs_instr = cs_instruction
        self.cs_operand = cs_operand
        self.cs_operand_position = cs_operand_position

    def __str__(self) -> str:
        return self.cs_instr.op_str.split(",")[self.cs_operand_position]

    @property
    def value(self) -> int | None:
        """
        Returns the immediate value (not addresses).
        """

        if self.is_immediate():
            return self.cs_operand.value.imm
        return None

    @property
    def type(self) -> OperandType:
        """Returns the capstone operand type"""
        op = self.cs_operand
        typ = OperandType.unknown
        cs_op_type = self.cs_operand.type

        if cs_op_type == capstone.CS_OP_REG:
            return OperandType.register
        elif cs_op_type == capstone.CS_OP_IMM:
            return OperandType.immediate
        elif cs_op_type == capstone.CS_OP_MEM:
            # A displacement is represented as [reg+hex] (example : [rdi+0x1234])
            # Then, base (reg) and disp (hex) should be different of 0
            if op.mem.base != 0 and op.mem.disp != 0:
                typ = OperandType.displacement
            # A phrase is represented as [reg1 + reg2] (example : [rdi + eax])
            # Then, base (reg1) and index (reg2) should be different of 0
            if op.mem.base != 0 and op.mem.index != 0:
                typ = OperandType.phrase
            if op.mem.disp != 0:
                typ = OperandType.displacement
        else:
            raise NotImplementedError(f"Unrecognized capstone type {cs_op_type}")
        return typ

    def is_immediate(self) -> bool:
        """
        Returns whether the operand is an immediate value (not considering addresses)
        """

        # Ignore jumps since the target is an immediate
        return self.type == OperandType.immediate and not self.cs_instr.group(capstone.CS_GRP_JUMP)


class InstructionBackendQuokka(AbstractInstructionBackend):
    """
    Backend loader of a Instruction using Quokka
    """

    def __init__(
        self,
        program: weakref.ref[ProgramBackendQuokka],
        qb_instruction: quokka.instruction.Instruction,
    ):
        super(InstructionBackendQuokka, self).__init__()

        self.program = program
        self.qb_instr = qb_instruction
        self.cs_instr = qb_instruction.cs_inst
        if self.cs_instr is None:
            logging.error(
                f"Capstone could not disassemble instruction at 0x{self.qb_instr.address:x} {self.qb_instr}"
            )

    def __del__(self):
        """
        Clean quokka internal state to deallocate memory
        """

        # Clear the reference to capstone object
        self.qb_instr._cs_instr = None
        # Unload cached instruction
        block = self.qb_instr.parent
        block._raw_dict[self.qb_instr.address] = self.qb_instr.proto_index

    def _cast_references(
        self, references: list[quokka.types.ReferenceTarget]
    ) -> list[ReferenceTarget]:
        """
        Cast the quokka references to qbindiff reference types

        :param references: list of Quokka references
        :return: list of corresponding references cast to qbindiff type
        """

        ret_ref = []
        for ref in references:
            match ref:
                case quokka.data.Data():
                    data_type = convert_data_type(ref.type)
                    ret_ref.append(Data(data_type, ref.address, ref.value))
                case quokka.structure.Structure(name=name):
                    ret_ref.append(self.program().get_structure(name))
                case quokka.structure.StructureMember(structure=qbe_struct, name=name):
                    ret_ref.append(
                        self.program().get_structure(qbe_struct.name).member_by_name(name)
                    )
                case quokka.Instruction():  # Not implemented for now
                    logging.warning("Skipping instruction reference")
        return ret_ref

    @property
    def addr(self) -> Addr:
        """
        The address of the instruction
        """
        return self.qb_instr.address

    @property
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        """
        return self.qb_instr.mnemonic

    @cached_property
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """
        Returns all the references towards the instruction

        :return: dictionary with reference type as key and the corresponding references list as values
        """

        ref = {}
        for ref_type, references in self.qb_instr.references.items():
            ref[convert_ref_type(ref_type)] = self._cast_references(references)
        return ref

    @property
    def operands(self) -> Iterator[OperandBackendQuokka]:
        """
        Returns an iterator over backend operand objects

        :return: list of Quokka operands
        """

        if self.cs_instr is None:
            return iter([])
        return (
            OperandBackendQuokka(self.cs_instr, o, i) for i, o in enumerate(self.cs_instr.operands)
        )

    @property
    def groups(self) -> list[str]:
        """
        Returns a list of groups of this instruction. Groups are capstone based but enriched.
        """

        return []  # Not supported

    @property
    def id(self) -> int:
        """
        Returns the capstone instruction ID as a non negative int. The ID is in the range [0, MAX_ID].
        The id is MAX_ID if there is no capstone instruction available.
        """
        if self.cs_instr is None:
            return self.MAX_ID  # Custom defined value representing a "unknown" instruction
        return self.cs_instr.id

    @property
    def comment(self) -> str:
        """
        Comment associated with the instruction
        """
        return ""  # Not supported

    @property
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the instruction
        """
        return bytes(self.qb_instr.bytes)


class BasicBlockBackendQuokka(AbstractBasicBlockBackend):
    """
    Backend loader of a BasicBlock using Quokka
    """

    def __init__(self, program: weakref.ref[ProgramBackendQuokka], qb_block: quokka.block.Block):
        super(BasicBlockBackendQuokka, self).__init__()

        self.qb_block = qb_block
        self.program = program

        # Private attributes
        self._addr = qb_block.start

    def __del__(self) -> None:
        """
        Clean quokka internal state by unloading from memory the Block object

        :return: None
        """

        chunk = self.qb_block.parent
        chunk._raw_dict[self.qb_block.start] = self.qb_block.proto_index

    def __len__(self) -> int:
        """
        The numbers of instructions in the basic block
        """
        return len(self.qb_block)

    @property
    def addr(self) -> Addr:
        """
        The address of the basic block
        """

        return self._addr

    @property
    def instructions(self) -> Iterator[InstructionBackendQuokka]:
        """
        Returns an iterator over backend instruction objects

        :return: iterator over Quokka instructions
        """
        return (
            InstructionBackendQuokka(self.program, instr) for instr in self.qb_block.instructions
        )

    @property
    def bytes(self) -> bytes:
        return b"".join(x.bytes for x in self.instructions)


class FunctionBackendQuokka(AbstractFunctionBackend):
    """
    Backend loader of a Function using Quokka
    """

    def __init__(
        self, program: weakref.ref[ProgramBackendQuokka], qb_func: quokka.function.Function
    ):
        super(FunctionBackendQuokka, self).__init__()

        self.qb_prog = qb_func.program
        self.qb_func = qb_func
        self.program = program

        # [TODO] Init all the properties and free the memory of qb_prog/qb_func

    @property
    def basic_blocks(self) -> Iterator[BasicBlockBackendQuokka]:
        """
        Returns an iterator over backend basic blocks objects

        :return: Iterator over the Quokka Basic Blocks
        """

        # Stop the exploration if it's an imported function
        if self.is_import():
            return iter([])

        return (
            BasicBlockBackendQuokka(self.program, self.qb_func.get_block(addr))
            for addr in self.qb_func.graph.nodes
        )

    @property
    def addr(self) -> Addr:
        """
        The address of the function
        """
        return self.qb_func.start

    @property
    def graph(self) -> networkx.DiGraph:
        """
        The Control Flow Graph of the function
        """
        return self.qb_func.graph

    @cached_property
    def parents(self) -> set[Addr]:
        """
        Set of function parents in the call graph
        """
        parents = set()
        for chunk in self.qb_func.callers:
            try:
                for func in self.qb_prog.get_function_by_chunk(chunk):
                    parents.add(func.start)
            except IndexError:
                pass  # Sometimes there can be a chunk that is not part of any function
        return parents

    @cached_property
    def children(self) -> set[Addr]:
        """
        Set of function children in the call graph
        """
        children = set()
        for chunk in self.qb_func.calls:
            try:
                for func in self.qb_prog.get_function_by_chunk(chunk):
                    children.add(func.start)
            except IndexError:
                pass  # Sometimes there can be a chunk that is not part of any function
        return children

    @cached_property
    def type(self) -> FunctionType:
        """
        The type of the function (as defined by IDA)
        """

        f_type = self.qb_func.type
        if f_type == quokka.types.FunctionType.NORMAL:
            return FunctionType.normal
        elif f_type == quokka.types.FunctionType.IMPORTED:
            return FunctionType.imported
        elif f_type == quokka.types.FunctionType.LIBRARY:
            return FunctionType.library
        elif f_type == quokka.types.FunctionType.THUNK:
            return FunctionType.thunk
        elif f_type == quokka.types.FunctionType.EXTERN:
            return FunctionType.extern
        elif f_type == quokka.types.FunctionType.INVALID:
            return FunctionType.invalid
        else:
            raise NotImplementedError(f"Function type {f_type} not implemented")

    @property
    def name(self) -> str:
        """
        The name of the function
        """

        return self.qb_func.name

    def is_import(self) -> bool:
        """
        True if the function is imported

        :return: whether the fonction is imported
        """

        # Should we consider also FunctionType.thunk?
        return self.type in (FunctionType.imported, FunctionType.extern)


class ProgramBackendQuokka(AbstractProgramBackend):
    """
    Backend loader of a Program using Quokka
    """

    def __init__(self, export_path: str, exec_path: str):
        super(ProgramBackendQuokka, self).__init__()

        self.qb_prog = quokka.Program(export_path, exec_path)
        self._exec_path = exec_path

        self._callgraph = networkx.DiGraph()
        self._fun_names = {}  # {fun_name : fun_address}

    @property
    def functions(self) -> Iterator[FunctionBackendQuokka]:
        """
        Returns an iterator over backend function objects
        """

        functions = {}
        for addr, func in self.qb_prog.items():
            # Pass a self (weak) reference for performance
            f = FunctionBackendQuokka(weakref.ref(self), func)
            if addr in functions:
                logging.error("Address collision for 0x%x" % addr)
            functions[addr] = f
            self._fun_names[f.name] = addr

            # Load the callgraph
            self._callgraph.add_node(addr)
            for c_addr in f.children:
                self._callgraph.add_edge(addr, c_addr)
            for p_addr in f.parents:
                self._callgraph.add_edge(p_addr, addr)

        return iter(functions.values())

    @property
    def name(self) -> str:
        return self.qb_prog.executable.exec_file.name

    @cached_property
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

    @cached_property
    def structures_by_name(self) -> dict[str, Structure]:
        """
        Returns the dictionary {name: structure}
        """

        # Hoping that there won't be two struct with the same name
        return {struct.name: struct for struct in self.structures}

    def get_structure(self, name: str) -> Structure:
        """
        Returns structure identified by the name
        """

        return self.structures_by_name[name]

    @property
    def callgraph(self) -> networkx.DiGraph:
        """
        The callgraph of the program
        """

        return self._callgraph

    @property
    def fun_names(self) -> dict[str, int]:
        """
        Returns a dictionary with function name as key and the function address as value
        """

        return self._fun_names

    @property
    def exec_path(self) -> str:
        """
        Returns the executable path
        """

        return self._exec_path
