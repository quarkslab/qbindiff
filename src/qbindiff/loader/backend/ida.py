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

"""IDA backend loader

It uses directly the IDA API to load the disassembly analysis.
"""

import networkx
import weakref
from functools import cached_property
from collections.abc import Iterator

# Ida API
import idautils
import ida_nalt
import ida_funcs
import ida_struct
import ida_idaapi
import ida_bytes
import ida_gdl
import ida_bytes
import ida_ua
import ida_lines
import ida_name

# local imports
from qbindiff.loader import Structure
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader.types import (
    DataType,
    StructureType,
    FunctionType,
    ReferenceType,
    ReferenceTarget,
    OperandType,
)
from qbindiff.types import Addr


# ===== General purpose utils functions =====


def extract_data_type(ida_flag: int) -> DataType:
    """
    Extract from a IDA flag the correct DataType

    :param ida_flag: the IDA flag
    :return: the corresponding qbindiff DataType
    """

    if ida_bytes.is_byte(ida_flag):
        return DataType.BYTE
    elif ida_bytes.is_word(ida_flag):
        return DataType.WORD
    elif ida_bytes.is_dword(ida_flag):
        return DataType.DOUBLE_WORD
    elif ida_bytes.is_qword(ida_flag):
        return DataType.QUAD_WORD
    elif ida_bytes.is_oword(ida_flag):
        return DataType.OCTO_WORD
    elif ida_bytes.is_float(ida_flag):
        return DataType.FLOAT
    elif ida_bytes.is_double(ida_flag):
        return DataType.DOUBLE
    elif ida_bytes.is_strlit(ida_flag):
        return DataType.ASCII
    else:
        return DataType.UNKNOWN


# ===========================================


class ImportManager:
    """
    Class responsible for identifying imported functions.
    This is needed as there is no direct IDA API to know if a function is external or not
    """

    def __init__(self):
        # { address : (name, ordinal) }
        self.imports: dict[Addr, tuple(str, int)] = {}

        for i in range(ida_nalt.get_import_module_qty()):
            ida_nalt.enum_import_names(i, lambda ea, name, ord: self.handle_import(ea, name, ord))

    def handle_import(self, ea: Addr, name: str, ord: int) -> int:
        """Handle the imported function by adding it to the collection"""

        if name == "":
            # IDA might know the name even though it is not reporting it now
            flags = ida_bytes.get_flags(ea)
            if ida_bytes.has_user_name(flags):
                name = ida_name.get_short_name(ea)

        self.imports[ea] = (name, ord)
        return 1

    def is_import(self, addr: Addr) -> bool:
        """Returns True if the address specified refers to an imported function"""

        return addr in self.imports


class OperandBackendIDA(AbstractOperandBackend):
    def __init__(self, ida_operand: ida_ua.op_t, inst_addr: Addr):
        self._addr = inst_addr
        self.ida_op = ida_operand

    def __str__(self) -> str:
        return ida_lines.tag_remove(ida_ua.print_operand(self._addr, self.ida_op.n))

    @property
    def value(self) -> int | None:
        """
        Returns the immediate value (not addresses) if the operand is constant.
        If not, None is returned.
        """

        if self.is_immediate():
            return self.ida_op.value
        return None

    @property
    def type(self) -> OperandType:
        """
        Returns the operand type.
        """

        match self.ida_op.type:
            case ida_ua.o_reg:
                return OperandType.register
            case ida_ua.o_mem:
                return OperandType.memory
            case ida_ua.o_phrase:
                return OperandType.phrase
            case ida_ua.o_displ:
                return OperandType.displacement
            case ida_ua.o_imm:
                return OperandType.immediate
            case ida_ua.o_far:
                return OperandType.far
            case ida_ua.o_near:
                return OperandType.near
            case _:
                return OperandType.unknown

    def is_immediate(self) -> bool:
        """
        Returns whether the operand is an immediate value (not considering addresses)
        """

        # Ignore jumps since the target is an immediate
        return self.type == OperandType.immediate


class InstructionBackendIDA(AbstractInstructionBackend):
    def __init__(self, addr: Addr):
        super(InstructionBackendIDA, self).__init__()

        self._addr = addr
        self.insn = ida_ua.insn_t()
        ida_ua.decode_insn(self.insn, addr)

    def __str__(self):
        return f"{self.mnemonic} {', '.join((str(op) for op in self.operands))}"

    @property
    def addr(self) -> Addr:
        """
        The address of the instruction
        """

        return self._addr

    @property
    def mnemonic(self) -> str:
        """
        Returns the instruction mnemonic as a string
        """

        return ida_ua.ua_mnem(self.addr)

    @property
    def references(self) -> dict[ReferenceType, list[ReferenceTarget]]:
        """
        Returns all the references towards the instruction
        """

        return {}  # TODO: to implement

    @property
    def operands(self) -> Iterator[OperandBackendIDA]:
        """
        Returns an iterator over backend operand objects
        """

        return (
            OperandBackendIDA(op, self.addr)
            for op in self.insn.ops
            if op.type != ida_ua.o_void and op.shown()
        )

    @property
    def groups(self) -> list[str]:
        """
        Returns a list of groups of this instruction
        """

        return []  # Not implemented for IDA backend

    @property
    def id(self) -> int:
        """
        Returns the IDA instruction ID as a non negative int. The ID is in the range [0, MAX_ID].
        The value MAX_ID means that there is no ID available.
        """

        return self.insn.itype

    @property
    def comment(self) -> str:
        """
        Comment associated with the instruction
        """

        return ida_bytes.get_cmt(self.addr, True) or ""  # return repeatable ones

    @property
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the instruction
        """

        return ida_bytes.get_bytes(self.addr, self.insn.size)


class BasicBlockBackendIDA(AbstractBasicBlockBackend):
    def __init__(self, start_addr: Addr, end_addr: Addr):
        super(BasicBlockBackendIDA, self).__init__()

        self._start_addr = start_addr
        self._end_addr = end_addr
        self._size = len(list(idautils.Heads(start_addr, end_addr)))

    def __len__(self) -> int:
        """
        The numbers of instructions in the basic block
        """

        return self._size

    @property
    def addr(self) -> Addr:
        """
        The address of the basic block
        """

        return self._start_addr

    @property
    def instructions(self) -> Iterator[InstructionBackendIDA]:
        """
        Returns an iterator over backend instruction objects
        """

        return (
            InstructionBackendIDA(addr) for addr in idautils.Heads(self._start_addr, self._end_addr)
        )

    @property
    def bytes(self) -> bytes:
        """
        Returns the bytes representation of the basic block
        """

        return ida_bytes.get_bytes(self._start_addr, self._end_addr - self._start_addr)


class FunctionBackendIDA(AbstractFunctionBackend):
    def __init__(
        self, program: weakref.ref["ProgramBackendIDA"], addr: Addr, import_manager: ImportManager
    ):
        super(FunctionBackendIDA, self).__init__()

        self._program = program
        self._addr = addr
        self.import_manager = import_manager
        self._ida_fun = ida_funcs.get_func(addr)
        self._cfg = networkx.DiGraph()

        self._load_cfg()

    def _load_cfg(self) -> None:
        """Load the CFG in memory"""

        for block in ida_gdl.FlowChart(self._ida_fun, flags=ida_gdl.FC_NOEXT):
            self._cfg.add_node(block.start_ea)
            for parent in block.preds():
                self._cfg.add_edge(parent.start_ea, block.start_ea)
            for child in block.succs():
                self._cfg.add_edge(block.start_ea, child.start_ea)

    @property
    def basic_blocks(self) -> Iterator[BasicBlockBackendIDA]:
        """
        Returns an iterator over backend basic blocks objects.

        :return: Iterator over the IDA Basic Blocks
        """

        if self.is_import():
            return iter([])

        return (
            BasicBlockBackendIDA(block.start_ea, block.end_ea)
            for block in ida_gdl.FlowChart(self._ida_fun, flags=ida_gdl.FC_NOEXT)
        )

    @property
    def addr(self) -> Addr:
        """
        The address of the function.
        """

        return self._addr

    @property
    def graph(self) -> networkx.DiGraph:
        """
        The Control Flow Graph of the function.
        """

        return self._cfg

    @cached_property
    def parents(self) -> set[Addr]:
        """
        Set of function parents in the call graph.
        """

        return set(self._program().callgraph.predecessors(self.addr))

    @cached_property
    def children(self) -> set[Addr]:
        """
        Set of function children in the call graph.
        """

        return set(self._program().callgraph.successors(self.addr))

    @property
    def type(self) -> FunctionType:
        """
        The type of the function.
        """

        if self._ida_fun.flags & ida_funcs.FUNC_THUNK:
            return FunctionType.thunk
        elif self._ida_fun.flags & ida_funcs.FUNC_LIB:
            return FunctionType.library
        elif self.import_manager.is_import(self.addr):
            return FunctionType.imported
        else:
            return FunctionType.normal

    @property
    def name(self) -> str:
        """
        The name of the function.
        """

        return ida_funcs.get_func_name(self.addr)

    def is_import(self) -> bool:
        """
        True if the function is imported
        """

        return self.type == FunctionType.imported


class ProgramBackendIDA(AbstractProgramBackend):
    """
    Backend loader of a Program using idapython API
    """

    def __init__(self):
        super(ProgramBackendIDA, self).__init__()

        self._callgraph = None
        self._fun_names = {}  # {fun_name : fun_address}
        self.import_manager = ImportManager()

    def __repr__(self):
        return f"<Program:{self.name}>"

    @property
    def functions(self) -> Iterator[FunctionBackendIDA]:
        """
        Returns an iterator over backend function objects.
        """

        functions = []
        self._callgraph = networkx.DiGraph()

        for fun_addr in idautils.Functions():
            functions.append(FunctionBackendIDA(weakref.ref(self), fun_addr, self.import_manager))
            self._fun_names[ida_funcs.get_func_name(fun_addr)] = fun_addr

            # Load the callgraph
            self._callgraph.add_node(fun_addr)
            for xref_addr in idautils.CodeRefsTo(fun_addr, 1):
                fun_parent = ida_funcs.get_func(xref_addr)
                if fun_parent:
                    self._callgraph.add_edge(fun_parent.start_ea, fun_addr)

        return iter(functions)

    @property
    def name(self) -> str:
        """
        The name of the program.
        """

        return ida_nalt.get_root_filename()

    @cached_property
    def structures(self) -> list[Structure]:
        """
        Returns the list of structures defined in program.
        """

        # TODO add enums
        struct_list = []
        for idx in range(ida_struct.get_struc_qty()):
            struct_id = ida_struct.get_struc_by_idx(idx)
            struct_name = ida_struct.get_struc_name(struct_id)
            struct_size = ida_struct.get_struc_size(struct_id)
            struct_type = (
                StructureType.UNION if ida_struct.is_union(struct_id) else StructureType.STRUCT
            )

            struct = Structure(struct_type, struct_name, struct_size)
            struct_list.append(struct)

            # Add members
            struct_ida = ida_struct.get_struc(struct_id)
            if struct_ida.memqty > 0:
                offset = ida_struct.get_struc_first_offset(struct_ida)

                while offset != ida_idaapi.BADADDR:
                    member_ida = ida_struct.get_member(struct_ida, offset)

                    # A None value might be the terminator of the struct
                    if member_ida is not None:
                        struct.add_member(
                            offset,
                            extract_data_type(member_ida.flag),
                            ida_struct.get_member_name(member_ida.id),
                            ida_struct.get_member_size(member_ida),
                            None,  # No default value
                        )

                    offset = ida_struct.get_struc_next_offset(struct_ida, offset)

        return struct_list

    @property
    def callgraph(self) -> networkx.DiGraph:
        """
        The callgraph of the program.
        """

        if self._callgraph is None:
            raise ValueError(
                "Callgraph not populated yet. You have to load the functions "
                "before accessing the callgraph"
            )
        return self._callgraph

    @property
    def fun_names(self) -> dict[str, Addr]:
        """
        Returns a dictionary with function name as key and the function address as value.
        """

        return self._fun_names

    @property
    def exec_path(self) -> str:
        """
        Returns the executable path
        """

        return ida_nalt.get_input_file_path()
