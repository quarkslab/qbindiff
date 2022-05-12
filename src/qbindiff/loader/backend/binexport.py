import logging, networkx, capstone
from collections import defaultdict
from functools import cache
from typing import Union

from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)
from qbindiff.loader.backend.binexport2_pb2 import BinExport2
from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand
from qbindiff.loader.types import LoaderType, FunctionType, OperandType
from qbindiff.types import Addr


# === General purpose binexport functions ===
def _get_instruction_address(pb, inst_idx):
    inst = pb.instruction[inst_idx]
    if inst.HasField("address"):
        return inst.address
    else:
        return _backtrack_instruction_address(pb, inst_idx)


def _backtrack_instruction_address(pb, idx):
    tmp_sz = 0
    tmp_idx = idx
    if tmp_idx == 0:
        return pb.instruction[tmp_idx].address
    while True:
        tmp_idx -= 1
        tmp_sz += len(pb.instruction[tmp_idx].raw_bytes)
        if pb.instruction[tmp_idx].HasField("address"):
            break
    return pb.instruction[tmp_idx].address + tmp_sz


def _get_basic_block_addr(pb, bb_idx):
    inst = pb.basic_block[bb_idx].instruction_index[0].begin_index
    return _get_instruction_address(pb, inst)


def _instruction_index_range(rng):
    return range(
        rng.begin_index, (rng.end_index if rng.end_index else rng.begin_index + 1)
    )


def _get_capstone_disassembler(binexport_arch: str):
    def capstone_context(arch, mode):
        context = capstone.Cs(arch, mode)
        context.detail = True
        return context

    if binexport_arch == "x86":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    elif binexport_arch == "x86-64":
        return capstone_context(capstone.CS_ARCH_X86, capstone.CS_MODE_64)

    raise NotImplementedError(f"Architecture {binexport_arch} has not be implemented")


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


class ProgramBackendBinExport(AbstractProgramBackend):
    def __init__(self, program: Program, file: str):
        super(ProgramBackendBinExport, self).__init__()

        self._pb = BinExport2()
        with open(file, "rb") as f:
            self.proto.ParseFromString(f.read())
        self._mask = (
            0xFFFFFFFF if self.architecture.endswith("32") else 0xFFFFFFFFFFFFFFFF
        )
        self._fun_names = {}
        self._fun_addr = set()
        self._callgraph = networkx.DiGraph()

        # Make the data refs map {instruction index -> address referred}
        data_refs = defaultdict(list)
        for entry in self.proto.data_reference[::-1]:
            data_refs[entry.instruction_index].append(entry.address)

        count_f = 0
        coll = 0
        # Load all the functions
        for i, pb_fun in enumerate(self.proto.flow_graph):
            f = Function(LoaderType.binexport, self, data_refs=data_refs, pb_fun=pb_fun)
            if f.addr in program:
                logging.error("Address collision for 0x%x" % f.addr)
                coll += 1
            program[f.addr] = f
            count_f += 1

        count_imp = 0
        # Load the callgraph
        cg = self.proto.call_graph
        for node in cg.vertex:
            self._callgraph.add_node(node.address)
            if node.address not in program and node.type == cg.Vertex.IMPORTED:
                program[node.address] = Function(
                    LoaderType.binexport,
                    self,
                    is_import=True,
                    addr=node.address,
                )
                count_imp += 1
            if node.address not in program:
                logging.error(
                    "Missing function address: 0x%x (%d)" % (node.address, node.type)
                )

            program[node.address].type = self.normalize_function_type(node.type)
            if node.demangled_name:
                program[node.address].name = node.demangled_name
            elif node.mangled_name:
                program[node.address].name = node.mangled_name

        for edge in cg.edge:
            src = cg.vertex[edge.source_vertex_index].address
            dst = cg.vertex[edge.target_vertex_index].address
            self._callgraph.add_edge(src, dst)
            program[src].children.add(dst)
            program[dst].parents.add(src)

        # Create a map of function names for quick lookup later on
        for f in program.values():
            self._fun_names[f.name] = f
            self._fun_addr.add(f.addr)

        logging.debug(
            "total all:%d, imported:%d collision:%d (total:%d)"
            % (count_f, count_imp, coll, (count_f + count_imp + coll))
        )

    def addr_mask(self, value):
        return value & self._mask

    @property
    def proto(self):
        return self._pb

    @property
    def name(self):
        return self.proto.meta_information.executable_name

    @property
    def architecture(self):
        return self.proto.meta_information.architecture_name

    def __repr_(self):
        return "<Program:%s>" % self.name

    @property
    def callgraph(self) -> networkx.DiGraph:
        return self._callgraph

    def get_function(self, name: str) -> Function:
        """Returns the qbindiff Function object associated with the function `name`"""
        if name in self._fun_names:
            return self._fun_names[name]
        return None

    def has_function(self, key: Union[str, Addr]) -> bool:
        """
        Returns True if the function exists, False otherwise.
        The parameter `key` can either be the function name or the function address.
        """
        match key:
            case str(name):
                return name in self._fun_names
            case int(addr):
                return addr in self._fun_addr

    def normalize_function_type(
        self, f_type: BinExport2.CallGraph.Vertex.Type
    ) -> FunctionType:
        """Convert a BinExport function type to a FunctionType"""

        if f_type == BinExport2.CallGraph.Vertex.NORMAL:
            return FunctionType.normal
        if f_type == BinExport2.CallGraph.Vertex.LIBRARY:
            return FunctionType.library
        if f_type == BinExport2.CallGraph.Vertex.IMPORTED:
            return FunctionType.imported
        if f_type == BinExport2.CallGraph.Vertex.THUNK:
            return FunctionType.thunk
        if f_type == BinExport2.CallGraph.Vertex.INVALID:
            return FunctionType.invalid
        raise NotImplementedError(f"Function type {f_type} not implemented")


class FunctionBackendBinExport(AbstractFunctionBackend):
    def __init__(
        self,
        function: Function,
        be_program: ProgramBackendBinExport,
        data_refs: defaultdict[int, list[Addr]] = None,
        pb_fun: BinExport2.FlowGraph = None,
        is_import: bool = False,
        addr: Addr = None,
    ):
        super(FunctionBackendBinExport, self).__init__()

        # Private attributes
        self._addr = addr  # Optional address
        self._parents = set()
        self._children = set()
        self._graph = networkx.DiGraph()
        self._basic_blocks_addr = set()
        self._type = None  # Set by the Program constructor
        self._name = None  # Set by the Program constructor (mangled name)

        if is_import:
            if self._addr is None:
                logging.error("Missing function address for imported function")
            return

        assert (
            data_refs is not None and pb_fun is not None
        ), "data_refs and pb_fun must be provided"

        self._addr = _get_basic_block_addr(
            be_program.proto, pb_fun.entry_basic_block_index
        )

        # Load the basic blocks
        bb_i2a = {}  # Map {basic block index -> basic block address}
        bb_count = 0
        for bb_idx in pb_fun.basic_block_index:
            bb_count += 1
            basic_block = BasicBlock(
                LoaderType.binexport,
                be_program,
                self,
                be_program.proto.basic_block[bb_idx],
                data_refs,
            )

            if basic_block.addr in function:
                logging.error(
                    "0x%x basic block address (0x%x) already in(idx:%d)"
                    % (self.addr, basic_block.addr, bb_idx)
                )

            function[basic_block.addr] = basic_block
            bb_i2a[bb_idx] = basic_block.addr
            self._graph.add_node(basic_block.addr)
            self._basic_blocks_addr.add(basic_block.addr)

        if bb_count != len(function):
            logging.error(
                "Wrong basic block number %x, bb:%d, self:%d"
                % (self.addr, len(pb_fun.basic_block_index), len(function))
            )

        # Load the edges between blocks
        for edge in pb_fun.edge:
            bb_src = bb_i2a[edge.source_basic_block_index]
            bb_dst = bb_i2a[edge.target_basic_block_index]
            self._graph.add_edge(bb_src, bb_dst)

    @property
    def addr(self) -> Addr:
        """The address of the function"""
        return self._addr

    @property
    def graph(self) -> networkx.DiGraph:
        return self._graph

    @property
    def parents(self) -> set[Addr]:
        """Set of function parents in the call graph"""
        return self._parents

    @property
    def children(self) -> set[Addr]:
        """Set of function children in the call graph"""
        return self._children

    @property
    def name(self):
        return self._name if self._name else "sub_%X" % self.addr

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def type(self) -> FunctionType:
        return self._type

    @type.setter
    def type(self, value: FunctionType):
        self._type = value

    def has_basic_block(self, addr: Addr) -> bool:
        """Returns True if the basic block with address `addr` exists, False otherwise."""
        return addr in self._basic_blocks_addr


class BasicBlockBackendBinExport(AbstractBasicBlockBackend):
    def __init__(
        self,
        basic_block: BasicBlock,
        be_program: ProgramBackendBinExport,
        be_function: FunctionBackendBinExport,
        pb_basic_block: BinExport2.BasicBlock,
        data_refs: defaultdict[int, list[Addr]],
    ):
        super(BasicBlockBackendBinExport, self).__init__()

        # Private attributes
        self._addr = None

        bb_asm = b""
        instr_count = 0

        # Ranges are in fact the true basic blocks but BinExport for some reason likes
        # to merge multiple basic blocks into one.
        # For example: BB_1 -- unconditional_jmp --> BB_2
        # might be merged into a single basic block
        for rng in pb_basic_block.instruction_index:
            for idx in _instruction_index_range(rng):
                pb_inst = be_program.proto.instruction[idx]
                inst_addr = _get_instruction_address(be_program.proto, idx)

                # The first instruction determines the basic block address
                if self._addr is None:
                    self._addr = inst_addr

                bb_asm += pb_inst.raw_bytes
                instr_count += 1

        md = _get_capstone_disassembler(
            be_program.proto.meta_information.architecture_name
        )
        instructions = list(md.disasm(bb_asm, self.addr))
        if len(instructions) != instr_count:
            logging.error(
                f"Mismatch between the disassembly of capstone and IDA for the basic block at address 0x{self.addr:x}"
            )
            exit(1)

        i = 0
        # Iterate once again to build the Instruction
        for rng in pb_basic_block.instruction_index:
            for idx in _instruction_index_range(rng):
                pb_inst = be_program.proto.instruction[idx]
                inst_addr = _get_instruction_address(be_program.proto, idx)

                basic_block.append(
                    Instruction(
                        LoaderType.binexport,
                        be_program,
                        be_function,
                        inst_addr,
                        idx,
                        data_refs,
                        instructions[i],
                    )
                )
                i += 1

    @property
    def addr(self) -> Addr:
        return self._addr


class InstructionBackendBinExport(AbstractInstructionBackend):
    def __init__(
        self,
        be_program: ProgramBackendBinExport,
        be_function: FunctionBackendBinExport,
        addr: Addr,
        i_idx: int,
        data_refs: defaultdict[int, list[Addr]],
        cs_instr: capstone.CsInsn,
    ):
        super(InstructionBackendBinExport, self).__init__()

        # Private attributes
        self._addr = addr
        self._be_program = be_program
        self._be_function = be_function
        self._instruction = be_program.proto.instruction[i_idx]
        self.data_refs = data_refs[i_idx]
        self.comment_index = be_program.proto.instruction[i_idx].comment_index
        self.cs_instr = cs_instr

    @property
    def addr(self):
        return self._addr

    @property
    def mnemonic(self):
        return self._be_program.proto.mnemonic[self._instruction.mnemonic_index].name

    @property
    @cache
    def data_references(self) -> set[Addr]:
        """Returns the collections of addresses that are accessed by the instruction"""
        return set(self.data_refs)

    @property
    @cache
    def operands(self):
        return [
            Operand(LoaderType.binexport, self.cs_instr, o)
            for o in self.cs_instr.operands
        ]

    @property
    def groups(self):
        return []  # not supported

    @property
    def capstone(self) -> "capstone.CsInsn":
        """Return the capstone instruction"""
        return self.cs_instr

    @property
    def comment(self):
        proto = self._be_program.proto  # Alias
        str_comment = ""

        for comment_idx in self.comment_index:
            if proto.comment[comment_idx].HasField("string_table_index"):
                string_idx = proto.comment[comment_idx].string_table_index

                if len(str_comment) > 0:
                    str_comment += "\n"  # Separator between comments
                str_comment += proto.string_table[string_idx]

        return str_comment


class OperandBackendBinexport(AbstractOperandBackend):
    def __init__(self, cs_instruction, cs_operand):
        super(OperandBackendBinexport, self).__init__()

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
            op_str += "]"
            return op_str
        else:
            raise NotImplementedError(f"Unrecognized capstone type {self.type}")

    @property
    def type(self) -> int:
        """Returns the capstone operand type"""
        return self.cs_operand.type

    @property
    def value(self):
        """Returns the capstone operand value"""
        return self.cs_operand.value
