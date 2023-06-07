from __future__ import absolute_import
import networkx
from typing import List, Dict, Iterator, Set
from functools import cached_property

from qbindiff.loader.function import Function
from qbindiff.loader.types import LoaderType, OperandType, FunctionType, ReferenceType, ReferenceTarget
from qbindiff.loader.instruction import Instruction
from qbindiff.types import Addr
# local imports
from qbindiff.loader.backend import (
    AbstractProgramBackend,
    AbstractFunctionBackend,
    AbstractBasicBlockBackend,
    AbstractInstructionBackend,
    AbstractOperandBackend,
)

import ida_nalt
import idautils
import ida_funcs
import ida_gdl
import ida_idaapi
import ida_bytes
import ida_ua
import ida_lines


class OperandBackendIDA(AbstractOperandBackend):
    def __init__(self, op_t, ea):
        self._addr = ea
        self.op_t = op_t

    @property
    def type(self) -> OperandType:
        return OperandType(self.op_t.type)

    def __str__(self) -> str:
        return ida_lines.tag_remove(ida_ua.print_operand(self._addr, self.op_t.n))

    def is_immediate(self) -> bool:
        """Returns whether the operand is an immediate value (not considering addresses)"""
        # Ignore jumps since the target is an immediate
        return self.type == OperandType.immediate

    @property
    def value(self) -> int | None:
        """
        Returns the immediate value (not addresses) used by the operand.
        """

        if self.is_immediate():
            return self.op_t.value
        return None


class InstructionBackendIDA(AbstractInstructionBackend):
    def __init__(self, addr):
        self._addr = addr
        self.insn = ida_ua.insn_t()
        ida_ua.decode_insn(self.insn, self.addr)
        self.nb_ops = self._nb_operand()

    def _nb_operand(self) -> int:
        return len([o for o in self.insn.ops if o.type != ida_ua.o_void])

    @property
    def addr(self) -> Addr:
        return self._addr

    @property
    def mnemonic(self) -> str:
        return ida_ua.ua_mnem(self.addr)

    @property
    def operands(self) -> Iterator[OperandBackendIDA]:
        return (OperandBackendIDA(self.insn.ops[i], self.addr) for i in range(self.nb_ops))

    @property
    def references(self) -> Dict[ReferenceType, List[ReferenceTarget]]:
        return {}  # TODO: to implement

    @property
    def groups(self) -> List[str]:
        return []  # Not implemented for IDA backend

    @property
    def id(self) -> int:
        """ Return the IDA itype of the instruction."""
        return self.insn.itype

    @property
    def comment(self) -> str:
        return ida_bytes.get_cmt(self.addr, True)  # return repeatable ones

    def __str__(self):
        return "%s %s" % (self.mnemonic, ", ".join((str(op) for op in self.operands)))

    @property
    def bytes(self) -> bytes:
        """Returns the bytes representation of the instruction """
        return ida_bytes.get_bytes(self.addr, self.insn.size)


class BasicBlockBackendIDA(AbstractBasicBlockBackend):
    def __init__(self, block: ida_gdl.BasicBlock):
        super(BasicBlockBackendIDA, self).__init__()

        self.ida_block = block

    @property
    def addr(self) -> Addr:
        return self.ida_block.start_ea

    @property
    def instructions(self) -> Iterator[InstructionBackendIDA]:
        """Returns an iterator over backend instruction objects"""
        # Then iterate over the instructions
        return (InstructionBackendIDA(x) for x in idautils.Heads(self.ida_block.start_ea, self.ida_block.end_ea))

    @property
    def bytes(self) -> bytes:
        return ida_bytes.get_bytes(self.ida_block.start_ea, self.ida_block.end_ea)


class FunctionBackendIDA(AbstractFunctionBackend):
    def __init__(self, addr, callgraph: networkx.DiGraph):
        self.pfn = ida_funcs.get_func(addr)
        self._cg = callgraph
        self.cfg = ida_gdl.FlowChart(self.pfn, flags=ida_gdl.FC_NOEXT)

    @property
    def addr(self) -> Addr:
        """The address of the function"""
        return self.pfn.start_ea

    @property
    def basic_blocks(self) -> Iterator[BasicBlockBackendIDA]:
        return (BasicBlockBackendIDA(x) for x in self.cfg)

    @property
    def name(self):
        return ida_funcs.get_func_name(self.addr)

    @property
    def type(self) -> FunctionType:
        return FunctionType(self.pfn.t)

    @cached_property
    def graph(self) -> networkx.DiGraph:
        g = networkx.DiGraph()
        for bb in self.cfg:
            for pred in bb.preds():
                g.add_edge(pred.id, bb.id)
            for succ in bb.succs():
                g.add_edge(bb.id, succ.id)
        return g

    @property
    def parents(self) -> Set[Addr]:
        return set(self._cg.predecessors(self.addr))

    @property
    def children(self) -> Set[Addr]:
        return set(self._cg.successors(self.addr))


class ProgramBackendIDA(object):
    def __init__(self, program):
        self._program = program
        self._load_functions()
        self._load_call_graph()
        self._callgraph = networkx.DiGraph()

    @property
    def name(self):
        return ida_nalt.get_root_filename()

    @property
    def exec_file(self) -> str:
        return ida_nalt.get_input_file_path()

    def _load_functions(self):
        for fun_addr in idautils.Functions():
            self._program[fun_addr] = Function(LoaderType.ida, fun_addr)

    def _load_call_graph(self):
        for fun_addr in self._program.keys():
            self._callgraph.add_node(fun_addr)
            for pred in idautils.CodeRefsTo(fun_addr, 1):
                f_pred = ida_funcs.get_func(pred)
                if f_pred:
                    pred = f_pred.start_ea
                    self._callgraph.add_edge(pred, fun_addr)
                    self._program[fun_addr].parents.add(pred)
                    self._program[pred].children.add(fun_addr)

    def __repr__(self):
        return "<Program:%s>" % self.name

    @property
    def callgraph(self) -> networkx.DiGraph:
        return self._callgraph
