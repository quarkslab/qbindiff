from __future__ import absolute_import
import networkx

from qbindiff.loader.function import Function
from qbindiff.loader.types import LoaderType, OperandType, FunctionType
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.operand import Operand

import ida_nalt
import idautils
import ida_funcs
import ida_gdl
import ida_idaapi
import ida_bytes
import ida_ua
import ida_lines


class OperandBackendIDA(object):
    def __init__(self, op_t, ea):
        self._addr = ea
        self.op_t = op_t

    @property
    def type(self):
        return OperandType(self.op_t.type)

    @property
    def expressions(self):
        blacklist = [
            "SCOLOR_ON",
            "SCOLOR_OFF",
            "SCOLOR_ESC",
            "SCOLOR_INV",
            "SCOLOR_UTF8",
            "SCOLOR_FG_MAX",
        ]
        tag_mapping = {
            getattr(ida_lines, x): x
            for x in dir(ida_lines)
            if (x.startswith("SCOLOR_") and x not in blacklist)
        }
        opnds = {"opnd1", "opnd2", "opnd3", "opnd4", "opnd5", "opnd6"}

        i = 0
        data = []  # {'type': XXX, 'value':XXX, childs
        b_opened = False
        raw = ida_ua.print_operand(self._addr, self.op_t.n)
        while i < len(raw):
            c = raw[i]
            if c == ida_lines.SCOLOR_ON:
                b_opened = True
                next_c = raw[i + 1]
                if next_c != ida_lines.SCOLOR_ADDR:
                    type = tag_mapping[next_c].split("_")[1].lower()
                    if type not in opnds:
                        if data:
                            yield data[-1]
                        data.append({"type": type})
                i += ida_lines.tag_skipcode(raw[i:])
            elif c == ida_lines.SCOLOR_OFF:
                b_opened = False
                i += 2
            else:
                if not data:  # for operand not in tags (yes it happens)
                    yield {"type": "unk", "value": c}
                    data.append({"type": "unk", "value": c})
                elif b_opened:
                    if "value" not in data[-1]:
                        data[-1]["value"] = ""
                    data[-1]["value"] += c
                else:  # if there is data and all brackets are closed (we ignore blank spaces)
                    pass
                i += 1
        for i in data:
            yield i

    def __str__(self):
        return "".join(y["value"] for y in self.expressions)


class InstructionBackendIDA(object):
    def __init__(self, addr):
        self._addr = addr
        self.insn = ida_ua.insn_t()
        ida_ua.decode_insn(self.insn, self.addr)
        self.nb_ops = self._nb_operand()

    def _nb_operand(self):
        i = 0
        while True:
            if not ida_ua.print_operand(self.addr, i):  # for either None or ''
                return i
            else:
                i += 1

    @property
    def addr(self):
        return self._addr

    @property
    def mnemonic(self):
        return ida_ua.ua_mnem(self.addr)

    @property
    def operands(self):
        return [
            Operand(LoaderType.ida, self.insn[i], self.addr) for i in range(self.nb_ops)
        ]

    @property
    def groups(self):
        return []  # Not implemented yet

    @property
    def comment(self):
        return ""  # TODO: Adding it to the export

    def __str__(self):
        return "%s %s" % (self.mnemonic, ", ".join((str(op) for op in self.operands)))


class FunctionBackendIDA(object):
    def __init__(self, fun, addr):
        self._function = fun
        self.addr = addr
        self.pfn = ida_funcs.get_func(self.addr)
        self._graph = networkx.DiGraph()
        self.parents = set()
        self.children = set()
        self._load_basic_blocks()

    def _load_basic_blocks(self):
        cfg = ida_gdl.FlowChart(self.pfn, flags=ida_gdl.FC_NOEXT)
        for idabb in cfg:  # First pass to create basic blocks
            bb = []
            cur_addr = idabb.start_ea
            while cur_addr != ida_idaapi.BADADDR:
                bb.append(Instruction(LoaderType.ida, cur_addr))
                cur_addr = ida_bytes.next_head(cur_addr, idabb.end_ea)
            self._function[idabb.start_ea] = bb
            self._graph.add_node(
                idabb.start_ea
            )  # also add the bb as attribute in the graph

        for idabb in cfg:  # Second pass to add edges
            for succs in idabb.succs():
                self._graph.add_edge(idabb.start_ea, succs.start_ea)
            for preds in idabb.preds():
                self._graph.add_edge(idabb.start_ea, preds.start_ea)

    @property
    def name(self):
        return ida_funcs.get_func_name(self.addr)

    @property
    def type(self):
        """
        We could have used lags & ida_funcs.FUNC_LIB etc, but
        as imports are not supported yet we just raise NotImplemented
        """
        raise NotImplementedError("function type not implemented for IDA backend")

    def is_import(self):
        raise NotImplementedError("is_import not implemented for IDA backend")

    @property
    def graph(self) -> networkx.DiGraph:
        return self._graph


class ProgramBackendIDA(object):
    def __init__(self, program):
        self._program = program
        self._load_functions()
        self._load_call_graph()
        self._graph = networkx.DiGraph()

    @property
    def name(self):
        return ida_nalt.get_root_filename()

    def _load_functions(self):
        for fun_addr in idautils.Functions():
            self._program[fun_addr] = Function(LoaderType.ida, fun_addr)

    def _load_call_graph(self):
        for fun_addr in self._program.keys():
            self._graph.add_node(fun_addr)
            for pred in idautils.CodeRefsTo(fun_addr, 1):
                f_pred = ida_funcs.get_func(pred)
                if f_pred:
                    pred = f_pred.start_ea
                    self._graph.add_edge(pred, fun_addr)
                    self._program[fun_addr].parents.add(pred)
                    self._program[pred].children.add(fun_addr)

    def __repr__(self):
        return "<Program:%s>" % self.name

    @property
    def callgraph(self) -> networkx.DiGraph:
        return self._graph
